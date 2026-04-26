"""
MobileAirLLM HTTP Server
Runs a local REST + Server-Sent Events (SSE) server so the Android app
can talk to the Python inference engine via HTTP on localhost.

Endpoints:
  POST /load         – load a model (sharding + engine init)
  POST /generate     – full (non-streaming) generation
  GET  /stream       – SSE streaming generation
  POST /reset        – clear KV cache
  GET  /status       – RAM & engine status
  GET  /health       – heartbeat

Android app connects to http://localhost:8765
"""

import json
import time
import logging
import threading
import traceback
from http.server import HTTPServer, BaseHTTPRequestHandler
from urllib.parse import urlparse, parse_qs
from typing import Optional

logger = logging.getLogger(__name__)

DEFAULT_PORT = 8765
_engine = None          # global MobileAirLLM AutoModel instance
_loading_lock = threading.Lock()
_status = {"state": "idle", "model": None, "error": None, "progress": 0}


# ──────────────────────────────────────────────────────────────────────────────
# Request handler
# ──────────────────────────────────────────────────────────────────────────────
class LLMHandler(BaseHTTPRequestHandler):

    def log_message(self, format, *args):
        logger.debug(f"[HTTP] {self.address_string()} {format % args}")

    # ── Routing ──────────────────────────────────────────────────────────────
    def do_GET(self):
        path = urlparse(self.path).path
        if path == "/health":
            self._json(200, {"status": "ok", "time": time.time()})
        elif path == "/status":
            self._json(200, _status)
        elif path == "/stream":
            self._handle_stream()
        else:
            self._json(404, {"error": "not found"})

    def do_POST(self):
        path = urlparse(self.path).path
        body = self._read_body()
        if path == "/load":
            self._handle_load(body)
        elif path == "/generate":
            self._handle_generate(body)
        elif path == "/reset":
            self._handle_reset()
        elif path == "/unload":
            self._handle_unload()
        else:
            self._json(404, {"error": "not found"})

    def do_OPTIONS(self):
        self._send_cors()
        self.end_headers()

    # ── Handlers ─────────────────────────────────────────────────────────────
    def _handle_load(self, body: dict):
        global _engine, _status
        model_path = body.get("model_path")
        if not model_path:
            self._json(400, {"error": "model_path required"})
            return

        if _loading_lock.locked():
            self._json(409, {"error": "model already loading"})
            return

        def _load():
            global _engine
            _status["state"] = "loading"
            _status["model"] = model_path
            _status["error"] = None
            _status["progress"] = 0
            try:
                from .auto_model import AutoModel
                _engine = AutoModel.from_pretrained(
                    model_path=model_path,
                    shard_dir=body.get("shard_dir"),
                    compression=body.get("compression", "4bit"),
                    max_ram_gb=body.get("max_ram_gb"),
                    device_profile=body.get("device_profile"),
                    prefetch=body.get("prefetch"),
                    max_kv_tokens=body.get("max_kv_tokens"),
                    delete_original=body.get("delete_original", False),
                    hf_token=body.get("hf_token"),
                )
                _status["state"] = "ready"
                _status["progress"] = 100
                logger.info(f"[Server] Model loaded: {model_path}")
            except Exception as e:
                _status["state"] = "error"
                _status["error"] = str(e)
                logger.error(f"[Server] Load error: {e}\n{traceback.format_exc()}")

        with _loading_lock:
            t = threading.Thread(target=_load, daemon=True)
            t.start()

        self._json(202, {"status": "loading started", "model": model_path})

    def _handle_generate(self, body: dict):
        global _engine
        if _engine is None:
            self._json(503, {"error": "no model loaded"})
            return
        prompt = body.get("prompt", "")
        kwargs = _gen_kwargs(body)
        try:
            result = _engine.generate(prompt, **kwargs)
            self._json(200, {"text": result, "prompt": prompt})
        except Exception as e:
            self._json(500, {"error": str(e)})

    def _handle_stream(self):
        global _engine
        if _engine is None:
            self._json(503, {"error": "no model loaded"})
            return

        qs = parse_qs(urlparse(self.path).query)
        prompt = qs.get("prompt", [""])[0]
        body = {
            "max_new_tokens": int(qs.get("max_new_tokens", [256])[0]),
            "temperature": float(qs.get("temperature", [0.7])[0]),
            "top_p": float(qs.get("top_p", [0.9])[0]),
            "top_k": int(qs.get("top_k", [40])[0]),
            "reset_cache": qs.get("reset_cache", ["true"])[0].lower() == "true",
        }
        kwargs = _gen_kwargs(body)

        # SSE headers
        self.send_response(200)
        self._send_cors()
        self.send_header("Content-Type", "text/event-stream")
        self.send_header("Cache-Control", "no-cache")
        self.send_header("X-Accel-Buffering", "no")
        self.end_headers()

        try:
            for chunk in _engine.stream(prompt, **kwargs):
                payload = json.dumps({"token": chunk})
                self.wfile.write(f"data: {payload}\n\n".encode())
                self.wfile.flush()
            self.wfile.write(b"data: [DONE]\n\n")
            self.wfile.flush()
        except BrokenPipeError:
            pass
        except Exception as e:
            err = json.dumps({"error": str(e)})
            self.wfile.write(f"data: {err}\n\n".encode())

    def _handle_reset(self):
        global _engine
        if _engine:
            _engine.reset()
        self._json(200, {"status": "cache cleared"})

    def _handle_unload(self):
        global _engine, _status
        _engine = None
        _status = {"state": "idle", "model": None, "error": None, "progress": 0}
        import gc; gc.collect()
        self._json(200, {"status": "unloaded"})

    # ── Util ─────────────────────────────────────────────────────────────────
    def _json(self, code: int, data: dict):
        body = json.dumps(data).encode()
        self.send_response(code)
        self._send_cors()
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _send_cors(self):
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")

    def _read_body(self) -> dict:
        length = int(self.headers.get("Content-Length", 0))
        if length == 0:
            return {}
        try:
            return json.loads(self.rfile.read(length))
        except Exception:
            return {}


def _gen_kwargs(body: dict) -> dict:
    out = {}
    for k in ("max_new_tokens", "temperature", "top_p", "top_k",
              "repetition_penalty", "reset_cache"):
        if k in body:
            out[k] = body[k]
    return out


# ──────────────────────────────────────────────────────────────────────────────
# Start server
# ──────────────────────────────────────────────────────────────────────────────
def run_server(host: str = "127.0.0.1", port: int = DEFAULT_PORT):
    """Start the blocking HTTP server."""
    server = HTTPServer((host, port), LLMHandler)
    logger.info(f"[MobileAirLLM Server] Listening on http://{host}:{port}")
    print(f"\n🚀 MobileAirLLM Server running at http://{host}:{port}")
    print("   Endpoints: /load  /generate  /stream  /reset  /status  /health")
    print("   Press Ctrl+C to stop.\n")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\n[Server] Shutting down.")
        server.shutdown()
