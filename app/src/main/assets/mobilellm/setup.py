from setuptools import setup, find_packages

setup(
    name="mobilellm",
    version="1.0.0",
    description="Run 30B+ LLMs on 8 GB RAM mobile devices via layer-sharding",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    author="MobileAirLLM",
    license="Apache-2.0",
    packages=find_packages(),
    python_requires=">=3.9",
    install_requires=[
        "torch>=2.0.0",
        "transformers>=4.35.0",
        "safetensors>=0.4.0",
        "huggingface_hub>=0.20.0",
        "psutil>=5.9.0",
        "sentencepiece>=0.1.99",
    ],
    extras_require={
        "quantization": ["bitsandbytes>=0.41.0"],
        "fast": ["accelerate>=0.24.0"],
    },
    entry_points={
        "console_scripts": [
            "mobilellm=mobilellm.cli:main",
        ]
    },
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: Apache Software License",
        "Operating System :: OS Independent",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
)
