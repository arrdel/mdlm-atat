from setuptools import setup, find_packages

setup(
    name="mdlm_atat",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        "torch>=2.0.0",
        "pytorch-lightning>=2.0.0",
        "transformers>=4.30.0",
        "hydra-core>=1.3.0",
        "omegaconf>=2.3.0",
        "datasets>=2.14.0",
        "wandb>=0.15.0",
        "numpy>=1.24.0",
        "tqdm>=4.65.0",
    ],
    python_requires=">=3.8",
    description="MDLM with Adaptive Token-level Attention for Text Generation",
    author="Your Name",
    author_email="your.email@example.com",
)
