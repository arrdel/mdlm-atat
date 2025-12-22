"""
MDLM-ATAT: Adaptive Time-Aware Token Masking for Masked Diffusion Language Models

This package extends the Masked Diffusion Language Models (MDLM) framework with
adaptive masking capabilities that learn token-level importance and adjust the
diffusion process accordingly.

Key Components:
- ImportanceEstimator: Learns token-level difficulty/importance
- AdaptiveMaskingScheduler: Adjusts masking probabilities based on importance
- CurriculumScheduler: Progressive learning from easy to hard tokens
- UncertaintyGuidedSampler: Confidence-based denoising at inference
- ATATDiT: DiT architecture enhanced with ATAT components
"""

from setuptools import setup, find_packages

# Read requirements
with open('requirements.txt') as f:
    requirements = [line.strip() for line in f if line.strip() and not line.startswith('#')]

# Read long description
with open('README.md', encoding='utf-8') as f:
    long_description = f.read()

setup(
    name='mdlm-atat',
    version='0.1.0',
    author='Your Name',
    author_email='your.email@example.com',
    description='Adaptive Time-Aware Token Masking for Masked Diffusion Language Models',
    long_description=long_description,
    long_description_content_type='text/markdown',
    url='https://github.com/yourusername/mdlm-atat',
    packages=find_packages(),
    classifiers=[
        'Development Status :: 3 - Alpha',
        'Intended Audience :: Science/Research',
        'License :: OSI Approved :: Apache Software License',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.9',
        'Programming Language :: Python :: 3.10',
        'Topic :: Scientific/Engineering :: Artificial Intelligence',
    ],
    python_requires='>=3.9',
    install_requires=requirements,
    extras_require={
        'dev': [
            'pytest>=7.0.0',
            'black>=22.0.0',
            'isort>=5.10.0',
            'flake8>=4.0.0',
        ],
        'flash': [
            'flash-attn>=2.8.0',
        ],
    },
    entry_points={
        'console_scripts': [
            'mdlm-atat-train=mdlm_atat.train:main',
            'mdlm-atat-eval=mdlm_atat.eval:main',
        ],
    },
)
