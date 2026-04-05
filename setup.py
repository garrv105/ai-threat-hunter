from setuptools import find_packages, setup

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="ai-threat-hunter",
    version="1.0.0",
    author="Garrv Sipani",
    author_email="fgarrvs1@jh.edu",
    description="AI-Powered Network Threat Detection System",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/garrv105/ai-threat-hunter",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Science/Research",
        "Topic :: Security",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3.8",
    ],
    python_requires=">=3.8",
    install_requires=[
        "numpy>=1.24.3",
        "pandas>=2.0.3",
        "scikit-learn>=1.3.0",
        "pyyaml>=6.0.1",
    ],
    extras_require={
        "gpu": ["torch>=2.0.1"],
        "capture": ["scapy>=2.5.0"],
        "viz": ["matplotlib>=3.7.2", "seaborn>=0.12.2"],
        "full": [
            "torch>=2.0.1",
            "scapy>=2.5.0",
            "matplotlib>=3.7.2",
            "seaborn>=0.12.2",
            "tqdm>=4.65.0",
        ],
    },
)
