from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="ai-threat-hunter",
    version="1.0.0",
    author="Your Name",
    author_email="your.email@university.edu",
    description="AI-Powered Network Threat Detection System",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/YOUR_USERNAME/ai-threat-hunter",
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
        "torch>=2.0.1",
        "matplotlib>=3.7.2",
        "seaborn>=0.12.2",
        "pyyaml>=6.0.1",
        "tqdm>=4.65.0",
    ],
)