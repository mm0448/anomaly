"""Setup script"""

from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="anomaly_detector",
    version="0.1.0",
    author="",
    description="Anomaly detection",
    long_description=long_description,
    long_description_content_type="text/markdown",
    packages=find_packages(),
    python_requires=">=3.7",
    install_requires=[
        "torch>=1.7.0",
        "torchvision>=0.8.0",
        "numpy>=1.19.0",
        "scipy>=1.5.0",
        "scikit-learn>=0.23.0",
        "scikit-image>=0.17.0",
        "matplotlib>=3.3.0",
        "Pillow>=8.0.0",
        "PyYAML>=5.3.0",
        "tqdm>=4.50.0",
    ],
    classifiers=[
        "Programming Language :: Python :: 3.9",
    ],
)
