from setuptools import setup, find_packages

setup(
    name="dpp-via-pytorch",
    version="0.1.0",
    description="Differential Dynamic Programming (DDP) implemented with PyTorch",
    author="Xilu Zeng",
    packages=find_packages(),
    install_requires=[
        "torch",
        "matplotlib",
        "numpy",
        "pillow"
    ],
    python_requires=">=3.8",
)
