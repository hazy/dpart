from setuptools import setup, find_packages


with open("VERSION", "r") as f:
    version = f.read().strip()


setup(
    name="dpart",
    version=version,
    author="Sofiane Mahiou, Kai Xu, Georgi Ganev",
    author_email="info@hazy.com",
    description="dpart: General, flexible, and scalable framework for differentially private synthetic data generation",
    url="https://github.com/hazy/dpart",
    packages=find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.7",
)
