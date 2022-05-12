from setuptools import setup, find_packages


with open("VERSION", "r") as f:
    version = f.read().strip()

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

with open("requirements.txt", "r") as fr:
    requirements = fr.read().splitlines()


setup(
    name="dpart",
    version=version,
    author="Sofiane Mahiou, Kai Xu, Georgi Ganev",
    author_email="info@hazy.com",
    description="dpart: General, flexible, and scalable framework for differentially private synthetic data generation",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/hazy/dpart",
    packages=find_packages("."),
    package_dir={"": "."},
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    project_urls={"Bug Tracker": "https://github.com/hazy/dpart/issues"},
    python_requires=">=3.7",
    install_requires=requirements,
)
