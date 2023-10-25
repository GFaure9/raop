from setuptools import setup, find_packages
import os

root_path = os.curdir

with open(f"{root_path}/requirements.txt") as f:
    requirements = f.read().splitlines()

VERSION = "0.0.15"
DESCRIPTION = "Warning: this package is in alpha stage of development. Use at your own risk."
# LONG_DESCRIPTION = ""

setup(
    name="raop",
    version=VERSION,
    author="GFaure9",
    author_email="",
    description=DESCRIPTION,
    # long_description=LONG_DESCRIPTION,
    packages=find_packages(),
    install_requires=requirements,
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Financial and Insurance Industry",
        "Programming Language :: Python :: 3.11",
        "Operating System :: OS Independent",
    ],
    url="https://github.com/GFaure9/raop",
)
