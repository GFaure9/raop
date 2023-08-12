from setuptools import setup, find_packages

with open('requirements.txt') as f:
    requirements = f.read().splitlines()

VERSION = "0.0.1"
DESCRIPTION = "Description."
LONG_DESCRIPTION = "Long Description."

setup(
    name="raop",
    version=VERSION,
    author="Guilhem Fauré",
    author_email="<guilhemfaure@outlook.fr>",
    description=DESCRIPTION,
    long_description=LONG_DESCRIPTION,
    packages=find_packages(),
    install_requires=requirements,
    # classifiers=[
    #     "Development Status :: 3 - Alpha",
    #     "Intended Audience :: Financial and Insurance Industry",
    #     "Programming Language :: Python :: 3.10",
    #     "Operating System :: OS Independent",
    # ]
    # url='https://github.com/yourusername/package_name',
)
