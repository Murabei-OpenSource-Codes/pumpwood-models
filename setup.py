"""module setup."""

import os
from setuptools import find_packages, setup

with open(os.path.join(os.path.dirname(__file__), 'README.rst')) as readme:
    README = readme.read()

# allow setup.py to be run from any path
os.chdir(os.path.normpath(os.path.join(os.path.abspath(__file__), os.pardir)))

setup(
    name='pumpwood_models',
    version='0.30',
    packages=find_packages(),
    include_package_data=True,
    license='',  # example license
    description='Package for creation of mathematical model on Pumpwood',
    long_description=README,
    url='https://github.com/Murabei-OpenSource-Codes/pumpwood-models',
    author='Murabei Data Science',
    author_email='a.baceti@murabei.com',
    classifiers=[
        "Programming Language :: Python :: 3",
        "Operating System :: OS Independent",
    ],
    package_dir={"": "src"},
    install_requires=requirements,
    packages=setuptools.find_packages(where="src"),
    python_requires=">=3.6",
)
