"""setup."""
import os
import setuptools
try:  # for pip >= 10
    from pip._internal.req import parse_requirements
except ImportError:  # for pip <= 9.0.3
    from pip.req import parse_requirements


with open(os.path.join(os.path.dirname(__file__), 'README.md')) as readme:
    README = readme.read()

requirements_path = os.path.join(
    os.path.dirname(__file__), 'requirements.txt')
install_reqs = parse_requirements(requirements_path, session=False)
try:
    requirements = [str(ir.req) for ir in install_reqs]
except Exception:
    requirements = [str(ir.requirement) for ir in install_reqs]


# allow setup.py to be run from any path
os.chdir(os.path.normpath(os.path.join(os.path.abspath(__file__), os.pardir)))

setuptools.setup(
    name='pumpwood_models',
    version='0.0',
    include_package_data=True,
    license='BSD-3-Clause License',
    description='Package for creation of mathematical model on Pumpwood',
    long_description=README,
    long_description_content_type="text/markdown",
    url='',
    author='André Andrade Baceti',
    author_email='a.baceti@murabei.com',
    classifiers=[
    ],
    install_requires=['requests',
                      'simplejson',
                      'pandas',
                      "pumpwood-communication>=0.25",
                      "pumpwood-miscellaneous>=0.18",
                      "pyarrow==3.0.0"],
    dependency_links=[]
)
