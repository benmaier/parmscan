from setuptools import setup, Extension
import setuptools
import os
import sys

# get __version__, __author__, and __email__
exec(open("./parmscan/metadata.py").read())

setup(
    name='parmscan',
    version=__version__,
    author=__author__,
    author_email=__email__,
    url='https://github.com/benmaier/parmscan',
    license=__license__,
    description="Provide tools to plot parameter scan results that are wrapped up in a high-dimensional numpy array",
    long_description='',
    packages=setuptools.find_packages(),
    python_requires='>=3.7',
    install_requires=[
                'numpy>=1.17',
                'matplotlib>=3.0.0',
                'bfmplot>=0.0.11',
    ],
    tests_require=['pytest', 'pytest-cov'],
    setup_requires=['pytest-runner'],
    classifiers=['License :: OSI Approved :: MIT License',
                 'Programming Language :: Python :: 3.7',
                 'Programming Language :: Python :: 3.8',
                 ],
    project_urls={
        'Documentation': 'http://parmscan.benmaier.org',
        'Contributing Statement': 'https://github.com/benmaier/parmscan/blob/master/CONTRIBUTING.md',
        'Bug Reports': 'https://github.com/benmaier/parmscan/issues',
        'Source': 'https://github.com/benmaier/parmscan/',
        'PyPI': 'https://pypi.org/project/parmscan/',
    },
    include_package_data=True,
    zip_safe=False,
)
