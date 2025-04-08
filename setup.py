#!/usr/bin/env python
"""
Setup script for the HydroAnalysis package.
"""

from setuptools import setup, find_packages

# Read long description from README
with open('README.md', 'r', encoding='utf-8') as f:
    long_description = f.read()

setup(
    name='hydroanalysis',
    version='0.1.0',
    description='A comprehensive Python package for hydrological data analysis',
    long_description=long_description,
    long_description_content_type='text/markdown',
    author='Your Name',
    author_email='your.email@example.com',
    url='https://github.com/username/hydroanalysis',
    packages=find_packages(),
    include_package_data=True,
    classifiers=[
        'Development Status :: 3 - Alpha',
        'Intended Audience :: Science/Research',
        'Topic :: Scientific/Engineering :: Hydrology',
        'License :: OSI Approved :: MIT License',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.8',
        'Programming Language :: Python :: 3.9',
        'Programming Language :: Python :: 3.10',
        'Programming Language :: Python :: 3.11',
        'Programming Language :: Python :: 3.12',
    ],
    python_requires='>=3.8',
    install_requires=[
        'numpy>=1.19.0',
        'pandas>=1.1.0',
        'matplotlib>=3.3.0',
        'scikit-learn>=0.24.0',
        'openpyxl>=3.0.0',
    ],
    extras_require={
        'dev': [
            'pytest>=6.0.0',
            'pytest-cov>=2.10.0',
            'flake8>=3.8.0',
            'black>=20.8b1',
            'sphinx>=3.0.0',
            'sphinx-rtd-theme>=0.5.0',
        ],
        'gee': ['earthengine-api>=0.1.300'],
        'netcdf': ['netCDF4>=1.5.0'],
    },
    entry_points={
        'console_scripts': [
            'hydroanalysis=hydroanalysis.cli:main',
        ],
    },
)