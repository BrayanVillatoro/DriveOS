"""
DriveOS Setup Script
Creates a Windows installer with automatic dependency installation
"""

from setuptools import setup, find_packages
import sys

# Read requirements
with open('requirements.txt', 'r') as f:
    requirements = [line.strip() for line in f if line.strip() and not line.startswith('#')]

setup(
    name='DriveOS',
    version='1.0.0',
    author='BrayanVillatoro',
    description='AI-powered racing line analysis for optimal track performance',
    long_description=open('README.md').read(),
    long_description_content_type='text/markdown',
    url='https://github.com/BrayanVillatoro/DriveOS',
    packages=find_packages(),
    install_requires=requirements,
    entry_points={
        'console_scripts': [
            'driveos=src.gui:launch_gui',
        ],
    },
    python_requires='>=3.9,<3.12',
    classifiers=[
        'Development Status :: 4 - Beta',
        'Intended Audience :: Developers',
        'Intended Audience :: End Users/Desktop',
        'Topic :: Scientific/Engineering :: Artificial Intelligence',
        'Topic :: Multimedia :: Video :: Analysis',
        'License :: OSI Approved :: MIT License',
        'Programming Language :: Python :: 3.9',
        'Programming Language :: Python :: 3.10',
        'Programming Language :: Python :: 3.11',
        'Operating System :: Microsoft :: Windows',
    ],
    include_package_data=True,
)
