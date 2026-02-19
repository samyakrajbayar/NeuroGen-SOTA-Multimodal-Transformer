#!/usr/bin/env python3
"""
Setup script for Air Writing System
"""

from setuptools import setup, find_packages
import os

# Read README
with open("README.md", "r", encoding="utf-8") as f:
    long_description = f.read()

# Read requirements
with open("requirements.txt", "r", encoding="utf-8") as f:
    requirements = [line.strip() for line in f if line.strip() and not line.startswith("#")]

setup(
    name="air-writing-system",
    version="2.0.0",
    author="AI Assistant",
    author_email="",
    description="Advanced AI-powered Air Writing System using hand gestures",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/yourusername/air-writing-system",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "Intended Audience :: Education",
        "Intended Audience :: End Users/Desktop",
        "Topic :: Education",
        "Topic :: Multimedia :: Graphics :: Capture",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Scientific/Engineering :: Image Recognition",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.8",
    install_requires=requirements,
    extras_require={
        "ai": ["tensorflow>=2.8.0", "keras>=2.8.0"],
        "full": [
            "tensorflow>=2.8.0",
            "keras>=2.8.0",
            "scikit-learn>=0.24.0",
            "scikit-image>=0.18.0",
            "matplotlib>=3.3.0",
        ],
    },
    entry_points={
        "console_scripts": [
            "air-writing=air_writing_advanced:main",
            "air-writing-launch=launch:main",
        ],
    },
    include_package_data=True,
    package_data={
        "": ["*.yaml", "*.json", "*.md"],
    },
    project_urls={
        "Bug Reports": "https://github.com/yourusername/air-writing-system/issues",
        "Source": "https://github.com/yourusername/air-writing-system",
        "Documentation": "https://github.com/yourusername/air-writing-system/wiki",
    },
)
