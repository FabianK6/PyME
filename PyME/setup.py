#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon May 18 23:08:41 2026

@author: fabian
"""

from setuptools import setup, find_packages

setup(
    name="PyME",
    version="0.1.0",
    description="Calculation of Machine elements according to Roloff Matek",
    author="Fabian Koch",
    packages=find_packages(),
    python_requires=">=3.12",
    install_requires=[
        # Hier deine Abhängigkeiten
        # "requests>=2.28",
        "matplotlib>=3.10.9",
        "numpy>=2.4.4",
        "sympy>=1.14.0"
    ],
)