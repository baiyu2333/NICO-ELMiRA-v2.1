#!/usr/bin/env python
from setuptools import find_packages, setup

setup(
    name="nicoaudio",
    version="1.0",
    packages=find_packages("scripts/"),
    package_dir={"": "scripts"},
    description="NICO api package for audio related modules",
    author="Connor Gaede",
    author_email="gaede@informatik.uni-hamburg.de",
    install_requires=[
        "audiotsm",
        "numpy>=1.21.6",
        "pydub",
        "pyaudio",
        "pyalsaaudio",
        "requests",
        "tts",
    ],
)
