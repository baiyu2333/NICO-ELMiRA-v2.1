import sys
import os

try:
    from pyrep import PyRep  # required to prevent vrep launch issues
except (ImportError, SyntaxError):
    pass  # pyrep may not be available or compatible; not needed for real robot
