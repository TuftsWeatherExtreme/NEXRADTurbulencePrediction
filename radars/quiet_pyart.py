# quiet_pyart.py
# Team Celestial Blue
# Spring 2025
# Purpose: import this file to import pyart without the annoying print statement
#   that comes with it!

import sys
import os
from contextlib import contextmanager

@contextmanager
def quiet():
    sys.stdout = sys.stderr = open(os.devnull, "w")
    try:
        yield
    finally:
        sys.stdout = sys.__stdout__
        sys.stderr = sys.__stderr__

with quiet():
    import pyart

# Expose everything from pyart so it behaves like a normal import
globals().update({k: v for k, v in pyart.__dict__.items() if not k.startswith("_")})
