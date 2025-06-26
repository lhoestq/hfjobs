#!/usr/bin/env python3
# /// script
# requires-python = ">=3.8"
# dependencies = [
#     "cowsay",
# ]
# ///
"""A simple UV script example for hfjobs.

This script demonstrates how UV scripts can specify their dependencies
inline, making them perfect for running with hfjobs.
"""

import cowsay
import sys


def main():
    message = " ".join(sys.argv[1:]) if len(sys.argv) > 1 else "Hello from hfjobs!"
    cowsay.cow(message)


if __name__ == "__main__":
    main()
