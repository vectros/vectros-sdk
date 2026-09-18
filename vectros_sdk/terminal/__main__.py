"""Enables `python -m vectros_sdk.terminal`."""

import sys

from vectros_sdk.terminal.repl import main

if __name__ == "__main__":
    sys.exit(main())
