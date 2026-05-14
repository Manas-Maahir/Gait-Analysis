"""Entry point for 'python -m stride' command."""

from .cli import main
import sys

if __name__ == "__main__":
    sys.exit(main())
