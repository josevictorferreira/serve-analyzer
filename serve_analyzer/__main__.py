"""Allow running CLI as python -m serve_analyzer"""
from .cli import main
import sys

if __name__ == '__main__':
    sys.exit(main())
