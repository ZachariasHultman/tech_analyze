import sys
import os

sys.path.insert(0, os.path.dirname(__file__))

from analyzer.main import main

if __name__ == "__main__":
    sys.exit(main())
