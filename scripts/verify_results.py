"""Verify the reported metrics and checkpoint selection using tracked artifacts.

No feature downloads or executed notebook outputs are required. Pass --recompute
to also run inference; options match research_extensions.evaluate_extension.
"""
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from research_extensions.evaluate_extension import main


if __name__ == '__main__':
    main()
