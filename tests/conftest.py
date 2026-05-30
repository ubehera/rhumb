import pathlib
import sys

# runners/ is not a package; put it on sys.path so tests can import the lib.
RUNNERS = pathlib.Path(__file__).resolve().parent.parent / "runners"
sys.path.insert(0, str(RUNNERS))
