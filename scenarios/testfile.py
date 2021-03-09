print("hello")
import matplotlib
print("2")

import sys
from pathlib import Path # if you haven't already done so
file = Path(__file__).resolve()
print("2.0")
parent, root = file.parent, file.parents[1]
print("2.1")
sys.path.append(str(root))

# Additionally remove the current file's directory from sys.path
try:
        sys.path.remove(str(parent))
except ValueError: # Already removed
        pass


import neurolib
print("3")

from neurolib.models.aln import ALNModel

aln = ALNModel()
print(aln.params.duration)
