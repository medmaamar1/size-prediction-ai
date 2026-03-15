import numpy as np
import inspect
import builtins
import sys

# 1. Patch inspect for Python 3.11+
if not hasattr(inspect, 'getargspec'):
    inspect.getargspec = inspect.getfullargspec

# 2. Patch numpy for NumPy 1.24+ (legacy aliases for chumpy)
# We must patch both the imported 'np' and the entry in sys.modules
for name, target in [('bool', bool), ('int', int), ('float', float), 
                     ('complex', complex), ('object', object), 
                     ('unicode', str), ('str', str)]:
    if not hasattr(np, name):
        setattr(np, name, target)
    if 'numpy' in sys.modules:
        setattr(sys.modules['numpy'], name, target)

# 3. Handle specific chumpy requirements
# Some versions of chumpy specifically look for numpy.bool
if not hasattr(np, 'bool_'):
    np.bool_ = bool
