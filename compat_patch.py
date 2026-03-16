import numpy as np
import inspect
import sys
import builtins

# 1. Patch inspect for Python 3.11+ (used by old chumpy/smplx)
if not hasattr(inspect, 'getargspec'):
    inspect.getargspec = inspect.getfullargspec

# 2. Patch numpy for NumPy 1.24+ (legacy aliases for chumpy)
# We must use NumPy's own types (bool_, int_, etc) instead of Python built-ins
# because internal numpy/pandas code (like numpy.ma) expects these to have 
# methods like .view().
mapping = [
    ('bool', 'bool_'),
    ('int', 'int_'),
    ('float', 'float_'),
    ('complex', 'complex_'),
    ('object', 'object_'),
    ('str', 'str_'),
    ('unicode', 'unicode_'), # Handle legacy unicode vs str
]

for legacy_name, current_name in mapping:
    if not hasattr(np, legacy_name):
        # Fallback to built-in only if numpy doesn't even have the '_' version
        target = getattr(np, current_name, getattr(builtins, legacy_name.replace('unicode', 'str'), None))
        if target:
            setattr(np, legacy_name, target)
            if 'numpy' in sys.modules:
                setattr(sys.modules['numpy'], legacy_name, target)

# 3. Handle specific chumpy requirements
if not hasattr(np, 'bool_'):
    np.bool_ = bool
