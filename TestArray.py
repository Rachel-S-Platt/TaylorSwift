import numpy as np

array = np.array([-500.0, 30.0, -340.0, 500.0, 700.0, -20.0])
min_val = np.min(array)

array = array + min_val
print(array)

array = array / np.sum(array)

print(array)
