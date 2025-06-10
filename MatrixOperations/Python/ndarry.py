import numpy as np
import cupy as cp
import time

array1 = np.random.rand(5_000,5_000)
array2 = np.random.rand(5_000, 5_000)
array1 = np.matrix(array1)
array2 = np.matrix(array2)

print("CPU / NumPy:")

start = time.perf_counter()
np_add = array1 + array2
end = time.perf_counter()
print(f"Time taken for addition: {end-start} s")

start = time.perf_counter()
np_mul = array1 * array2
end = time.perf_counter()
print(f"Time taken for multiplication: {end-start} s")

print("GPU / CuPy:")

array1 = cp.asarray(array1)
array2 = cp.asarray(array2)

cp.cuda.Device(0).synchronize() # Ensure GPU finishes previous work
start = time.perf_counter()
cp_add = array1 + array2
cp.cuda.Device(0).synchronize() # Ensure GPU finishes before stopping timer
end = time.perf_counter()
print(f"Time taken for addition: {end-start} s")

cp.cuda.Device(0).synchronize() # Ensure GPU finishes previous work
start = time.perf_counter()
cp_mul = array1 @ array2
cp.cuda.Device(0).synchronize() # Ensure GPU finishes before stopping timer
end = time.perf_counter()
print(f"Time taken for multiplication: {end-start} s")