
"""
Performance Analysis of Matrix Multiplication
This script compares different matrix multiplication methods
and analyzes their execution time.
"""

import numpy as np
import time
import matplotlib.pyplot as plt

# Strassen algorithm
def strassen(A, B):
    n = A.shape[0]
    
    if n <= 64:  # For small matrices, normal multiplication is faster
        return A @ B
    
    # If n is odd, pad matrices to the next even size
    if n % 2 != 0:
        A = np.pad(A, ((0,1),(0,1)))
        B = np.pad(B, ((0,1),(0,1)))
        n += 1
    
    mid = n // 2
    A11, A12 = A[:mid, :mid], A[:mid, mid:]
    A21, A22 = A[mid:, :mid], A[mid:, mid:]
    B11, B12 = B[:mid, :mid], B[:mid, mid:]
    B21, B22 = B[mid:, :mid], B[mid:, mid:]
    
    # Compute M1 to M7
    M1 = strassen(A11 + A22, B11 + B22)
    M2 = strassen(A21 + A22, B11)
    M3 = strassen(A11, B12 - B22)
    M4 = strassen(A22, B21 - B11)
    M5 = strassen(A11 + A12, B22)
    M6 = strassen(A21 - A11, B11 + B12)
    M7 = strassen(A12 - A22, B21 + B22)
    
    # Combine submatrices
    C11 = M1 + M4 - M5 + M7
    C12 = M3 + M5
    C21 = M2 + M4
    C22 = M1 - M2 + M3 + M6
    
    # Merge
    C = np.zeros((n, n))
    C[:mid, :mid] = C11
    C[:mid, mid:] = C12
    C[mid:, :mid] = C21
    C[mid:, mid:] = C22
    
    # Remove padding if added
    return C[:A11.shape[0]*2 if n>len(A11)*2 else n, :B11.shape[1]*2 if n>len(B11)*2 else n]

# Matrix sizes (powers of 2 are better for Strassen)
sizes = [1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048]

times_normal = []
times_strassen = []

for n in sizes:
    A = np.random.rand(n, n)
    B = np.random.rand(n, n)
    
    # Normal multiplication
    start = time.perf_counter()
    _ = A @ B
    end = time.perf_counter()
    times_normal.append(end - start)
    
    # Strassen multiplication
    start = time.perf_counter()
    _ = strassen(A, B)
    end = time.perf_counter()
    times_strassen.append(end - start)
    
    # Print results
    print(f"Matrix size {n}×{n}:")
    print(f"  Normal multiplication time: {times_normal[-1]:.4f} seconds")
    print(f"  Strassen multiplication time: {times_strassen[-1]:.4f} seconds\n")

# Plot comparison
plt.figure(figsize=(8,5))
plt.plot(sizes, times_normal, marker='o', label="Normal Multiplication")
plt.plot(sizes, times_strassen, marker='s', label="Strassen")
plt.xlabel("Matrix size (n × n)")
plt.ylabel("Execution time (seconds)")
plt.title("Matrix Multiplication Comparison: Normal vs Strassen")
plt.legend()
plt.grid(True)
plt.show()
