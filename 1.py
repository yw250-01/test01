import numpy as np
from scipy import linalg
# Get number of dimensions
a = np.array([[1, 2], [3, 4]])
print(np.ndim(a))  # Or a.ndim - returns 2

# Get total number of elements
print(np.size(a))  # Or a.size - returns 4

# Get array shape
print(np.shape(a))  # Or a.shape - returns (2, 2)

# Access specific dimension size
print(a.shape[0])  # a.shape[n-1] - returns 2 (first dimension)

# Create 2D array
arr = np.array([[1., 2., 3.], [4., 5., 6.]])
print(arr)

# Create block matrix from sub-matrices
b = np.array([[7., 8.], [9., 10.]])
b = np.array([[7., 8.], [9., 10.]])
block_arr = np.block([[a, b], [b, a]])
print(block_arr)

# Array indexing examples
print(a[-1])      # Last row
print(arr[1, 1])  # Element at row 1, column 1 (0-based indexing)
print(arr[1])     # Or arr[1, :] - entire second row
print(arr[0:2])   # Or arr[:2] - first two rows
print(arr[-1:])   # Last row

# Advanced slicing
print(arr[0:2, 1:3])  # Rows 0-1, columns 1-2

# Indexed selection using np.ix_
print(arr[np.ix_([0, 1], [0, 2])])  # Rows 0,1 and columns 0,2

# Strided slicing
print(arr[::2, :])    # Every other row
print(arr[::-1, :])   # Reverse row order

# Transpose operations
print(a.transpose())  # Or a.T - transpose matrix
print(a.conj().transpose())  # Or a.conj().T - conjugate transpose

# Matrix multiplication and element-wise operations
print(a @ b)         # Matrix multiplication
print(a * b)         # Element-wise multiplication
print(a / b)         # Element-wise division

# Element-wise power
print(a**3)

# Boolean masking
mask = (a > 2)
print(mask)

# Find indices where condition is true
indices = np.nonzero(a > 2)
print(indices)

# Boolean indexing
print(a[a > 2])      # Extract elements > 2
print(a.T > 2)       # Transpose and compare

# Conditional assignment
a[a < 2] = 0         # Set elements < 2 to zero
print(a)

# Masked multiplication
print(a * (a > 2))   # Keep only elements > 2, others become 0

# Array copying and flattening
y = a.copy()         # Create deep copy
row_copy = a[1, :].copy()  # Copy specific row
flat_arr = a.flatten()     # Flatten to 1D array

# Create sequences
seq1 = np.arange(1., 11.)  # 1.0 to 10.0
seq2 = np.arange(10.)      # 0.0 to 9.0

# Add new axis for 2D
seq_2d = np.arange(1., 11.)[:, np.newaxis]

# Array initialization
zeros_arr = np.zeros((3, 4))        # 3x4 zeros
ones_arr = np.ones((3, 4))          # 3x4 ones
eye_arr = np.eye(3)                 # 3x3 identity matrix

# Diagonal operations
diag_elements = np.diag(a)          # Extract diagonal
diag_matrix = np.diag([1, 2, 3])    # Create diagonal matrix

# Random number generation
from numpy.random import default_rng
rng = default_rng(42)
random_arr = rng.random((3, 4))

# Linear spacing
lin_arr = np.linspace(1, 3, 4)  # 4 points between 1 and 3

# Grid creation
x, y = np.mgrid[0:9., 0:6.]      # Dense grids
x_sparse, y_sparse = np.ogrid[0:9., 0:6.]  # Open grids

# Custom grid points
xx, yy = np.meshgrid([1, 2, 4], [2, 4, 5])

# Array tiling and concatenation
tiled = np.tile(a, (2, 3))  # Repeat array 2x3 times

# Horizontal stacking
h_stack = np.concatenate((a, b), axis=1)  # Or np.hstack

# Vertical stacking
v_stack = np.concatenate((a, b), axis=0)  # Or np.vstack

# Maximum value
max_val = a.max()  # Or np.nanmax(a) ignoring NaN values

# Axis-wise operations
max_rows = a.max(axis=0)  # Maximum along rows (per column)
max_cols = a.max(axis=1)  # Maximum along columns (per row)

# Element-wise maximum
elementwise_max = np.maximum(a, b)

# Vector norm
v = np.array([3, 4, 5])
norm = np.sqrt(v @ v)  # Or np.linalg.norm(v)

# Logical operations
logical_and = np.logical_and(a > 1, a < 4)
logical_or = np.logical_or(a > 3, b < 2)

# Bitwise operations (for boolean arrays)
bitwise_and = (a > 1) & (a < 4)
bitwise_or = (a > 3) | (b < 2)

# Linear algebra
inv_a = np.linalg.inv(a)          # Matrix inverse
pinv_a = np.linalg.pinv(a)        # Pseudo-inverse
rank_a = np.linalg.matrix_rank(a) # Matrix rank

# Linear system solving
if a.shape[0] == a.shape[1]:  # If square matrix
    solution = np.linalg.solve(a, b)
else:                         # Otherwise use least squares
    solution = np.linalg.lstsq(a, b, rcond=None)[0]

# Solve transpose system
solution_T = np.linalg.solve(a.T, b.T).T

# Matrix decompositions
U, S, vh = np.linalg.svd(a)  # Singular Value Decomposition
V = vh.T

a = np.array([[4, 1], [1, 3]])
chol = np.linalg.cholesky(a)  # Cholesky decomposition (for positive definite)

# Eigen decomposition
D, V = np.linalg.eig(a)       # Eigenvalues and eigenvectors

# QR and LU decompositions
Q, R = np.linalg.qr(a)        # QR decomposition
P, L, U = linalg.lu(a)     # LU decomposition with permutation

# Fourier Transform
fft_result = np.fft.fft(a)    # Fast Fourier Transform
ifft_result = np.fft.ifft(a)  # Inverse FFT

# Sorting
sorted_arr = np.sort(a, axis=0)  # Sort along columns

# Sort along rows
sorted_rows = np.sort(a, axis=1)  # Sort each row

# Sort by specific column and reorder matrix
I = np.argsort(a[:, 0])  # Get indices that would sort first column
b = a[I, :]              # Reorder entire matrix based on sorted indices

# Least squares fitting
z = x + y
x = np.linalg.lstsq(z, y, rcond=None)[0]

# Signal processing (requires scipy.signal)
from scipy import signal
resampled = signal.resample(x, int(np.ceil(len(x)/2)))

# Unique elements
unique_vals = np.unique(a)  # Get sorted unique values

# Remove singleton dimensions
squeezed = a.squeeze()  # Remove dimensions of size 1