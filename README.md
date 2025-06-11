# Gaussian Elimination in Python

## Description

This project solves systems of linear equations using Gaussian Elimination and Back Substitution, implemented in Python with NumPy.

## Features

* Forward elimination to row echelon form
* Back substitution to find solutions
* Row swapping for pivoting
* Inconsistency detection

## Usage

1. Edit the augmented matrix `A` in the script:

```python
A = np.array([
    [ 2, -3,  2,  3],
    [ 1, -1, -2, -1],
    [-1,  2, -3, -4]
])
```

2. Run the script:

```bash
python gaussian_elimination.py
```

3. Output includes the reduced matrix and solution vector.

## Requirements

* Python 3.x
* NumPy

## License

Free to use and modify.
