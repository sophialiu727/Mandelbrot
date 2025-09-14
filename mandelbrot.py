import numpy as np


def get_escape_time(c: complex, max_iterations: int) -> int | None:
    """takes complex number and sets z value to that. loops through till max iteration
    and applies mandelbrot calculation to it. if calculated z value is greater than 2, then returns
    the number of iterations, which is i. returns None if never exceeds 2 after iterating
    the maximum number of times"""
    z = 0 + 0j #notes z is a complex number
    for i in range(max_iterations+1): #looping through max times
        z = z*z + c # calculation for mandelbrot set
        if abs(z) > 2: # if the z value is greater than two, then set will go to infinity
            return i # therefore returns the i value, which is the number of iterations
    return None # returns none for complex numbers that don't exceed 2 after maximum iterations

# print(get_escape_time(2+1j, 5))
# print(get_escape_time(1+1j, 10))
# print(get_escape_time(0.5+0.5j, 2))
# print(get_escape_time(0.5+0.5j, 4))
# print(get_escape_time(0.38+0.25j, 100))

def get_escape_time_color_arr(
    c_arr: np.ndarray,
    max_iterations: int
) -> np.ndarray:
    """This function allows for the coloring of the Mendalbrot set.
    Thus the calculation for the actual Mendalbrot set can be given pixel declartions.
    The zeros and ones in the calculation are given a numpy color
    it goes through the entire equation until its fully printed."""
    z= np.zeros_like(c_arr, dtype=np.complex128)
    escape_time =np.full(c_arr.shape, max_iterations+1, dtype=np.int32)
    mask=np.ones(c_arr.shape, dtype=np.bool)
    for i in range(1, max_iterations+1):
        z[mask]=z[mask]**2+c_arr[mask]
        escape=np.abs(z)>2
        escape_time[mask * escape]=i
        mask= mask*(~escape)
    color_arr = (max_iterations-escape_time+1)/(max_iterations+1)
    return color_arr

def get_julia_color_arr(c_arr: np.ndarray, c: complex, max_iterations: int) -> np.ndarray:
    """
    Computes julia color array for grid of complex numbers

    Parameters:
    ----------
    c_arr : np.ndarray -> 2D array of complex numbers
    c : complex number -> Julia set parameter
    max_iterations : int -> Max number of iterations

    Returns:
    -------
    np.ndarray -> 2D array of julia color arrays
    """
    z = np.array(c_arr, dtype=np.complex128)
    escape_time = np.full(c_arr.shape, max_iterations + 1, dtype=np.int32)
    mask = np.ones(c_arr.shape, dtype=np.bool) # ensure proper boolean array

    for i in range(1, max_iterations + 1):
        z[mask] = z[mask] ** 2 + c # julia iteration
        escape=np.abs(z) > max(abs(c), 2) # escape condition
        escape_time[mask * escape] = i # store escape time
        mask = mask * (~escape) # update mask

    color_arr = (max_iterations - escape_time + 1) / (max_iterations + 1) # changes escape times to grey scale
    return color_arr

def get_complex_grid(top_left: complex, bottom_right: complex, step: float) -> np.ndarray:
    """creates a 2D grid with evenly spaced lines. creates array of zeros and begins finding
    complex values for the grid.

    top_left : complex -> 2D numpy array of complex numbers for top left corner
    bottom_right : complex -> 2D numpy array of complex numbers for bottom right corner
    step : float -> Step size for mandelbrot computation"""

    real_part = np.arange(top_left.real, bottom_right.real, step)
    imaginary_part = np.arange(top_left.imag, bottom_right.imag, step * -1)  # taking negative step to decrease

    created_grid = np.zeros((len(imaginary_part), len(real_part)), dtype=complex)  # creates grid of zeros

    for i in range(len(imaginary_part)):
        for j in range(len(real_part)):
            created_grid[i, j] = real_part[j] + 1j * imaginary_part[i]  # finds complex values for corresponding position in grid

    return created_grid





