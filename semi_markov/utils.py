import numpy as np

np_singldim: type[np.ndarray[int, np.dtype[np.float64]]] = np.ndarray[int,np.dtype[np.float64]]
singldim: type[list[float] | np.ndarray[int, np.dtype[np.float64]]] = np_singldim | list[float]

# shape typing is not well supported currently by numpy, but in the future we can change this to better reflect the multidimensionality
np_multidim: type[np.ndarray[tuple[int, int], np.dtype[np.float64]]] = np.ndarray[tuple[int,int],np.dtype[np.float64]]
multidim: type[list[list[float]] | np.ndarray[tuple[int, int], np.dtype[np.float64]]] = np_multidim | list[list[float]]

def TODO(message:str):
    raise NotImplementedError("[TODO]: "+message)