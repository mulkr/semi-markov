import numpy as np

singldim = np.ndarray[int,np.dtype[np.float64]] | list[float]
np_singldim = np.ndarray[int,np.dtype[np.float64]]

# shape typing is not well supported currently by numpy, but in the future we can change this to better reflect the multidimensionality
multidim = np.ndarray[tuple[int,int],np.dtype[np.float64]] | list[list[float]]
np_multidim = np.ndarray[tuple[int,int],np.dtype[np.float64]]

def TODO(message:str):
    raise NotImplementedError("[TODO]: "+message)