from .utils import multidim, singldim
import setuptools_dso
import ctypes as c
from numpy.ctypeslib import as_ctypes as c_array
from numpy.ctypeslib import as_array as np_array
import numpy as np

dso = setuptools_dso.find_dso('semi_markov.extension')
lib = c.CDLL(dso, c.RTLD_GLOBAL)

__all__ = ["viterbi","forward_backward"]

def viterbi(trans_mat: multidim,
            init_state: singldim, state_num: int,
            obs_probs: multidim, time_len: int,
            dur_probs: multidim, max_dur: int,
            out_state_seq: singldim
            ) -> singldim:

    buff_p_c = c.POINTER(c.c_double)
    out = np.array(out_state_seq).astype(float)
    buff_p = out.ctypes.data_as(buff_p_c)
    
    lib.viterbi(c_array(np.array(trans_mat,dtype=float)),
                c_array(np.array(init_state,dtype=float)), c.c_size_t(state_num),
                c_array(np.array(obs_probs,dtype=float)), c.c_size_t(time_len),
                c_array(np.array(dur_probs,dtype=float)), c.c_size_t(max_dur),
                buff_p
                )
    return out

def forward_backward():
    lib.forward_backward()