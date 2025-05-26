from .utils import multidim, singldim
import setuptools_dso
import ctypes as c
from numpy.ctypeslib import as_ctypes as c_array
from numpy.ctypeslib import as_array as np_array
import numpy as np

dso = setuptools_dso.find_dso('semi_markov.extension')
lib = c.CDLL(dso, c.RTLD_GLOBAL)

__all__ = ["viterbi","forward_backward"]

def viterbi(init_state: singldim, state_num: int,
            obs_probs: multidim, time_len: int,
            dur_probs: multidim, max_dur: int,
            out_state_seq: singldim) -> singldim:

    lib.viterbi(c_array(np.array(init_state,dtype=float)), c.c_int(state_num),
                c_array(np.array(obs_probs,dtype=float)), c.c_int(time_len),
                c_array(np.array(dur_probs,dtype=float)), c.c_int(max_dur),
                c_array(np.array(out_state_seq,dtype=float)))
    return np.array(np_array(out_state_seq),dtype=float)

def forward_backward():
    lib.forward_backward()