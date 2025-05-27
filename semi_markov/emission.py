from __future__ import annotations # support for lazy annotations with python<3.14
from abc import abstractmethod
import numpy as np
import numpy.typing as npt
from numpy.random import Generator as rng
from scipy.stats import norm
from typing_extensions import override
from .utils import singldim, multidim, np_singldim, np_multidim

__all__ = ["EmissionProb","GaussianProb"]

class EmissionProb():
    
    @abstractmethod
    def sample_state(self, state: int, amount:int = 1, generator: rng|None = None) -> npt.NDArray[np.float64]|float:
        raise NotImplementedError("Abstract method not implemented")
    @abstractmethod
    def update(self, prob: multidim, obs: singldim) -> None:
        raise NotImplementedError("Abstract method not implemented")
    @abstractmethod
    def likelihood(self, observations: float|singldim) -> npt.NDArray[np.float64]:
        raise NotImplementedError("Abstract method not implemented")
    @abstractmethod
    def copy(self) -> EmissionProb:
        raise NotImplementedError("Abstract method not implemented")
    @abstractmethod
    def state_num(self)->int:
        raise NotImplementedError("Abstract method not implemented")

class GaussianProb(EmissionProb):
    def __init__(self, mean: float|singldim|multidim, std: float|singldim|multidim) -> None:
        temp_mean = np.array(mean)
        temp_std = np.array(std)
        if temp_mean.shape != temp_std.shape:
            raise ValueError("Emission mean and std shapes do not match")
        self.__mean = temp_mean
        self.__std = temp_std

    @property
    def mean(self) -> np_singldim|np_multidim:
        return self.__mean
    @mean.setter
    def mean(self,mean: float|singldim|multidim) -> None:
        temp_mean = np.array(mean)
        if self.std is not None and temp_mean.shape != self.std.shape:
            raise ValueError("Emission mean and std shapes do not match")
        self.__mean = np.array(mean)
    @property
    def std(self) -> np_singldim|np_multidim:
        return self.__std
    @std.setter
    def std(self,std: float|singldim|multidim) -> None:
        if self.mean is not None and self.mean.shape != self.std.shape:
            raise ValueError("Emission mean and std shapes do not match")
        self.__std = np.array(std)

    @override
    def likelihood(self, observations: float|singldim) -> npt.NDArray[np.float64]:
        obs = np.squeeze(np.array(observations))
        return norm.pdf(obs,loc=self.mean[:,np.newaxis],scale=self.std[:,np.newaxis]) #type: ignore

    @override
    def sample_state(self, state: int, amount:int = 1, generator: rng|None = None) -> npt.NDArray[np.float64]|float:
        return norm.rvs(self.mean[state], self.std[state], amount, random_state=generator) #type: ignore

    @override
    def copy(self) -> EmissionProb:
        return GaussianProb(self.mean.copy(),self.std.copy())

    @override
    def update(self, prob: multidim, obs: singldim) -> None:
        re_obs = np.array(obs)[np.newaxis,:]
        prob_sum = np.sum(prob,axis=1)
        self.mean = np.sum(prob*re_obs,axis=1)/prob_sum
        self.std = np.sqrt(np.sum(np.square(re_obs-self.mean[:,np.newaxis])*prob,axis=1)/prob_sum) #TODO: rewrite this?
    
    @override
    def state_num(self) -> int:
        return self.mean.shape[0]

#TODO: add more built in emission probabilities

# del abstractmethod
# del annotations
# del norm
# del np
# del npt
# del override