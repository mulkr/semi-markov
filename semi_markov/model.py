from .utils import singldim, multidim
from .emission import EmissionProb
import numpy as np
from numpy.random import Generator as rng

__all__ = ["HSMM"]

class HSMM():
    def __init__(self, emission: EmissionProb, transition: multidim, duration: multidim , init_dist: singldim) -> None:
        temp_transition = np.array(transition)
        temp_duration = np.array(duration)
        
        self.__state_num = temp_transition.shape[0]
        self.__duration_len = temp_duration.shape[1]
        
        self.emission = emission
        self.transition = transition
        self.duration = duration
        self.init_dist = init_dist
    
    @property
    def emission(self)->EmissionProb:
        return self.__emission
    @emission.setter
    def emission(self,other: EmissionProb)->None:
        if other.state_num() != self.__state_num:
            raise ValueError("State numbers defined by emission and transition matrix do not match")
        self.__emission = other

    @property
    def transition(self)->multidim:
        return self.__transition
    @transition.setter
    def transition(self,tmat: multidim)->None:
        temp_tmat = np.array(tmat)
        if self.emission is not None and self.emission.state_num() != temp_tmat.shape[0]:
            raise ValueError("State numbers defined by emission and transition matrix do not match")
        prob_sum = np.sum(temp_tmat,axis=1)
        if not np.all(np.isclose(np.ones_like(prob_sum),prob_sum)):
            raise ValueError("Transition matrix rows do not add up to one")
        self.__transition = temp_tmat
        
    @property
    def duration(self)->multidim:
        return self.__duration
    @duration.setter
    def duration(self,durs: multidim)->None:
        temp_dur = np.array(durs)
        if self.__state_num != temp_dur.shape[0]:
            raise ValueError("State numbers defined by duration distribution and transition matrix do not match")
        prob_sum = np.sum(temp_dur,axis=1)
        if not np.all(np.isclose(np.ones_like(prob_sum),prob_sum)):
            raise ValueError("Duration distribution rows do not add up to one")
        self.__duration = temp_dur
    
    @property
    def init_dist(self)->singldim:
        return self.__init_dist
    @init_dist.setter
    def init_dist(self,start:singldim)->None:
        temp_start = np.array(start)
        if self.__state_num != temp_start.shape[0]:
            raise ValueError("State numbers defined by initial distribution and transition matrix do not match")
        prob_sum = np.sum(temp_start)
        if not np.all(np.isclose(np.ones_like(prob_sum),prob_sum)):
            raise ValueError("Initial distribution does not add up to one")
        self.__init_dist = temp_start

    def decode(self):
        raise NotImplementedError("TODO, Viterbi algo")

    def sample(self, amount: int = 1, random: rng|None = None)-> tuple[singldim|float,singldim|int]:
        if random is None:
            random = np.random.default_rng()

        state = random.choice(self.__state_num, p=self.init_dist)
        if amount == 1:
            obs = self.emission.sample_state(state,generator=random)
            return obs, state

        sampled_states = np.empty(amount)
        sampled_obs = np.empty(amount)

        # Generate states
        state_idx = 0
        duration = random.choice(self.__duration_len, p=self.duration[state]) + 1
        while state_idx < amount:
            # Adjust for right censoring
            # (the last state may still be going on when we reach the limit on the number of samples to generate)
            if state_idx + duration > amount:
                duration = amount - state_idx

            sampled_states[state_idx:state_idx+duration] = state
            state_idx += duration

            state = random.choice(self.__state_num, p=self.transition[state])
            duration = random.choice(self.__duration_len, p=self.duration[state]) + 1

        # Generate observations
        for state in range(self.__state_num):
            state_mask = sampled_states == state
            sampled_obs[state_mask] = self.emission.sample_state(state, state_mask.sum(), random)

        return sampled_obs, sampled_states

    def fit(self):
        raise NotImplementedError("TODO, FB algo")

# del rng
# del np