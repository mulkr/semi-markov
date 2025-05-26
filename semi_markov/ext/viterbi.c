/*
A -> a_ij:     transition matrix and elements
B -> b_j(O_t): probability that state j generates observation O at time t
N:             number of states
T:             time length
pi:            initial distribution
pi_i:          probability of being in state i
delta_t(j):    likelihood of most probable state seq until time t and ends in state j
d_max:         maximum time expected to stay in a state
q_t:           state at t
qstar_t:       most likely state at t
psi_t:         most likely previous state at t
D_t:           maximum delta at t (?)
*/
#include <float.h>
#include <stdlib.h>
#include <math.h>
#include "utils.h"

#ifdef _WIN32
__declspec(dllexport) void viterbi(void);
#endif

// We assume that Bayesian correction and other probability modeling is already done.
// I tried to reference the relevant Matlab code in Springer's implementation.
// Potentially this function could return a double* with the state sequence
// but I'd rather let Python manage that memory instead :)
void viterbi(const double *init_state, size_t state_num, const double *obs_probs, size_t time_len, const double *dur_probs, size_t max_dur, double *out_state_seq) {
    size_t pad_len = (time_len + max_dur -1); //?? -1

    // delta = ones(T+ max_duration_D-1,N)*-inf;
    size_t size_2d_pad = pad_len*state_num; 
    double *delta = malloc(size_2d_pad * sizeof(double));
    if (delta == NULL){
        fprintf(stderr,"Failure to allocate delta array in Viterbi algo");
        return;
    }
    memset(delta,-DBL_MAX,size_2d_pad);
    
    // psi = zeros(T+ max_duration_D-1,N);
    double *psi = calloc(size_2d_pad, sizeof(double));
    if (psi == NULL){
        fprintf(stderr,"Failure to allocate psi array in Viterbi algo");
        return;
    }
    
    // psi_duration =zeros(T + max_duration_D-1,N);
    double *psi_dur = calloc(size_2d_pad, sizeof(double));
    if (psi_dur == NULL){
        fprintf(stderr,"Failure to allocate psi duration array in Viterbi algo");
        return;
    }

    // delta(1,:) = log(pi_vector) + log(observation_probs(1,:))
    for(size_t i=0;i<state_num;++i)
        delta[i*pad_len+0] = log(init_state[i]) + log(obs_probs[i*time_len+0]);
    
    for(int t=1; t<time_len+max_dur-1; ++t){
        for(int state=0; state<state_num; ++state){
            for(int dur=1; dur<=max_dur; ++dur){
                
                int start = t-dur;
                int end   = t;
                
                if(start<0) start=0;
                if(start>time_len-1) start=time_len;
                
                if(end>time_len) end=time_len;
                TODO("update delta, D, psi");
            }
        }
    }
    TODO("backtracking");
}