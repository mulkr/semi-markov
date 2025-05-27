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
void viterbi(const double* trans_mat,
             const double* init_state, size_t state_num,
             const double* obs_probs, size_t time_len,
             const double* dur_probs, size_t max_dur,
             double* out_state_seq) {

    size_t pad_len = (time_len + max_dur -1); //?? -1

    // delta = ones(T+ max_duration_D-1,N)*-inf;
    size_t size_2d_pad = pad_len*state_num;
    double* delta = malloc(size_2d_pad * sizeof(double));
    if (delta == NULL){
        fprintf(stderr,"Failure to allocate delta array in Viterbi algo");
        return;
    }
    memset(delta,-DBL_MAX,size_2d_pad);
    
    // psi = zeros(T+ max_duration_D-1,N);
    double* psi = calloc(size_2d_pad, sizeof(double));
    if (psi == NULL){
        fprintf(stderr,"Failure to allocate psi array in Viterbi algo");
        return;
    }
    
    // psi_duration =zeros(T + max_duration_D-1,N);
    double* psi_dur = calloc(size_2d_pad, sizeof(double));
    if (psi_dur == NULL){
        fprintf(stderr,"Failure to allocate psi duration array in Viterbi algo");
        return;
    }

    double* dur_sum = calloc(state_num,sizeof(double));
    for(size_t i=0;i<state_num;++i){
        double sum = 0;
        for(size_t j=0;j<max_dur;++j){
            sum += dur_probs[i*max_dur+j];
        }
        dur_sum[i] = sum;
    }

    // delta(1,:) = log(pi_vector) + log(observation_probs(1,:))
    for(size_t i=0;i<state_num;++i)
        delta[i*pad_len+0] = log(init_state[i]) + log(obs_probs[i*time_len+0]);
    
    for(size_t t=1; t<pad_len; ++t){
        for(size_t state=0; state<state_num; ++state){
            for(size_t dur=1; dur<=max_dur; ++dur){
                
                int start = t-dur;
                int end   = t;
                
                if(start<0) start=0;
                if(start>time_len-1) start=time_len;
                
                if(end>time_len) end=time_len;

                // [max_delta, max_index] = max(delta(start_t,:)+log(a_matrix(:,j))');
                double max_delta = 0;
                size_t max_index = 0;
                for(size_t i=0;i<state_num;++i){
                    double val = delta[i*pad_len+start]+log(trans_mat[i*state_num+state]);
                    if(val>max_delta){
                        max_delta=val;
                        max_index=i; 
                    }
                }
                
                // probs = prod(observation_probs(start_t:end_t,j));
                double probs = 0;
                for(size_t i=start;i<end;++i){
                    probs *= obs_probs[i*state_num+state];
                }
                
                // emission_probs = log(probs);
                double emit_probs = log(probs);

                // delta_temp = max_delta + (emission_probs)+ log((duration_probs(j,d)./duration_sum(j)));
                double delta_temp = max_delta + emit_probs + log(dur_probs[state*max_dur+dur]/ dur_sum[state]);

                if(delta_temp>delta[state*pad_len+t]){
                    delta[state*pad_len+t] = delta_temp;
                    psi[state*pad_len+t] = max_index;
                    psi_dur[state*pad_len+t] = dur;
                }
            }
        }
    }
    TODO("backtracking");
}