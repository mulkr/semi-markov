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
Tstar:         time of the maximum delta_t(j) after T (until max duration)
qstar_Tstar:   most likely state at Tstar
*/
#include <float.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include "utils.h"

#ifdef _WIN32
#define VIT_PREFIX __declspec(dllexport)
#else
#define VIT_PREFIX
#endif

size_t indexing(size_t x, size_t y, size_t width) {
    return y * width + x;
}

// We assume that Bayesian correction and other probability modeling is already done.
// I tried to reference the relevant Matlab code in Springer's implementation.
// Potentially this function could return a double* with the state sequence
// but I'd rather let Python manage that memory instead :)
VIT_PREFIX void viterbi(const double* trans_mat,
             const double* init_state, size_t state_num,
             const double* obs_probs, size_t time_len,
             const double* dur_probs, size_t max_dur,
             int* out_state_seq) {

    size_t pad_len = (time_len + max_dur -1); //?? -1

    // delta = ones(T+ max_duration_D-1,N)*-inf;
    size_t size_2d_pad = pad_len*state_num;
    //TODO: We may need to change the order of columns and rows for some matrices
    double* delta = malloc(size_2d_pad * sizeof(double));
    if (delta == NULL){
        fprintf(stderr,"Failure to allocate delta array in Viterbi algo");
        return;
    }
    for (size_t i=0;i<size_2d_pad;++i){
        delta[i] = -DBL_MAX;
    }

    // psi = zeros(T+ max_duration_D-1,N);
    size_t* psi = calloc(size_2d_pad, sizeof(size_t));
    if (psi == NULL){
        fprintf(stderr,"Failure to allocate psi array in Viterbi algo");
        return;
    }

    // psi_duration =zeros(T + max_duration_D-1,N);
    size_t* psi_dur = calloc(size_2d_pad, sizeof(size_t));
    if (psi_dur == NULL){
        fprintf(stderr,"Failure to allocate psi duration array in Viterbi algo");
        return;
    }

    double* dur_sum = calloc(state_num,sizeof(double));
    for(size_t i=0;i<state_num;++i){
        double sum = 0;
        for(size_t j=0;j<max_dur;++j){
            sum += dur_probs[indexing(j,i,max_dur)];
        }
        dur_sum[i] = sum;
    }

    // delta(1,:) = log(pi_vector) + log(observation_probs(1,:))
    for(size_t i=0;i<state_num;++i)
        delta[indexing(0,i,pad_len)] = log(init_state[i]) + log(obs_probs[indexing(0,i,time_len)]);

    for(size_t t=1; t<pad_len; ++t){
        for(size_t state=0; state<state_num; ++state){
            for(size_t dur=1; dur<=max_dur; ++dur){

                int start = t-dur;
                int end   = t;

                if(start<0) start=0;
                if(start>time_len-1) start=time_len;

                if(end>time_len) end=time_len;

                // [max_delta, max_index] = max(delta(start_t,:)+log(a_matrix(:,j))');
                double max_delta = delta[indexing(start,0,pad_len)];
                size_t max_index = 0;
                for(size_t i=0;i<state_num;++i){
                    double val = delta[indexing(start,i,pad_len)]+log(trans_mat[indexing(state,i,state_num)]);
                    if(val>max_delta){
                        max_delta=val;
                        max_index=i;
                    }
                }

                // probs = prod(observation_probs(start_t:end_t,j));
                double probs = 1;
                for(size_t i=start;i<end;++i){
                    probs *= obs_probs[indexing(state,i,state_num)];
                }

                // emission_probs = log(probs);
                double emit_probs = log(probs);

                // delta_temp = max_delta + (emission_probs)+ log((duration_probs(j,d)./duration_sum(j)));
                double delta_temp = max_delta + emit_probs + log(dur_probs[indexing(dur,state,max_dur)]/ dur_sum[state]);

                if(delta_temp>delta[indexing(t,state,pad_len)]){
                    delta[indexing(t,state,pad_len)] = delta_temp;
                    psi[indexing(t,state,pad_len)] = max_index;
                    psi_dur[indexing(t,state,pad_len)] = dur;
                    // printf("t:%zu state:%zu dur:%zu\n",t,state,dur);
                }
            }
        }
    }

    //select pos from delta after time_len until time_len+max_dur where maximal value
    int state_time_after = time_len;
    for(size_t i=time_len+1;i<pad_len;++i)
        for(size_t j=0;j<state_num;++j)
            if(delta[indexing(i,j,pad_len)]>delta[indexing(state_time_after,j,pad_len)])
                state_time_after = i;
    //select state from delta at state_time_after where maximal value
    int state_after = 0;
    for(size_t i=0;i<state_num;++i)
        if (delta[indexing(state_time_after,i,pad_len)]>delta[indexing(state_time_after,state_after,pad_len)])
            state_after = i;

    //backtrack
    // for(size_t i=0;i<pad_len;++i){
    //     for(size_t j=0;j<state_num;++j){
    //         printf("%zu ",psi[indexing(i,j,pad_len)]);
    //     }
    //     printf("\n");
    // }

    int time_loc = state_time_after;
    // printf("time_loc: %d\n",time_loc);
    // printf("state: %d\n",state_after);
    while(time_loc>1){
        int dstar = psi_dur[indexing(time_loc,state_after,pad_len)];
        for(size_t i=time_loc-dstar;i<=time_loc-1;++i){
            out_state_seq[i] = state_after;
        }
        state_after = psi[indexing(time_loc,state_after,pad_len)];
        // printf("%d,%d,%d ",time_loc-dstar,time_loc-1, state_after);
        time_loc = time_loc - dstar;
    }
    printf("\n");

    //don't forget to free the memory :^)
    free(delta);
    free(psi);
    free(psi_dur);
    free(dur_sum);
}
