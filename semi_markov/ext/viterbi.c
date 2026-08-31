#include <float.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>

#ifdef _WIN32
#define VIT_PREFIX __declspec(dllexport)
#else
#define VIT_PREFIX
#endif

#define NEG_INF (-INFINITY)

static inline double max_array(const double *arr, int n) {
    double best = NEG_INF;
    for (int i = 0; i < n; i++)
        if (arr[i] > best) best = arr[i];
    return best;
}

static inline int argmax_array(const double *arr, int n) {
    double best = NEG_INF;
    int idx = 0;
    for (int i = 0; i < n; i++)
        if (arr[i] > best) { best = arr[i]; idx = i; }
    return idx;
}

static inline double compute_log_likelihood(const double* obs_probs, size_t time_len, size_t time, size_t state, size_t dur) {
    // duration covers time steps [time-dur, time]
    int start = time - dur;
    if (start < 0) return NEG_INF;
    if (time >= time_len) return NEG_INF;

    double log_likelihood = 0.0; // sum of log(p(x))

    for (size_t i = start; i <= time; i++) {
        double p = obs_probs[state * time_len + i];
        if (p <= 0) return NEG_INF;
        log_likelihood += log(p);
    }
    return log_likelihood;
}

// We assume that Bayesian correction and other probability modeling is already done.
// Potentially this function could return a double* with the state sequence
// but I'd rather let Python manage that memory instead :)
VIT_PREFIX void viterbi(const double* trans_mat,
             const double* init_state, size_t state_num,
             const double* obs_probs, size_t time_len,
             const double* dur_probs, size_t max_dur,
             int* out_state_seq) {

    double* delta = malloc(sizeof(double) * time_len * state_num);
    int* psi = malloc(sizeof(int) * time_len  * state_num * 2);
    double* state_val_buffer = malloc(sizeof(double) * state_num);
    double* dur_buffer = malloc(sizeof(double) * max_dur);
    int* dur_buffer_states = malloc(sizeof(int) * max_dur);

    //check memory allocation
    if (delta==NULL){
        fprintf(stderr, "Failure to allocate delta array in Viterbi");
        return;
    }
    if (psi==NULL){
        fprintf(stderr, "Failure to allocate psi array in Viterbi");
        return;
    }
    if (state_val_buffer==NULL){
        fprintf(stderr, "Failure to allocate state value buffer in Viterbi");
        return;
    }
    if (dur_buffer==NULL){
        fprintf(stderr, "Failure to allocate duration buffer in Viterbi");
        return;
    }
    if (dur_buffer_states==NULL){
        fprintf(stderr, "Failure to allocate duration state buffer in Viterbi");
        return;
    }

    //convert inputs to log space
    double* log_startprob = malloc(sizeof(double) * state_num);
    double* log_transmat = malloc(sizeof(double) * state_num * state_num);
    double* log_duration = malloc(sizeof(double) * state_num * max_dur);
    if (log_startprob==NULL){
        fprintf(stderr, "Failure to allocate log start probability array in Viterbi");
        return;
    }
    if (log_transmat==NULL){
        fprintf(stderr, "Failure to allocate log transition matrix in Viterbi");
        return;
    }
    if (log_duration==NULL){
        fprintf(stderr, "Failure to allocate log duration array in Viterbi");
        return;
    }

    for (size_t i = 0; i < state_num; i++) {
        log_startprob[i] = (init_state[i] > 0) ? log(init_state[i]) : NEG_INF;

        for (size_t j = 0; j < state_num; j++) {
            double p = trans_mat[i * state_num + j];
            log_transmat[i * state_num + j] = (p > 0) ? log(p) : NEG_INF;
        }

        for (size_t d = 0; d < max_dur; d++) {
            double p = dur_probs[i * max_dur + d];
            log_duration[i * max_dur + d] = (p > 0) ? log(p) : NEG_INF;
        }
    }

    //forward
    for (size_t t = 0; t < time_len ; t++) {
        for (size_t curr_state = 0; curr_state < state_num; curr_state++) {
            for (size_t dur = 0; dur < max_dur; dur++) {

                double emit_prob_log = compute_log_likelihood(obs_probs, time_len, t, curr_state, dur);

                if (!isfinite(emit_prob_log)) {
                    dur_buffer[dur] = NEG_INF;
                    dur_buffer_states[dur] = -1;
                    continue;
                }

                if (t - dur == 0) {
                    //sequence start
                    dur_buffer[dur] = log_startprob[curr_state] + log_duration[curr_state * max_dur + dur] + emit_prob_log;
                    dur_buffer_states[dur] = -1;

                } else if (t - dur > 0) {
                    //regular transitions
                    for (size_t prev_state = 0; prev_state < state_num; prev_state++) {
                        state_val_buffer[prev_state] = delta[(t - dur - 1) * state_num + prev_state] + log_transmat[prev_state * state_num + curr_state] + emit_prob_log;
                    }
                    double best_prev = max_array(state_val_buffer, state_num);
                    int best_prev_state = argmax_array(state_val_buffer, state_num);

                    dur_buffer[dur] = best_prev + log_duration[curr_state * max_dur + dur];
                    dur_buffer_states[dur] = best_prev_state;

                } else {
                    //infeasible duration
                    dur_buffer[dur] = NEG_INF;
                    dur_buffer_states[dur] = -1;
                }
            }

            double best = max_array(dur_buffer, max_dur);
            int best_d = argmax_array(dur_buffer, max_dur);

            delta[t * state_num + curr_state] = best;
            psi[(t * state_num + curr_state) * 2 + 0] = best_d;
            psi[(t * state_num + curr_state) * 2 + 1] = dur_buffer_states[best_d];
        }
    }

    int back_state = argmax_array(&delta[(time_len - 1) * state_num], state_num);
    int back_dur   = psi[((time_len  - 1) * state_num + back_state) * 2 + 0];

    //backtrack
    int t = time_len  - 1;
    while (t >= 0) {
        for (int k = 0; k < back_dur + 1 && t - k >= 0; k++) {
            out_state_seq[t - k] = back_state;
        }

        int prev_state = psi[(t * state_num + back_state) * 2 + 1];

        t -= (back_dur + 1);
        back_state = prev_state;
        if (t >= 0){
            back_dur = psi[(t * state_num + back_state) * 2 + 0];
        }
    }

    //don't forget to free the memory :^)
    free(delta);
    free(psi);
    free(state_val_buffer);
    free(dur_buffer);
    free(dur_buffer_states);
    free(log_startprob);
    free(log_transmat);
    free(log_duration);
}
