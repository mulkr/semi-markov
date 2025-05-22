# API outline
## Model
### Attributes
* EmissionProb
* TransitionMat -> 2D numpy array (N x N)
* InitStateProb -> 1D numpy array (N)
* DurationProb -> 2D numpy array (N x K)
### Methods
* Decode - get the best fitting hidden states (Viterbi)
* Sample - generate an output observation sequence
* Fit(?) - fit the transition matrix, duration probabilities, and initial probabilities (Forward-Backward)
## EmissionProb
### Methods
* SampleProb - sample the probability given a state
* UpdateProb - update the emission parameters
* Likelihood - compute likelihood of observations
* Copy(?) - copy the object