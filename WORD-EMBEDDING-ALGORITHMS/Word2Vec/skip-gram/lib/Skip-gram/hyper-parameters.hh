/*
    ML/NLP/unsupervised/Word2Vec/Skip-gram/hyper-parameters.hh
    Q@khaa.pk
 */

#ifndef R3PLICA_SKIP_GRAM_HYPER_PARAMETERS_HEADER_HH
#define R3PLICA_SKIP_GRAM_HYPER_PARAMETERS_HEADER_HH

#define SKIP_GRAM_DEFAULT_EPOCH 100

/*
   The learning rate controls the step size at each iteration of the optimization process
 */
#define SKIP_GRAM_DEFAULT_LEARNING_RATE 0.1

/*
    Number of neurons in the hidden layer and this represents the size of the hidden layer in the neural network.
    10 neurons is small size, suitable for small vocabulary.
    However, for larger vocabularies and more complex tasks, a larger hidden layer size may be required to capture more intricate relationships 
    between the input and output. 

    Use ifdef, undef define preprocessor directives
 */
#define SKIP_GRAM_EMBEDDNG_VECTOR_SIZE 100

/*
   Size of window of context words around a target word, and use the context words to predict the target word(in CBOW/Skip-Gram model) 
   In the Skip-gram model, the model predicts the context words given a target word
 */ 
#define SKIP_GRAM_WINDOW_SIZE 2

#endif