/*
    lib/NLP/transformers/encoder-decoder/header.hh
    Q@khaa.pk
 */

#include "./../../../../numcy/header.hh"

#ifndef NLP_ENCODER_DECODER_TRANSFORMER_MODEL_HEADER_HH
#define NLP_ENCODER_DECODER_TRANSFORMER_MODEL_HEADER_HH

/*
    In transformers, a common technique for incorporating sequence information is by adding positional encodings to the input embeddings.
    The positional encodings are generated using sine and cosine functions of different frequencies.
    The expression wrapped in the following macro is used to scale the frequency of the sinusoidal functions used to generate positional encodings. 

    div_term
    ----------    
    This expression is used in initializing "div_term".

    The expression is multiplyed by -1
    ------------------------------------
    The resulting "div_term" array contains values that can be used as divisors when computing the sine and cosine values for positional encodings.
    
    Later on "div_term" and "positions" are used with sin() cos() functions to generate those sinusoidal positional encodings.
    The idea is that by increasing the frequency linearly with the position,
    the model can learn to make fine-grained distinctions for smaller positions and coarser distinctions for larger positions.
    
    @sfc, Scaling Factor Constant.
    @d_model, Dimensions of the transformer model.
 */
#define SCALING_FACTOR(sfc, d_model) -1*(log(sfc)/d_model)

/*
    It's worth noting that there's no universally "correct" value for this constant; the choice of using 10000.0 as a constant in the expression wrapped
    in macro "SCALING_FACTOR" is somewhat arbitrary but has been found to work well in practice. 
    Consider trying different values for this constant during experimentation to see if it has an impact on your model's performance.
 */
#define SCALING_FACTOR_CONSTANT 10000.0

/*  Hyperparameters */
/* -----------------*/
/*
    Commonly used values for d_model range from 128 to 1024,
    with 256 being a frequently chosen value because in "Attention is All You Need" paper, the authors used an embedding dimension of 256.
 */
#define DEFAULT_DIMENTIONS_OF_THE_TRANSFORMER_MODEL_HYPERPARAMETER 0x100

/*
    TODO, big description about this hyperparameter is needed
 */
#define DEFAULT_EPOCH_HYPERPARAMETER 1

/* Hyperparameters end here */
/* ------------------------ */

#endif