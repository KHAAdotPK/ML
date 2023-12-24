/*
    lib/NLP/transformers/encoder-decoder/header.hh
    Q@khaa.pk
 */

/* TODO, remove this ugly, when things are done here, I cant do it quick enough */
#include <iostream>

#include "./../../../string/src/string_character_traits.hh"

#ifndef NLP_ENCODER_DECODER_TRANSFORMER_MODEL_HEADER_HH
#define NLP_ENCODER_DECODER_TRANSFORMER_MODEL_HEADER_HH

/*  Hyperparameters */
/* -----------------*/
/*
    Commonly used values for d_model range from 128 to 1024,
    with 256 being a frequently chosen value because in "Attention is All You Need" paper, the authors used an embedding dimension of 256.
 */
#define DEFAULT_DIMENTIONS_OF_THE_TRANSFORMER_MODEL_HYPERPARAMETER 0x100
/*
    Typical values for number of layers in the Transformer model range from 4 to 8 or even higher, depending on the specific application.
    It's common to have an equal number of layers in both the encoder and decoder parts of the Transformer.
 */
#define DEFAULT_NUMBER_OF_LAYERS_FOR_ENCODER_HYPERPARAMETER 0x04
/*
    Number of attention heads
 */
#define DEFAULT_NUMBER_OF_ATTENTION_HEADS_HYPERPARAMETER 0x08
/*
    The dropout_rate is a hyperparameter typically set between 0.1 and 0.5

    Dropout Rate: Dropout is a regularization technique commonly used in neural networks, including the Transformer model. 
    The dropout_rate represents the probability of randomly "dropping out" or deactivating units (neurons) in a layer during training.
    It helps prevent overfitting and encourages the model to learn more robust and generalized representations by reducing interdependence between units.
    The dropout_rate is a hyperparameter typically set between 0.1 and 0.5, indicating the fraction of units that are dropped out during training.
 */
#define DEFAULT_DROP_OUT_RATE_HYPERPARAMETER 0.1

/*
    TODO, big description about this hyperparameter is needed
 */
#define DEFAULT_EPOCH_HYPERPARAMETER 1

/*
   ======================================================= 
  *| On the topic of EPOCH, FORWARD PASS and BATCH SIZE. |*
   =======================================================

   In each epoch, there will be multiple forward passes, one for each batch of data.
   For example, if you have 1000 training samples and a batch size of 10,
   each epoch will consist of 100 forward passes (10 batches x 100 forward passes = 1000 samples processed).
   In 100 epochs, you will have a total of 10,000 forward passes (100 epochs x 100 forward passes per epoch = 10,000).
 */

#define NUMBER_OF_GPUS 1
/*
    In natural language processing, the input sequence size might corresponds to one single sentence or one single paragraph. 
    The number of input sequences then becomes the total number of such sentences or paragraphs.
 */
#define NUMBER_OF_INPUT_SEQUENCES 1
/*
    The batch size determines how many input sequences(like how many lines of texts of paragraphs, also read comments about NUMBER_OF_INPUT_SEQUENCES)
    are processed together in each forward pass, aiming to balance efficiency and memory considerations during training or inference.

    Still our NUMBER_OF_INPUT_SEQUENCES are 1 and also we are using some LEN(INPUT_SEQUENCE)/BATCH_SIZE, we are processing whole of the
    input sequence data in one single forward pass. 
    What would've happened if we had 2 GPUS instead of 1. Then instead of 1 forward pass, it would've taken 2 forward passes to process the whole of
    input sequence per epoch. 
 */
#define DEFAULT_BATCH_SIZE_HYPERPARAMETER(l) (l/NUMBER_OF_GPUS)
/* Hyperparameters end here */

/*
    I HAD LONG DISCUSSION ABOUT THIS TOPIC WITH ChatGPT 3.5 AND ITS LINK IS... https://chat.openai.com/c/52abffc2-98f7-4cba-b923-cea60f7eb213

    #define COLUMNS_IN_SHAPE_OF_ARRAY 10/NUMBER_OF_INPUT_SEQUENCES 
    #define ROWS_IN_SHAPE_OF_ARRAY    10/NUMBER_OF_INPUT_SEQUENCES 
    #define COLUMNS_IN_SHAPE_OF_ARRAY 10/NUMBER_OF_GPUS
    #define ROWS_IN_SHAPE_OF_ARRAY    10/NUMBER_OF_GPUS
    #define BATCH_SIZE (ROWS_IN_SHAPE_OF_ARRAY*COLUMNS_IN_SHAPE_OF_ARRAY)*NUMBER_OF_INPUT_SEQUENCES
    #define BATCH_SIZE (ROWS_IN_SHAPE_OF_ARRAY*COLUMNS_IN_SHAPE_OF_ARRAY)
 */ 

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

                                                
#endif 