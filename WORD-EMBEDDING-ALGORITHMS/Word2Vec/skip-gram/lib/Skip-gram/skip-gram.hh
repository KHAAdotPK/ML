/*
    lib/ML/NLP/unsupervised/Word2Vec/Skip-Gram/skip-gram.h
    Q@khaa.pk
 */

//#ifndef CC_TOKENIZER_REPLIKA_PK_SKIP_GRAM_SKIP_GRAM_H_HH
#include <algorithm>

#ifndef CORPUS_FOR_LM_HH
//#include "../../../../../corpus/corpus.hh"
#endif
#ifndef KHAA_PK_NUMCY_HEADER_HH
//#include "../../../../../Numcy/header.hh"
#endif

#ifndef R3PLICA_SKIP_GRAM_HEADER_HH
//#include "header.hh"
#endif
//#endif

//#include "hyper-parameters.hh"
//#include "skip-gram-pairs.hh"

//#include "../../../../../numc/numc.hh"

#ifndef CC_TOKENIZER_REPLIKA_PK_SKIP_GRAM_SKIP_GRAM_H_HH
#define CC_TOKENIZER_REPLIKA_PK_SKIP_GRAM_SKIP_GRAM_H_HH

#include "../../../../../corpus/corpus.hh"
#include "../../../../../Numcy/header.hh"
#include "header.hh"

//#define SKIP_GRAM_DEFAULT_EPOCH         100

/*
    Instance of class pairs has a method named len(), and operator []. The operator [] takes an index into the array held by the instance of pairs.
    This index should stay less than the returned value of method len() of class pairs.
 */
#define SKIP_GRAM_DEFAULT_PAIR_SIZE     2
#define SKIP_GRAM_PAIR_TARGET_INDEX     0 // Center word
#define SKIP_GRAM_PAIR_CONTEXT_INDEX    1

/*
    Number of neurons in the hidden layer and this represents the size of the hidden layer in the neural network.
    10 neurons is small size, suitable for small vocabulary.
    However, for larger vocabularies and more complex tasks, a larger hidden layer size may be required to capture more intricate relationships 
    between the input and output. 

    Use ifdef, undef define preprocessor directives
 */
//#define SKIP_GRAM_HIDDEN_SIZE 10
//#define SKIP_GRAM_EMBEDDNG_VECTOR_SIZE 100

/*
   The learning rate controls the step size at each iteration of the optimization process
 */
//#define SKIP_GRAM_DEFAULT_LEARNING_RATE 0.1

#define SKIP_GRAM_PAIR_TARGET_INDEX 0
#define SKIP_GRAM_PAIR_CONTEXT_INDEX 1

/*
   Size of window of context words around a target word, and use the context words to predict the target word(in CBOW/Skip-Gram model) 
   In the Skip-gram model, the model predicts the context words given a target word
 */ 
//#define SKIP_GRAM_WINDOW_SIZE  2

/*
    Forward declarations
 */
//struct skip_gram_pairs;

/*
template<typename E>
E* softmax(E*, class corpus&, class numc&);
 */

/*
    The following structure is a container designed to hold gradients calculated during backpropagation
    in a two-layer neural network used for word embeddings. The presence of grad_W1 and grad_W2 implies a 
    neural network with two layers. W1 represents the weights between the input layer and the first hidden layer, 
    and W2 represents the weights between the first hidden layer and the output layer.
    - The gradients (partial derivatives of the loss function) with respect to the network's weights
    - Backpropagation, this structure plays a crucial role in backpropagation, 
      an algorithm used to train neural networks. Backpropagation calculates the 
      gradients (partial derivatives of the loss function) with respect to the network's weights.
      These gradients are then used to update the weights in a way that minimizes the loss function

    In summary, the backward_propogation<E> structure is a container designed to hold gradients calculated during 
                backpropagation in a two-layer neural network used for word embeddings.  
 */
template<typename E>
struct backward_propogation 
{
    /*
    backward_propogation(void) : grad_W1(Collective<E>{NULL, DIMENSIONS{0, 0, NULL, NULL}}), grad_W2(Collective<E>{NULL, DIMENSIONS{0, 0, NULL, NULL}}), grad_h_with_respect_to_center_or_target_word(Collective<E>{NULL, DIMENSIONS{0, 0, NULL, NULL}})
    {

    }
     */

    backward_propogation() : grad_weights_input_to_hidden(Collective<E>{NULL, DIMENSIONS{0, 0, NULL, NULL}}), grad_weights_hidden_to_output(Collective<E>{NULL, DIMENSIONS{0, 0, NULL, NULL}}), grad_hidden_with_respect_to_center_word(Collective<E>{NULL, DIMENSIONS{0, 0, NULL, NULL}})
    {
        
    }

    backward_propogation(Collective<E>& grad_W1, Collective<E>& grad_W2) : grad_weights_input_to_hidden(grad_W1), grad_weights_hidden_to_output(grad_W2), grad_hidden_with_respect_to_center_word(Collective<E>{NULL, DIMENSIONS{0, 0, NULL, NULL}})
    {

    }

    //private:
        /*
            Both arrays has shape which is (corpus::len(), REPLIKA_HIDDEN_SIZE) and (REPLIKA_HIDDEN_SIZE, corpus::len()) respectovely
         */
        //E* grad_W1;
        /*
            Stores the gradients(The gradients (partial derivatives of the loss function) with respect to the network's weights)
            for the first layer weights (W1)
         */
        //Collective<E> grad_W1;
        /*
         * grad_weights_input_to_hidden: This collective object stores the gradients with respect to the weights between the input layer and the hidden layer (W1).
         * It has a shape of (corpus::len(), REPLIKA_HIDDEN_SIZE).
         */
        Collective<E> grad_weights_input_to_hidden;
        //E* grad_W2;
        /*
            Similar to grad_W1, this member stores the gradients for the second layer weights (W2)
         */
        //Collective<E> grad_W2;
        /*
         * grad_weights_hidden_to_output: This collective object stores the gradients with respect to the weights between the hidden layer and the output layer (W2).
         * It has a shape of (REPLIKA_HIDDEN_SIZE, corpus::len()).
         */
        Collective<E> grad_weights_hidden_to_output;
        /*
            Which are the gradients of the loss function with respect to the first layer weights, second layer weights, and the center word input, respectively.
            (REPLIKA_VOCABULARY_LENGTH,, SKIP_GRAM_HIDDEN_SIZE)
         */
        /*
            This member stores the gradients with respect to the center word input (likely the word used as a reference in the word embedding task)
         */
        //E* grad_h_with_respect_to_center_or_target_word;
        //Collective<E> grad_h_with_respect_to_center_or_target_word;
        /*
         * grad_hidden_with_respect_to_center_word: This collective object stores the gradients with respect to the center word input (the word used as a reference in the word embedding task).
         * It has a shape of (REPLIKA_VOCABULARY_LENGTH, SKIP_GRAM_HIDDEN_SIZE).
         */
        Collective<E> grad_hidden_with_respect_to_center_word;
};

/*
    The following composite represents the internal state of the forward pass in a neural network,
    specifically for CBOW or Skip-gram models.

    This template allows the struct to be used with different data types for the internal variables (h, y_pred, and u).
    Common choices for E could be float or double.
 */
template<typename E>
struct forward_propogation 
{
    forward_propogation(void) : hidden_layer_vector(Collective<E>{NULL, DIMENSIONS{0, 0, NULL, NULL}}), predicted_probabilities(Collective<E>{NULL, DIMENSIONS{0, 0, NULL, NULL}}), intermediate_activation(Collective<E>{NULL, DIMENSIONS{0, 0, NULL, NULL}})
    {        
    }

    forward_propogation<E>(Collective<E>& h, Collective<E>& y_pred, Collective<E>& u) : hidden_layer_vector(h), predicted_probabilities(y_pred), intermediate_activation(u)
    {

    }

    //private:
        /*
            In the context of our CBOW/Skip-Gram model, h refers to the hidden layer vector obtained by averaging the embeddings of the context words.
            It is used in both the forward and backward passes of the neural network.

            The shape of this array is (1, SKIP_GRAM_EMBEDDNG_VECTOR_SIZE), 
            a single row vector with the size of the embedding dimension.
         */
        //E* h;
        Collective<E> hidden_layer_vector;
        /*
            y_pred is a Numcy array of predicted probabilities of the output word given the input context. 
            In our implementation, it is the output of the forward propagation step.

            The shape of this array is (1, len(vocab)), indicating a single row vector with the length of the vocabulary and 
            where each element corresponds to the predicted probability of a specific word.
         */
        //E* y_pred;
        Collective<E> predicted_probabilities;  

        /*	
            Represents an intermediat gradient.	 
            This vector has shape (1, len(vocab)), similar to y_pred. 
            It represents the result of the dot product operation between the center or target word vector "h" and the weight matrix W2.
            The result stored in "u” captures the combined influence of hidden neurons on predicting context words. It provides a
            numerical representation of how likely each word in the vocabulary is to be a context word of a given target 
            word (within the skip-gram model).

            The variable "u" serves as an intermediary step in the forward pass, representing the activations before applying 
            the “softmax” function to generate the predicted probabilities. 

            It represents internal state in the neural network during the working of "forward pass".
            This intermediate value is used in calculations involving gradients in "backward pass" or "back propogation"(the function backward).
         */
        //E* u;
        Collective<E> intermediate_activation;  
};

//struct skip_gram_pairs;

template<typename E>
E* one_hot(unsigned int y_true, size_t vocab_len, class numc& numc_obj) 
{
    E* ptr = numc_obj.zeros<E>(1, vocab_len);

    if (ptr != NULL)
    {
        ptr[y_true] = 1;
    }

    return ptr;
}

/*
  TODO, details about how does it work
   u = h * W2 
   h is a row vector with shape (1, REPLIKA_HIDDEN_SIZE) 
   W2 is vector and has shape (REPLIKA_HIDDEN_SIZE, corpus::len())
   u is a row vector as well and has shape (1, corpus::len())

template<typename E>
E* softmax(E* u, class corpus& corpus_obj, class numc& numc_obj)
{   
    cc_tokenizer::allocator<char> alloc_obj;

    E* u_minus_max = NULL; // Stores all elements of a u, minus an scalar.
    //SUBTRACT_SCALAR(u, numc_obj.max<double>(u, REPLIKA_HIDDEN_SIZE), REPLIKA_HIDDEN_SIZE, u_minus_max, E);
    SUBTRACT_SCALAR(u, numc_obj.max<double>(u, corpus_obj.len()), corpus_obj.len(), u_minus_max, E);

    //E* e_u_minus_max = numc_obj.exp<double>(u_minus_max, REPLIKA_HIDDEN_SIZE);
    
         //   Each element of array is transformed into a base^power form
    
    E* e_u_minus_max = numc_obj.exp<double>(u_minus_max, corpus_obj.len());

    E e_u_minus_max_sum = 0; 
    //ARRAY_SUM(e_u_minus_max, REPLIKA_HIDDEN_SIZE, e_u_minus_max_sum);
    ARRAY_SUM(e_u_minus_max, corpus_obj.len(), e_u_minus_max_sum);

    double* e_u_minus_max_divided_by_e_u_minus_max_sum;
    //DIVIDE_ARRAY_BY_SCALAR(e_u_minus_max, e_u_minus_max_sum, REPLIKA_HIDDEN_SIZE, e_u_minus_max_divided_by_e_u_minus_max_sum, double);
    DIVIDE_ARRAY_BY_SCALAR(e_u_minus_max, e_u_minus_max_sum, corpus_obj.len(), e_u_minus_max_divided_by_e_u_minus_max_sum, double);

    alloc_obj.deallocate(reinterpret_cast<char*>(u_minus_max));
    alloc_obj.deallocate(reinterpret_cast<char*>(e_u_minus_max));
    
    
      // array of exp(u - max)/sum_of(exp(u - max)) and shape of this array is (1, REPLIKA_VOCABULARY_LENGTH or vocab.len)
    
    return e_u_minus_max_divided_by_e_u_minus_max_sum;
}
 */

/*
    The softmax function is a mathematical function that takes a vector of real numbers as input and normalizes
    them into a probability distribution.
    This distribution ensures that all the output values lie between 0 and 1, and they sum up to 1. 
    It's particularly useful in scenarios where you want to interpret the output as probabilities,
    such as the probabilities of a word belonging to different categories.
 */
template <typename T>
Collective<T> softmax(Collective<T>& a, bool verbose = false) throw (ala_exception)
{
    if (verbose)
    {
        std::cout<< "In softmax" << std::endl;
    }

    Collective<T> m = Numcy::max(a);

    if (verbose)
    {
        std::cout<< "MAX = " << m[0] << std::endl;
    }

    // a_m, a minus m
    Collective<T> a_m = Numcy::subtract(a, m);
    
    // e_a_m, exp over a_m
    Collective<T> e_a_m = Numcy::exp(a);
    // Sum of e_a_m
    Collective<T> s_e_a_m = Numcy::sum(e_a_m);

    if (verbose)
    {
        std::cout<< "s_e_a_m = " << s_e_a_m[0] << std::endl;
    }

    Collective<T> e_u_minus_max_divided_by_e_u_minus_max_sum = Numcy::divide(e_a_m, s_e_a_m);
            
    return e_u_minus_max_divided_by_e_u_minus_max_sum;
}

/*
    Performs part of the forward propagation step in a Skip-gram model
    
    @W1, embedding matrix (Collective object)
    @W2, output layer weight matrix (Collective object)
    @vocab,
    @pair, a pointer to a word pair object (containing center word index and context word indices) 
 */
template <typename T>
forward_propogation<T> forward(Collective<T> W1, Collective<T>& W2, CORPUS_REF vocab, WORDPAIRS_PTR pair, bool verbose = false) throw (ala_exception)
{   
    if (verbose)
    { 
        /*
            Left context words
         */
        for (int i = SKIP_GRAM_WINDOW_SIZE - 1; i >= 0; i--)
        {
            std::cout<< vocab[(*(pair->getLeft()))[i]].c_str() << " ";
        }
        /*
            The center word
         */
        std::cout<< vocab[pair->getCenterWord()].c_str() << " ";
        /*
            The right context words
         */
        for (int i = 0; i < SKIP_GRAM_WINDOW_SIZE; i++)
        {
            std::cout<< vocab[(*(pair->getRight()))[i]].c_str() << " ";
        }
        std::cout<< std::endl;
    }
    
    if (verbose)
    {
        std::cout<< "W1 number of columns -> " << W1.getShape().getNumberOfColumns() << std::endl;
        std::cout<< "W1 number of rows -> " << W1.getShape().getDimensionsOfArray().getNumberOfInnerArrays() << std::endl;    
        std::cout<< "W2 number of columns -> " << W2.getShape().getNumberOfColumns() << std::endl;
        std::cout<< "W2 number of rows -> " << W2.getShape().getDimensionsOfArray().getNumberOfInnerArrays() << std::endl;
    }

    //T* h = W1.ptr + pair->getCenterWord()*W1.getShape().getNumberOfColumns();
    
    /*
        TODO, 
     */
    if (pair->getCenterWord() > W1.getShape().getDimensionsOfArray().getNumberOfInnerArrays())
    {
        throw ala_exception("Skip-gram forward() Error: ");
    }

    T* h = NULL;

    try 
    {
        h = cc_tokenizer::allocator<T>().allocate(W1.getShape().getNumberOfColumns());
    }
    catch (ala_exception& e)
    {
        throw ala_exception(e.what());
    }

    for (cc_tokenizer::string_character_traits<char>::size_type i = 0; i < W1.getShape().getNumberOfColumns(); i++)
    {
        *(h + i) = W1[W1.getShape().getNumberOfColumns()*(pair->getCenterWord() - INDEX_ORIGINATES_AT_VALUE) + i];
    }

    if (verbose)
    {
        for (int i = 0; i < W1.getShape().getNumberOfColumns(); i++)
        {
            std::cout<< h[i] << ", ";
        }    
        std::cout<< std::endl;
    }
    
    Collective<T> u = Numcy::dot(Collective<T>{h, DIMENSIONS{W1.getShape().getNumberOfColumns(), 1, NULL, NULL}}, W2);

    if (verbose)
    {
        std::cout<< "u number of columns -> " << u.getShape().getNumberOfColumns() << std::endl;
        std::cout<< "u number of rows -> " << u.getShape().getDimensionsOfArray().getNumberOfInnerArrays() << std::endl;
    }

    Collective<T> y_pred = softmax<T>(u);
        
    //cc_tokenizer::allocator<T>().deallocate(h);
        
    //return forward_propogation<T>{Collective<T>{NULL, DIMENSIONS{0, 0, NULL, NULL}}, Collective<T>{NULL, DIMENSIONS{0, 0, NULL, NULL}}, Collective<T>{NULL, DIMENSIONS{0, 0, NULL, NULL}}}; 
    return forward_propogation<T>{Collective<T>{h, DIMENSIONS{W1.getShape().getNumberOfColumns(), 1, NULL, NULL}}, y_pred, u};
}

template <typename E = double>
backward_propogation<E> backward(Collective<E>& W1, Collective<E>& W2, CORPUS_REF vocab, forward_propogation<E>& fp, WORDPAIRS_PTR pair, bool verbose = false) throw (ala_exception)
{        
    /* The hot one array is row vector, and has shape (1, vocab.len = REPLIKA_VOCABULARY_LENGTH a.k.a no redundency) */
    Collective<E> oneHot;    
    /*
        Creating a One-Hot Vector, using Numcy::zeros with a shape of (vocab.numberOfUniqueTokens(), 1).
        This creates a zero-filled column vector with a length equal to the vocabulary size
     */
    try
    {
        oneHot = Numcy::zeros(DIMENSIONS{vocab.numberOfUniqueTokens(), 1, NULL, NULL});
    }
    catch(ala_exception& e)
    {
        std::cerr << e.what() << std::endl;
    }
    
    /*
       The following code block, iterates through the context word indices (left and right) from the pair object.
       For each valid context word index (i), it sets the corresponding element in the oneHot vector to 1.
       This effectively creates a one-hot encoded representation of the context words.
     */
    for (int i = SKIP_GRAM_WINDOW_SIZE - 1; i >= 0; i--)
    {       
        if (((*(pair->getLeft()))[i] - INDEX_ORIGINATES_AT_VALUE) < vocab.numberOfUniqueTokens())
        {
            oneHot[(*(pair->getLeft()))[i] - INDEX_ORIGINATES_AT_VALUE] = 1;
        }
    }
    for (int i = 0; i < SKIP_GRAM_WINDOW_SIZE; i++)
    {
        if ((*(pair->getRight()))[i] <= vocab.numberOfUniqueTokens())
        {
            oneHot[(*(pair->getRight()))[i] - INDEX_ORIGINATES_AT_VALUE] = 1;
        }        
    }
    /*
    for (int i = 0; i < vocab.numberOfUniqueTokens(); i++)
    {
        std::cout<< oneHot[i] << ", ";
    }
    std::cout<< std::endl <<  "---- - - -- ---- ---- ------------ -- - - - - - - " << std::endl;
     */

    //std::cout<< fp.predicted_probabilities.getShape().getN() << std::endl;

    /* The shape of grad_u is the same as y_pred which is (1, len(vocab)) */
    Collective<E> grad_u;
    try 
    {   
        // fp.predicted_probabilities, (1, len(vocab))
        // oneHot, (1, len(vocab))
        // grad_u, (1, len(vocab)) 
        grad_u = Numcy::subtract(fp.predicted_probabilities, oneHot);
    }
    catch (ala_exception& e)
    {
        std::cout<< e.what() << std::endl;   
    }
    /*
    for (int i = 0; i < grad_u.getShape().getN(); i++)
    {
        if (grad_u[i] != fp.predicted_probabilities[i])
        {
            std::cout<< grad_u[i] << ", [" << fp.predicted_probabilities[i] << "], ";
        }
    }
    std::cout<< std::endl;
     */

    Collective<E> grad_W2;
    try 
    {
        // fp.intermediate_activation, (1, len(vocab))
        // grad_u, (1, len(vocab))
        // grad_W2, (len(vocab), len(vocab))
        grad_W2 = Numcy::outer(fp.intermediate_activation, grad_u);
    }
    catch (ala_exception& e)
    {
        std::cout<< e.what() << std::endl;
    }

    Collective<E> W2_T;
    try 
    {
        W2_T = Numcy::transpose(W2);
    }
    catch (ala_exception& e)
    { 
        std::cout<< e.what() << std::endl;
    }

    Collective<E> grad_h;
    try
    {                
        grad_h = Numcy::dot(grad_u, W2_T);
    }
    catch (ala_exception& e)
    {
        std::cout<< e.what() << std::endl;
    }

    Collective<E> grad_W1;
    try
    {
        grad_W1 = Numcy::zeros(DIMENSIONS{SKIP_GRAM_EMBEDDNG_VECTOR_SIZE, vocab.numberOfUniqueTokens(), NULL, NULL});
    }
    catch(ala_exception& e)
    {
        std::cerr << e.what() << std::endl;
    }
        
    //std::cout<< "grad_h -> " << grad_h.getShape().getNumberOfColumns() << ", " << grad_h.getShape().getDimensionsOfArray().getNumberOfInnerArrays() << std::endl;
    //std::cout<< "grad_W1 -> " << grad_W1.getShape().getNumberOfColumns() << ", " << grad_W1.getShape().getDimensionsOfArray().getNumberOfInnerArrays() << std::endl;
    //std::cout<< "W2_T -> " << W2_T.getShape().getNumberOfColumns() << ", " << W2_T.getShape().getDimensionsOfArray().getNumberOfInnerArrays() << std::endl;
    //std::cout<< "W2 -> " << W2.getShape().getNumberOfColumns() << ", " << W2.getShape().getDimensionsOfArray().getNumberOfInnerArrays() << std::endl;

    for (cc_tokenizer::string_character_traits<char>::size_type i = 0; i < grad_W1.getShape().getNumberOfColumns(); i++)
    {
        grad_W1[(pair->getCenterWord() - INDEX_ORIGINATES_AT_VALUE)*SKIP_GRAM_EMBEDDNG_VECTOR_SIZE + i] += grad_h[i];
    }

    return backward_propogation<E>{grad_W1, grad_W2};
}

/*
    EPOCH?    
    In the context of training a machine learning model, an epoch is defined as a complete pass over the entire training dataset during training.
    One epoch is completed when the model has made one update to the weights based on each training sample in the dataset.
    In other words, during one epoch, the model has seen every example in the dataset once and has made one update to the model parameters for each example.     
 */
/*
    Training loop
    -------------
    @epoch, number of times the training loop would iterate
    @W1, embedding matrix. Each row in W1 is a unique word's embedding vector, representing its semantic relationship with other words
    @W2, output layer. Weights for predicting context words
    @el, epoch loss
    @vocab, instance of class corpus
    @pairs, inctance of class skip gram pairs. The target/center word and its context words
    @lr, learning rate. The learning rate controls the step size at each iteration of the optimization process
    @rs, regulirazation strength. To prevent the model over-learning from the data
    @t, 
    @verbose, when true puts more text on screen to help debug code    
 */
#define SKIP_GRAM_TRAINING_LOOP(epoch, W1, W2, el, vocab, pairs, lr, rs, t, verbose)  {\
    for (cc_tokenizer::string_character_traits<char>::size_type i = 1; i <= epoch; i++)\
    {\
        el = 0;\
        if (verbose)\
        {\
            std::cout<< "Epoch# " << i << " of " << epoch << " epochs." << std::endl;\
        }\
        \
        Numcy::Random::shuffle<SKIPGRAMPAIRS>(pairs, pairs.get_number_of_word_pairs());\
        while (pairs.go_to_next_word_pair() != cc_tokenizer::string_character_traits<char>::eof())\
        {\
            WORDPAIRS_PTR pair = pairs.get_current_word_pair();\
            forward_propogation<t> fp;\
            backward_propogation<t> bp;\
            /* So now we've a pair, a pair is LEFT_CONTEXT_WORD/S CENTER_WORD RIGHT_CONTEXT_WORD/S */\
            try\
            {\
                fp = forward<t>(W1, W2, vocab, pair);\
                bp = backward<t>(W1, W2, vocab, fp, pair);\
                /*std::cout<< "bp.grad_weights_input_to_hidden -> " << bp.grad_weights_input_to_hidden.getShape().getNumberOfColumns() << ", " << bp.grad_weights_input_to_hidden.getShape().getDimensionsOfArray().getNumberOfInnerArrays() << std::endl;*/\
                /*std::cout<< "bp.grad_weights_hidden_to_output -> " << bp.grad_weights_hidden_to_output.getShape().getNumberOfColumns() << ", " << bp.grad_weights_hidden_to_output.getShape().getDimensionsOfArray().getNumberOfInnerArrays() << std::endl;*/\
            }\
            catch (ala_exception& e)\
            {\
                std::cout<< e.what() << std::endl;\
            }\
            /*W1 -= bp.grad_weights_input_to_hidden * lr;*/\
            /*bp.grad_weights_hidden_to_output * lr;*/\
            /*std::cout<< "W2 -> " << W2.getShape().getNumberOfColumns() << ", " << W2.getShape().getDimensionsOfArray().getNumberOfInnerArrays() << std::endl;*/\
            Collective<t> W2_reshaped;\
            try\
            {\
                W2_reshaped = Numcy::reshape(W2, bp.grad_weights_hidden_to_output);\
            }\
            catch(ala_exception& e)\
            {\
                std::cout<< e.what() << std::endl;\
            }\
            /*std::cout<< "W2_reshaped -> " << W2_reshaped.getShape().getNumberOfColumns() << ", " << W2_reshaped.getShape().getDimensionsOfArray().getNumberOfInnerArrays() << std::endl;*/\
            /*for (cc_tokenizer::string_character_traits<char>::size_type i = 0; i < W2_reshaped.getShape().getDimensionsOfArray().getNumberOfInnerArrays(); i++)*/\
            /*{*/\
                /*for (cc_tokenizer::string_character_traits<char>::size_type j = 0; j < W2_reshaped.getShape().getNumberOfColumns(); j++)*/\
                /*{*/\
                    /*std::cout<< W2_reshaped[i*W2_reshaped.getShape().getNumberOfColumns() + j] << " ";*/\
                /*}*/\
                /*std::cout<< std::endl;*/\
            /*}*/\
            /*std::cout<<" ----------------------------------------------------------------------------------------------------------------- " << std::endl;*/\
            W1 -= bp.grad_weights_input_to_hidden * lr;\
            W2_reshaped -= bp.grad_weights_hidden_to_output * lr;\
            for (cc_tokenizer::string_character_traits<char>::size_type i = 0; i < W2.getShape().getDimensionsOfArray().getNumberOfInnerArrays(); i++)\
            {\
                for (cc_tokenizer::string_character_traits<char>::size_type j = 0; j < W2.getShape().getNumberOfColumns(); j++)\
                {\
                    W2[i*W2.getShape().getNumberOfColumns() + j] = W2_reshaped[i*W2_reshaped.getShape().getNumberOfColumns() + j];\
                }\
            }\
            /*std::cout<< fp.predicted_probabilities.getShape().getNumberOfColumns() << fp.predicted_probabilities.getShape().getDimensionsOfArray().getNumberOfInnerArrays() << std::endl;*/\
            /*std::cout<< "-> " << fp.predicted_probabilities[pair->getCenterWord() - INDEX_ORIGINATES_AT_VALUE] << std::endl;*/\
            if (!_isnanf(fp.predicted_probabilities[pair->getCenterWord() - INDEX_ORIGINATES_AT_VALUE]))\
            {\
                el = el + (-1*log(fp.predicted_probabilities[pair->getCenterWord() - INDEX_ORIGINATES_AT_VALUE]));\
            }\
        }\
        std::cout<< "el = " << el/pairs.get_number_of_word_pairs() << std::endl;\
    }\
}\

#ifdef ONLY_FOR_DOCUMENTATION_PURPOSES
/*    
    A pair is a center/target word and a context word
    Many(maximum 4) context words can have same center/target word
     
    @p, instance of struct skip_gram_pairs
    @v, vocabulary, instance of class corpus
    @verbose,

    This is what it is doing
    context_words = corpus.split()[max(0, i-window_size):i] + corpus.split()[i+1:i+window_size+1]
 */
#define GENERATE_SKIP_GRAM_PAIRS(p, v, verbose) {\
}\

void backward_old(Corpus::corpus_index_type target_or_center_word_index_in_vocabulary, Corpus::corpus_index_type context_word_index_in_vocabulary, double (*W1)[SKIP_GRAM_HIDDEN_SIZE], double* W2, class corpus& vocab, class numc& numc_obj, forward_propogation<double>& fp)
{
    std::cout<<"Hello World"<<std::endl;

    double* hot_one_array = one_hot<double>(context_word_index_in_vocabulary - REPLIKA_PK_INDEX_ORIGINATES_AT, vocab.len(), numc_obj);
    double* grad_u = numc_obj.subtract_matrices<double>(fp.y_pred, hot_one_array, vocab.len());
    double* grad_W2 = numc_obj.outer<double>(fp.h, grad_u, SKIP_GRAM_HIDDEN_SIZE, vocab.len());
    double* W2_T = numc_obj.transpose_matrix<double>(reinterpret_cast<double*>(W2), SKIP_GRAM_HIDDEN_SIZE, vocab.len());
    double* grad_h = numc_obj.dot(grad_u, W2_T, {vocab.len(), 1, NULL, NULL}, {SKIP_GRAM_HIDDEN_SIZE, vocab.len(), NULL, NULL}, REPLIKA_PK_NUMC_YAXIS);
    double* grad_W1 = numc_obj.zeros<double>(2, vocab.len(), SKIP_GRAM_HIDDEN_SIZE);

    /*
        The ones in place of context indexes, shape of nes_in_place_of_context_indexes is (1, REPLIKA_HIDDEN_SIZE)
     */
    double* ones_in_place_of_context_indexes = numc_obj.ones<double>(1, vocab.get_size_of_context_line());
    /*
        The shape of matrix(outer_of_grad_h_and_ones_in_place_of_context_indexes) is (REPLIKA_HIDDEN_SIZE, corpus_obj.get_size_of_context_line())
     */
    /*
        The shape of grad_h (1, REPLIKA_HIDDEN_SIZE), the shape of ones_in_place_of_context_indexes is (1, corpus_obj.get_size_of_context_line())
        The outer of grad_h and ones in place of context indexes and the shape of outer_of_grad_h_and_ones_in_place_of_context_indexes (REPLIKA_HIDDEN_SIZE, corpus_obj.get_size_of_context_line())
     */
    double* outer_of_grad_h_and_ones_in_place_of_context_indexes = numc_obj.outer<double>(grad_h, ones_in_place_of_context_indexes, SKIP_GRAM_HIDDEN_SIZE, vocab.get_size_of_context_line());
    /*
        The transpose of outer_of_grad_h_and_ones_in_place_of_context_indexes and it has shape of (corpus_obj.get_size_of_context_line(), REPLIKA_HIDDEN_SIZE)
     */
    double* transpose_of_outer_of_grad_h_and_ones_in_place_of_context_indexes = numc_obj.transpose_matrix<double>(outer_of_grad_h_and_ones_in_place_of_context_indexes, SKIP_GRAM_HIDDEN_SIZE, vocab.get_size_of_context_line());
    /*
        It is a two dimensional array... the grad_W1_subarray
        number of lines of this array is equal to the returned value of this funtion call corpus_obj.get_size_of_context_line() and columns equal to REPLIKA_HIDDEN_SIZE
     */
    double (*grad_W1_subarray)[SKIP_GRAM_HIDDEN_SIZE];
    std::cout<<"-----------> "<<vocab.get_size_of_context_line()<<std::endl;
    //SUBARRAY(double, grad_W1_subarray, reinterpret_cast<double(*)[SKIP_GRAM_HIDDEN_SIZE]>(grad_W1), vocab.len(), SKIP_GRAM_HIDDEN_SIZE, context_word_index_in_vocabulary, vocab.get_size_of_context_line());

}

/* 
    @center_or_target_word_index_in_vocabulary,
    @W1,
    @W2,
    @vocab,
    @numc_obj,
 */
forward_propogation<double> forward(corpus::corpus_index_type center_or_target_word_index_in_vocabulary, double (*W1)[SKIP_GRAM_HIDDEN_SIZE], double* W2, class corpus& vocab, class numc& numc_obj)
{
    cc_tokenizer::allocator<char> alloc_obj;

    /*  
        Center or target word implies that it is a vec from W1
     */
    
    /*
        Center or target word vetor is what h has shape of (1, SKIP_GRAM_HIDDEN_SIZE)
     */
    //double* h = W1[center_or_target_word_index_in_vocabulary];

    double* h = reinterpret_cast<double*>(alloc_obj.allocate(sizeof(double)*SKIP_GRAM_HIDDEN_SIZE));
    memcpy(h, W1[center_or_target_word_index_in_vocabulary], SKIP_GRAM_HIDDEN_SIZE*sizeof(double));
    
    /*
        Shape of u is (1, vocab.len())
     */
    double* u = numc_obj.dot(h, W2, {SKIP_GRAM_HIDDEN_SIZE, 1, NULL, NULL}, {vocab.len(), SKIP_GRAM_HIDDEN_SIZE}, REPLIKA_PK_NUMC_YAXIS);   

    // y_pred has shape of (1, vocab.len())
    double* y_pred = softmax(u, vocab, numc_obj); 
    
    //alloc_obj.deallocate(reinterpret_cast<char*>(u));

    return {h /* shape (1, SKIP_GRAM_HIDDEN_SIZE) */, y_pred /* shape (1, vocab.len()) */, u /* (1, vocab.len()) */};
}

/*
    @a, array of size @n and of @tp type
    @b, array of size @n and of @tp type
    @n, size of arrays @a and @b
    @tp, type of arrays @a and @b
 */
#define SUBTRACT_ARRAY_FROM_ARRAY(a, b, n, tp)   {\
                                                    for (cc_tokenizer::string_character_traits<char>::size_type i = 0; i < n; i++)\
                                                    {\
                                                        reinterpret_cast<tp*>(a)[i] = reinterpret_cast<tp*>(a)[i] - reinterpret_cast<tp*>(b)[i];\
                                                    }\
                                                 }\

/*
  @a, array of size @n
  @n, number of elements in an array @a  
  @s, scalar @s is placeholder for the calculated sum
 */
#define ARRAY_SUM(a, n, s) {\
                              for (cc_tokenizer::string_character_traits<char>::size_type i = 0; i < n; i++)\
                              {\
                                  s += a[i];\
                              }\
                           }\

/*
    @a, array of type @tp and size @n
    @s, divisor of @tparam
    @n, size of array @a of type @tparam
    @p, pointer to @tp, the new array of the same size as the @a. Each value of @a is divided by @s 
    @tp, type of divisor @s and dividend @a. The @s and @a has same type which is @tp
 */
#define DIVIDE_ARRAY_BY_SCALAR(a, s, n, p, tp) {\
                                                cc_tokenizer::allocator<char> alloc_obj;\
                                                p = reinterpret_cast<tp*>(alloc_obj.allocate(sizeof(tp)*n));\
                                                for(cc_tokenizer::string_character_traits<char>::size_type i = 0; i < n; i++)\
                                                {\
                                                    p[i] = a[i]/s;\
                                                }\
                                               }\

/*    
    A pair is a center/target word and a context word
    Many(maximum 4) context words can have same center/target word
     
    @p, instance of struct skip_gram_pairs
    @v, vocabulary, instance of class corpus
    @verbose,

    This is what it is doing
    context_words = corpus.split()[max(0, i-window_size):i] + corpus.split()[i+1:i+window_size+1]
 */
#define GENERATE_SKIP_GRAM_PAIRS(p, v, verbose) {\
                                                cc_tokenizer::allocator<char> alloc_obj;\
                                                skip_gram_pairs::number_of_pairs_type n = 0;\
                                                for (skip_gram_pairs::number_of_pairs_type i = 0; i < v.len(false); /* Length corpus, length of corpus is more than vocabulary */ i++)\
                                                {\
                                                    if (verbose)\
                                                    {\
                                                        std::cout<<v(i + REPLIKA_PK_INDEX_ORIGINATES_AT, false).c_str()<<", (i - SKIP_GRAM_WINDOW_SIZE) = "<<i - SKIP_GRAM_WINDOW_SIZE<<", std::max<long>(0, (i - SKIP_GRAM_WINDOW_SIZE)) = "<<std::max<long>(0, (i - SKIP_GRAM_WINDOW_SIZE))<<" -> ";\
                                                    }\
                                                    for (skip_gram_pairs::number_of_pairs_type j = std::max<long>(0, (i - SKIP_GRAM_WINDOW_SIZE)); /* It would work and not break, because max is parametrized on long and not on unsigned long */ j < i; j++)\
                                                    {\
                                                        if (verbose)\
                                                        {\
                                                            std::cout<<v(j + REPLIKA_PK_INDEX_ORIGINATES_AT, false).c_str()<<" ";\
                                                        }\
                                                        n = n + 1 /*SKIP_GRAM_DEFAULT_PAIR_SIZE*/;\
                                                    }\
                                                    if (verbose)\
                                                    {\
                                                        std::cout<<" - - - ";\
                                                    }\
                                                    for (skip_gram_pairs::number_of_pairs_type j = (i + 1); j < (i + SKIP_GRAM_WINDOW_SIZE + 1); j++)\
                                                    {\
                                                        if ((j < v.len(false)))\
                                                        {\
                                                            if (verbose)\
                                                            {\
                                                                std::cout<<v(j + REPLIKA_PK_INDEX_ORIGINATES_AT, false).c_str()<<" ";\
                                                            }\
                                                            n = n + 1 /*SKIP_GRAM_DEFAULT_PAIR_SIZE*/;\
                                                        }\
                                                    }\
                                                    if (verbose)\
                                                    {\
                                                        std::cout<<std::endl;\
                                                    }\
                                                }\
                                                if (verbose)\
                                                {\
                                                    std::cout<<std::endl;\
                                                    std::cout<<"Number of \"targets word and context word\" pairs is = "<<n<<std::endl<<std::endl;\
                                                }\
                                                /*--------------------------------------------------------------------------------------*/\
                                                corpus::corpus_index_type* ptr = reinterpret_cast<corpus::corpus_index_type*>(alloc_obj.allocate(sizeof(corpus::corpus_index_type)*n*SKIP_GRAM_DEFAULT_PAIR_SIZE));\
                                                corpus::corpus_index_type k = 0;\
                                                for (skip_gram_pairs::number_of_pairs_type i = 0; i < v.len(false); /* Length corpus, length of corpus is more than vocabulary */ i++)\
                                                {\
                                                    if (verbose)\
                                                    {\
                                                        /*std::cout<<v(i - REPLIKA_PK_INDEX_ORIGINATES_AT, false).c_str()<<" -> ";*/\
                                                    }\
                                                    for (skip_gram_pairs::number_of_pairs_type j = std::max<long>(0, (i - SKIP_GRAM_WINDOW_SIZE)); j < i; j++)\
                                                    {\
                                                        if (verbose)\
                                                        {\
                                                            /*std::cout<<v(j - REPLIKA_PK_INDEX_ORIGINATES_AT, false).c_str()<<" ";*/\
                                                        }\
                                                        /*n = n + 1*/ /*SKIP_GRAM_DEFAULT_PAIR_SIZE*/;\
                                                        if (verbose == true)\
                                                        {\
                                                            std::cout<<v(i + REPLIKA_PK_INDEX_ORIGINATES_AT, false).c_str()<<" --> "<<v[v(i + REPLIKA_PK_INDEX_ORIGINATES_AT, false)]->index<<" -- "<<v(j + REPLIKA_PK_INDEX_ORIGINATES_AT, false).c_str()<<" --> "<<v[v(j + REPLIKA_PK_INDEX_ORIGINATES_AT, false)]->index<<std::endl;\
                                                        }\
                                                        ptr[k + SKIP_GRAM_PAIR_TARGET_INDEX] = v[v(i + REPLIKA_PK_INDEX_ORIGINATES_AT, false)]->index;\
                                                        ptr[k + SKIP_GRAM_PAIR_CONTEXT_INDEX] = v[v(j + REPLIKA_PK_INDEX_ORIGINATES_AT, false)]->index;\
                                                        \
                                                        k = k +  SKIP_GRAM_DEFAULT_PAIR_SIZE;\
                                                    }\
                                                    if (verbose)\
                                                    {\
                                                        /*std::cout<<" - - - ";*/\
                                                    }\
                                                    for (skip_gram_pairs::number_of_pairs_type j = (i + 1); j < (i + SKIP_GRAM_WINDOW_SIZE + 1); j++)\
                                                    {\
                                                        if ((j < v.len(false)))\
                                                        {\
                                                            if (verbose)\
                                                            {\
                                                                /*std::cout<<v(j - REPLIKA_PK_INDEX_ORIGINATES_AT, false).c_str()<<" ";*/\
                                                            }\
                                                            /*n = n + 1*/ /*SKIP_GRAM_DEFAULT_PAIR_SIZE*/;\
                                                            /*std::cout<<v(i - REPLIKA_PK_INDEX_ORIGINATES_AT, false).c_str()<<" -- "<<v(j - REPLIKA_PK_INDEX_ORIGINATES_AT, false).c_str()<<std::endl;*/\
                                                            if (verbose == true)\
                                                            {\
                                                                std::cout<<v(i - REPLIKA_PK_INDEX_ORIGINATES_AT, false).c_str()<<" --> "<<v[v(i - REPLIKA_PK_INDEX_ORIGINATES_AT, false)]->index<<" -- "<<v(j - REPLIKA_PK_INDEX_ORIGINATES_AT, false).c_str()<<" --> "<<v[v(j - REPLIKA_PK_INDEX_ORIGINATES_AT, false)]->index<<std::endl;\
                                                            }\
                                                            ptr[k + SKIP_GRAM_PAIR_TARGET_INDEX] = v[v(i + REPLIKA_PK_INDEX_ORIGINATES_AT, false)]->index;\
                                                            ptr[k + SKIP_GRAM_PAIR_CONTEXT_INDEX] = v[v(j + REPLIKA_PK_INDEX_ORIGINATES_AT, false)]->index;\
                                                            \
                                                            k = k +  SKIP_GRAM_DEFAULT_PAIR_SIZE;\
                                                        }\
                                                        /*std::cout<<v(i - REPLIKA_PK_INDEX_ORIGINATES_AT, false).c_str()<<" -- "<<v(j - REPLIKA_PK_INDEX_ORIGINATES_AT, false).c_str()<<std::endl;*/\
                                                    }\
                                                    if (verbose)\
                                                    {\
                                                        /*std::cout<<std::endl;*/\
                                                    }\
                                                }\
                                                struct skip_gram_pairs obj(reinterpret_cast<corpus::corpus_index_type (*)[SKIP_GRAM_DEFAULT_PAIR_SIZE]>(ptr), n);\
                                                p = obj;\
                                                alloc_obj.deallocate(reinterpret_cast<char*>(ptr));\
                                                /*--------------------------------------------------------------------------------------*/\
                                             }\

/*
    @a, array of type @tp and size @n 
    @m, multiplyer of type @tp
    @n, size of array @a of type @tp
    @p, pointer to @tp, the new array of the same size as the @a. Each value of @a is multiplyed by @m and it is stored in @p
    @tp, type of multiplyer @m and of array @a and @p
 */
#define MULTIPLY_ARRAY_BY_SCALAR(a, m, n, p, tp)  {\
                                                    cc_tokenizer::allocator<char> alloc_obj;\
                                                    p = reinterpret_cast<tp*>(alloc_obj.allocate(sizeof(tp)*n));\
                                                    for (cc_tokenizer::string_character_traits<char>::size_type i = 0; i < n; i++)\
                                                    {\
                                                       p[i] = reinterpret_cast<tp*>(a)[i]*m;\
                                                    }\
                                                  }\

/*
    EPOCH?    
    In the context of training a machine learning model, an epoch is defined as a complete pass over the entire training dataset during training.
    One epoch is completed when the model has made one update to the weights based on each training sample in the dataset.
    In other words, during one epoch, the model has seen every example in the dataset once and has made one update to the model parameters for each example.     
 */
/*
    Training loop
    @p, inctance of SKIP_GRAM_PAIRS
    @numc, instance of class numc
    @vocab, instance of class corpus. Vocabulary = corpus - redundancy
    @epoch, training epochs. The number of loops for the main training loop
    @lr, learning rate
    @rs, 
    @verbose, when true then put more words than necessary
 */
#define SKIP_GRAM_TRAINING_LOOP(W1, W2, pairs, numc_obj, vocab, epoch, lr, rs, verbose)  {\
                                                        cc_tokenizer::allocator<char> alloc_obj;\
                                                        SKIP_GRAM_PAIRS copy_of_pairs = pairs;\
                                                        \
                                                        double* product_of_lr_and_grad_W1 = NULL;\
                                                        double* product_of_lr_and_grad_W2 = NULL;\
                                                        /* L1 related detail starts here */\
                                                        double* numcy_signed_of_target_in_W1 = NULL;\
                                                        double* product_of_numcy_signed_of_target_in_W1_and_rs = NULL;\
                                                        double* numcy_signed_of_W2 = NULL;\
                                                        double* product_of_numcy_signed_of_W2_and_rs = NULL;\
                                                        /* L1 related detail ends here */\
                                                        /* L2 related detail starts here */\
                                                        /*double* product_of_rs_and_W1 = NULL;*/\
                                                        /*double* product_of_rs_and_W2 = NULL;*/\
                                                        /* L2 related detail ends here */\
                                                        for (unsigned long i = 0; i < epoch; i++)\
                                                        {\
                                                            double epoch_loss = 0;\
                                                            /*\
                                                                The purpose of shuffling the pairs array is to ensure that the training data is randomly ordered before training the skip-gram model.\
                                                                This helps to avoid bias that could occur if the data was ordered in some non-random way, such as if all of the data for a certain context\
                                                                word occurred before or after the data for another context word. By shuffling the data,\
                                                                the model can be trained on a more representative sample of the data, which can lead to better results.\
                                                             */\
                                                            copy_of_pairs.shuffle();\
                                                            skip_gram_pairs::number_of_pairs_type j = 0;\
                                                            corpus::corpus_index_type* pair = NULL;\
                                                            while ((pair = copy_of_pairs[j + REPLIKA_PK_INDEX_ORIGINATES_AT]) != NULL)\
                                                            {\
                                                                if (verbose == true)\
                                                                {\
                                                                    std::cout<<pair[SKIP_GRAM_PAIR_TARGET_INDEX]<<", "<<vocab[pair[SKIP_GRAM_PAIR_TARGET_INDEX]].c_str()<<" - "<<pair[SKIP_GRAM_PAIR_CONTEXT_INDEX]<<", "<<vocab[pair[SKIP_GRAM_PAIR_CONTEXT_INDEX]].c_str()<<std::endl;\
                                                                }\
                                                                forward_propogation<double> fp = forward(pair[SKIP_GRAM_PAIR_TARGET_INDEX], W1, W2, vocab, numc_obj);\
                                                                backward_propogation<double> bp = backward(pair[SKIP_GRAM_PAIR_TARGET_INDEX], pair[SKIP_GRAM_PAIR_CONTEXT_INDEX], W1, W2, vocab, numc_obj, fp);\
                                                                \
                                                                \
                                                                /* L1 related detail starts here */\
                                                                numcy_signed_of_target_in_W1 = numc_obj.sign(W1[SKIP_GRAM_PAIR_TARGET_INDEX], SKIP_GRAM_HIDDEN_SIZE);\
                                                                MULTIPLY_ARRAY_BY_SCALAR(numcy_signed_of_target_in_W1, rs, SKIP_GRAM_HIDDEN_SIZE, product_of_numcy_signed_of_target_in_W1_and_rs, double);\
                                                                SUM_OF_TWO_ARRAYS(bp.grad_W1, product_of_numcy_signed_of_target_in_W1_and_rs, bp.grad_W1, 1, SKIP_GRAM_HIDDEN_SIZE, double);\
                                                                numcy_signed_of_W2 = numc_obj.sign(W2, SKIP_GRAM_HIDDEN_SIZE*vocab.len());\
                                                                MULTIPLY_ARRAY_BY_SCALAR(numcy_signed_of_W2, rs, SKIP_GRAM_HIDDEN_SIZE*vocab.len(), product_of_numcy_signed_of_W2_and_rs, double);\
                                                                SUM_OF_TWO_ARRAYS(bp.grad_W2, product_of_numcy_signed_of_W2_and_rs, bp.grad_W2, SKIP_GRAM_HIDDEN_SIZE, vocab.len(), double);\
                                                                /* L1 related detail ends here */\
                                                                \
                                                                \
                                                                /* L2 related detail starts here */\
                                                                /* Apply L2 regularization(Ridge Regression) to the gradients */\
                                                                /* In L2 regularization, you add a term proportional to the weights themselves to the gradients. */\
                                                                /* This term is computed by multiplying the regularization strength(rs) with the weights. */\
                                                                /* The goal is to penalize large weight values, which helps prevent overfitting. */\
                                                                /*MULTIPLY_ARRAY_BY_SCALAR(W1[SKIP_GRAM_PAIR_TARGET_INDEX], rs, SKIP_GRAM_HIDDEN_SIZE, product_of_rs_and_W1, double);*/\
                                                                /*SUM_OF_TWO_ARRAYS(bp.grad_W1, product_of_rs_and_W1, bp.grad_W1, 1, SKIP_GRAM_HIDDEN_SIZE, double);*/\
                                                                /*MULTIPLY_ARRAY_BY_SCALAR(W2, rs, SKIP_GRAM_HIDDEN_SIZE*vocab.len(), product_of_rs_and_W2, double);*/\
                                                                /*SUM_OF_TWO_ARRAYS(bp.grad_W2, product_of_rs_and_W2, bp.grad_W2, SKIP_GRAM_HIDDEN_SIZE, vocab.len(), double);*/\
                                                                /* L2 related detail ends here */\
                                                                \
                                                                \
                                                                MULTIPLY_ARRAY_BY_SCALAR(bp.grad_W1, lr, vocab.len()*SKIP_GRAM_HIDDEN_SIZE, product_of_lr_and_grad_W1, double);\
                                                                MULTIPLY_ARRAY_BY_SCALAR(bp.grad_W2, lr, vocab.len()*SKIP_GRAM_HIDDEN_SIZE, product_of_lr_and_grad_W2, double);\
                                                                SUBTRACT_ARRAY_FROM_ARRAY(W1, product_of_lr_and_grad_W1, vocab.len()*SKIP_GRAM_HIDDEN_SIZE, double);\
                                                                SUBTRACT_ARRAY_FROM_ARRAY(W2, product_of_lr_and_grad_W2, vocab.len()*SKIP_GRAM_HIDDEN_SIZE, double);\
                                                                \
                                                                /*epoch_loss = epoch_loss + (-1*log(fp.y_pred[copy_of_pairs[j][SKIP_GRAM_PAIR_CONTEXT_INDEX]]));*/\
                                                                epoch_loss = epoch_loss + (-1*log(fp.y_pred[pair[SKIP_GRAM_PAIR_CONTEXT_INDEX]]));\
                                                                \
                                                                alloc_obj.deallocate(reinterpret_cast<char*>(fp.y_pred));\
                                                                alloc_obj.deallocate(reinterpret_cast<char*>(fp.h));\
                                                                alloc_obj.deallocate(reinterpret_cast<char*>(bp.grad_W1));\
                                                                alloc_obj.deallocate(reinterpret_cast<char*>(bp.grad_W2));\
                                                                alloc_obj.deallocate(reinterpret_cast<char*>(pair));\
                                                                \
                                                                alloc_obj.deallocate(reinterpret_cast<char*>(product_of_lr_and_grad_W1));\
                                                                alloc_obj.deallocate(reinterpret_cast<char*>(product_of_lr_and_grad_W2));\
                                                                /* L1 related detail starts here */\
                                                                alloc_obj.deallocate(reinterpret_cast<char*>(numcy_signed_of_target_in_W1));\
                                                                alloc_obj.deallocate(reinterpret_cast<char*>(product_of_numcy_signed_of_target_in_W1_and_rs));\
                                                                alloc_obj.deallocate(reinterpret_cast<char*>(numcy_signed_of_W2));\
                                                                alloc_obj.deallocate(reinterpret_cast<char*>(product_of_numcy_signed_of_W2_and_rs));\
                                                                numcy_signed_of_target_in_W1 = NULL;\
                                                                product_of_numcy_signed_of_target_in_W1_and_rs = NULL;\
                                                                numcy_signed_of_W2 = NULL;\
                                                                product_of_numcy_signed_of_W2_and_rs = NULL;\
                                                                /* L1 related detail ends here */\
                                                                /* L2 related detail starts here */\
                                                                /*alloc_obj.deallocate(reinterpret_cast<char*>(product_of_rs_and_W1));*/\
                                                                /*alloc_obj.deallocate(reinterpret_cast<char*>(product_of_rs_and_W2));*/\
                                                                /* L2 related detail ends here */\
                                                                pair = NULL;\
                                                                fp.y_pred = NULL;\
                                                                fp.h = NULL;\
                                                                bp.grad_W1 = NULL;\
                                                                bp.grad_W2 = NULL;\
                                                                product_of_lr_and_grad_W1 = NULL;\
                                                                product_of_lr_and_grad_W2 = NULL;\
                                                                /*product_of_rs_and_W1 = NULL;*/\
                                                                /*product_of_rs_and_W2 = NULL;*/\
                                                                j = j + 1;\
                                                            }\
                                                            /*if (verbose)*/\
                                                            {\
                                                                std::cout<<"Current epoch is = "<<i + 1<<" and epoch loss is "<<epoch_loss/pairs.len()<<std::endl;\
                                                            }\
                                                        }\
                                                        \
                                                     }\

/*
   @tp, type of p
   @p, subarray pointer
   @n, sizeof(*p)
   @w, weight ndarray
   @wr, rows of w
   @wc, columns of w
   @c, context array
   @cc, c columns
 */
#define  SUBARRAY(tp, p, w, wr, wc, c, cc)    {\
                                                  cc_tokenizer::allocator<char> alloc_obj_local;\
                                                  p = reinterpret_cast<tp(*)[wc]>(alloc_obj_local.allocate(sizeof(**p)*cc*wc));\
                                                  for (cc_tokenizer::string_character_traits<char>::size_type i = 0; i < cc; i++)\
                                                  {\
                                                      /*std::cout<<c[i]<<" ";*/\
                                                      for (cc_tokenizer::string_character_traits<char>::size_type j = 0; j < wc; j++)\
                                                      {\
                                                        /*std::cout<<w[c[i]][j]<<" ";*/\
                                                        /* Each word of vocabulary is assigned an index. The index value begins at REPLIKA_PK_INDEX_ORIGINATES_AT */\
                                                        /* It can be any value other than 0 as well. */\
                                                        /* The w array has embeddings of the vocabulary words, it is a two dimentional array and in which each line is a single word embedding for a single vocabulary word.*/\
                                                        /* The is why we need to deduct we need to deduct REPLIKA_PK_INDEX_ORIGINATES_AT */\
                                                        p[i][j] = w[c[i] - REPLIKA_PK_INDEX_ORIGINATES_AT][j];\
                                                      }\
                                                      /*std::cout<<std::endl;*/\
                                                  }\
                                              }\

/*
  @a, array of type @tp and of size @n
  @s, scalar value of type @tp. This value gets substracted from each element of @a
  @n, number of elements in an array @a
  @p, pointer to @tp, the new array of the same size as the @a. Each value of @a is reduced by @s 
  @tp, type of each element of aray @a and type of scalar @s

  On completion this macro would return a new array of type @tp with number of elements equals to @n
 */
#define SUBTRACT_SCALAR(a, s, n, p, tp) {\
                                          cc_tokenizer::allocator<char> alloc_obj_local;\
                                          p = reinterpret_cast<tp*>(alloc_obj_local.allocate(sizeof(tp)*n));\
                                          for (cc_tokenizer::string_character_traits<char>::size_type i = 0; i < n; i++)\
                                          {\
                                            p[i] = a[i] - s;\
                                          }\
                                        }\

/*
    @a, array of type @tp and of size @r rows and @c columns
    @b, array of type @tp and of size @r rows and @c columns
    @d, destintion array of same shape as @a and @b
    @r, number of rows in arrays @a, @b, @d
    @c, number of columns in arrays @a, @b, @d
    @tp, type of each element of aray @a and type of scalar @b
 */
#define SUM_OF_TWO_ARRAYS(a, b, d, r, c, tp) {\
                                                    \
                                                    \
                                                    for (cc_tokenizer::String<char>::size_type i = 0; i < r; i++)\
                                                    {\
                                                        for (cc_tokenizer::String<char>::size_type j = 0; j < c; j++)\
                                                        {\
                                                            d[i*c +j] =  a[i*c + j] + b[i*c + j];\
                                                        }\
                                                    }\
                                             }\

/*
  TODO, details about how does it work
   u = h * W2 
   h is a row vector with shape (1, REPLIKA_HIDDEN_SIZE) 
   W2 is vector and has shape (REPLIKA_HIDDEN_SIZE, corpus::len())
   u is a row vector as well and has shape (1, corpus::len())
 */
template<typename E>
E* softmax(E* u, class corpus& corpus_obj, class numc& numc_obj)
{   
    cc_tokenizer::allocator<char> alloc_obj;

    E* u_minus_max = NULL; // Stores all elements of a u, minus an scalar.
    //SUBTRACT_SCALAR(u, numc_obj.max<double>(u, REPLIKA_HIDDEN_SIZE), REPLIKA_HIDDEN_SIZE, u_minus_max, E);
    SUBTRACT_SCALAR(u, numc_obj.max<double>(u, corpus_obj.len()), corpus_obj.len(), u_minus_max, E);

    //E* e_u_minus_max = numc_obj.exp<double>(u_minus_max, REPLIKA_HIDDEN_SIZE);
    /*
        Each element of array is transformed into a base^power form
     */
    E* e_u_minus_max = numc_obj.exp<double>(u_minus_max, corpus_obj.len());

    E e_u_minus_max_sum = 0; 
    //ARRAY_SUM(e_u_minus_max, REPLIKA_HIDDEN_SIZE, e_u_minus_max_sum);
    ARRAY_SUM(e_u_minus_max, corpus_obj.len(), e_u_minus_max_sum);

    double* e_u_minus_max_divided_by_e_u_minus_max_sum;
    //DIVIDE_ARRAY_BY_SCALAR(e_u_minus_max, e_u_minus_max_sum, REPLIKA_HIDDEN_SIZE, e_u_minus_max_divided_by_e_u_minus_max_sum, double);
    DIVIDE_ARRAY_BY_SCALAR(e_u_minus_max, e_u_minus_max_sum, corpus_obj.len(), e_u_minus_max_divided_by_e_u_minus_max_sum, double);

    alloc_obj.deallocate(reinterpret_cast<char*>(u_minus_max));
    alloc_obj.deallocate(reinterpret_cast<char*>(e_u_minus_max));
    
    /*
       array of exp(u - max)/sum_of(exp(u - max)) and shape of this array is (1, REPLIKA_VOCABULARY_LENGTH or vocab.len)
     */
    return e_u_minus_max_divided_by_e_u_minus_max_sum;
}

/*
    for (skip_gram_pairs::number_of_pairs_type i = 0; i < pairs.len(); i++)
    {
        corpus::corpus_index_type* ptr = pairs[i];

        std::cout<<"Target-Word -> "<<ptr[0]<<" - "<<vocab[ptr[0]].c_str()<<", "<<"Context-Word -> "<<ptr[1]<<" - "<<vocab[ptr[1]].c_str()<<std::endl;

        alloc_obj.deallocate(reinterpret_cast<char*>(ptr));
    }
 */                                             
struct skip_gram_pairs 
{   
    typedef unsigned long number_of_pairs_type;
    //typedef long number_of_pairs_type;

    number_of_pairs_type len(void)
    {
        return n;
    }

    skip_gram_pairs() : pairs(NULL), n(0)
    {                        
    }

    // Copy constructor
    skip_gram_pairs(const skip_gram_pairs& other)
    {
        cc_tokenizer::allocator<char> alloc_obj;

        pairs = reinterpret_cast<corpus::corpus_index_type (*)[SKIP_GRAM_DEFAULT_PAIR_SIZE]>(alloc_obj.allocate(sizeof(corpus::corpus_index_type)*other.n*SKIP_GRAM_DEFAULT_PAIR_SIZE));
        
        for (unsigned long i = 0; i < other.n; i++)
        {
            for (unsigned char j = 0; j < SKIP_GRAM_DEFAULT_PAIR_SIZE; j++)
            {
                pairs[i][j] = other.pairs[i][j];
            }
        }

        n = other.n;
    }

    skip_gram_pairs(corpus::corpus_index_type (*target_context_pairs)[SKIP_GRAM_DEFAULT_PAIR_SIZE], number_of_pairs_type n_pairs) : n(n_pairs)
    {
        cc_tokenizer::allocator<char> alloc_obj;

        pairs = reinterpret_cast<corpus::corpus_index_type (*)[SKIP_GRAM_DEFAULT_PAIR_SIZE]>(alloc_obj.allocate(sizeof(corpus::corpus_index_type)*n*SKIP_GRAM_DEFAULT_PAIR_SIZE));

        for (corpus::corpus_index_type i = 0; i < n; i++)
        {
            for (unsigned char j = 0; j < SKIP_GRAM_DEFAULT_PAIR_SIZE; j++)
            {
                pairs[i][j] = target_context_pairs[i][j];
            }
        }
    }

    // Assignmnet operator
    skip_gram_pairs& operator= (const skip_gram_pairs& other)
    {
        cc_tokenizer::allocator<char> alloc_obj;
        
        // Check for self-assignment
        if (this == &other)
        {
            return *this;
        }

        if (pairs != NULL)
        {
            alloc_obj.deallocate(reinterpret_cast<char*>(pairs));

            pairs = NULL;
            n = 0;
        }
        
        pairs = reinterpret_cast<corpus::corpus_index_type (*)[SKIP_GRAM_DEFAULT_PAIR_SIZE]>(alloc_obj.allocate(sizeof(corpus::corpus_index_type)*other.n*SKIP_GRAM_DEFAULT_PAIR_SIZE));
        
        for (unsigned long i = 0; i < other.n; i++)
        {
            for (unsigned char j = 0; j < SKIP_GRAM_DEFAULT_PAIR_SIZE; j++)
            {
                pairs[i][j] = other.pairs[i][j];
            }
        }

        n = other.n;

        return *this;
    }

    corpus::corpus_index_type* operator[] (number_of_pairs_type i)
    {
        cc_tokenizer::allocator<char> alloc_obj;

        corpus::corpus_index_type* ptr = NULL;

        if (i < len())
        {            
            ptr = reinterpret_cast<corpus::corpus_index_type*>(alloc_obj.allocate(sizeof(corpus::corpus_index_type)*SKIP_GRAM_DEFAULT_PAIR_SIZE));
            for (unsigned char j = 0; j < SKIP_GRAM_DEFAULT_PAIR_SIZE; j++)
            {
                ptr[j] = pairs[i][j];
            }
        }

        return ptr;
    }

    void shuffle(void)
    {
        class numc numc_obj;

        /*
        for (int i = 0; i < n; i++)
        {
            for (int j = 0; j < SKIP_GRAM_WINDOW_SIZE; j++)
            {
                std::cout<<pairs[i][j]<<" ";
            }

            std::cout<<std::endl;
        }
         */

        numc_obj.shuffle<corpus::corpus_index_type>(reinterpret_cast<corpus::corpus_index_type*>(pairs), {SKIP_GRAM_WINDOW_SIZE, n, NULL, NULL});
        
        /*
        std::cout<<" --------------------------- --------------------------- ----------------- ---- -  --- -- - - - -- - ------------------- "<<std::endl;

        for (int i = 0; i < n; i++)
        {
            for (int j = 0; j < SKIP_GRAM_WINDOW_SIZE; j++)
            {
                std::cout<<pairs[i][j]<<" ";
            }

            std::cout<<std::endl;
        }
         */
    }

    ~skip_gram_pairs()
    {
        cc_tokenizer::allocator<char> alloc_obj;

        alloc_obj.deallocate(reinterpret_cast<char*>(pairs));

        pairs = NULL;
        n = 0;        
    }

    private:
        number_of_pairs_type n;
        corpus::corpus_index_type (*pairs)[SKIP_GRAM_DEFAULT_PAIR_SIZE];            
}; 

typedef struct skip_gram_pairs SKIP_GRAM_PAIRS;


/*
    def backward(center_word_index, y_pred, context_word_index):
        grad_u = y_pred - one_hot(context_word_index, len(vocab))
        grad_W2 = np.outer(h, grad_u)
        grad_h = np.dot(grad_u, W2.T)
        grad_W1 = np.zeros_like(W1)
        grad_W1[center_word_index] += grad_h
        return grad_W1, grad_W2
 */
/*
    Shape of u is (1, vocab.len())
    Shape of h is (1, SKIP_GRAM_HIDDEN_SIZE)
 */
/*
    def backward(center_word_index, y_pred, context_word_index, u):
    grad_u = y_pred - one_hot(context_word_index, len(vocab))
    grad_W2 = np.outer(u, grad_u)  # Use u instead of h for gradient computation
    grad_h = np.dot(grad_u, W2.T)
    grad_W1 = np.zeros_like(W1)
    grad_W1[center_word_index] += grad_h
    
    # Backpropagation with intermediate gradients
    grad_h_with_respect_to_u = grad_h
    grad_h_with_respect_to_h = np.dot(grad_h_with_respect_to_u, W2.T)
    grad_h_with_respect_to_center_word = grad_h_with_respect_to_h
    
    return grad_W1, grad_W2, grad_h_with_respect_to_center_word
 */
struct backward_propogation<double> backward(corpus::corpus_index_type target_or_center_word_index_in_vocabulary, corpus::corpus_index_type context_word_index_in_vocabulary, double (*W1)[SKIP_GRAM_HIDDEN_SIZE], double* W2, class corpus& vocab, class numc& numc_obj, forward_propogation<double>& fp)
{
    cc_tokenizer::allocator<char> alloc_obj;

    //std::cout<<"Hello"<<std::endl;
    //std::cout<<vocab[context_word_index_in_vocabulary].c_str()<<std::endl;
    
    /* The hot_one_array is row vector, and has shape (1, vocab.len = REPLIKA_VOCABULARY_LENGTH a.k.a no redundency) */
    double* hot_one_array = one_hot<double>(context_word_index_in_vocabulary - REPLIKA_PK_INDEX_ORIGINATES_AT, vocab.len(), numc_obj);
    /* The shape of grad_u is the same as y_pred which is (1, len(vocab)) */
    double* grad_u = numc_obj.subtract_matrices<double>(fp.y_pred, hot_one_array, vocab.len());
    /* The grad_W2 has shape/dimensions of (REPLIKA_HIDDEN_SIZE, REPLIKA_VOCABULARY_LENGTH) */
    //double* grad_W2 = numc_obj.outer<double>(fp.h, grad_u, SKIP_GRAM_HIDDEN_SIZE, vocab.len());
    // NEW
    // Shape of u is (1, vocab.len())
    // The shape of grad_u is (1, len(vocab))
    /* The grad_W2 has shape/dimensions of (REPLIK_VOCABULARY_LENGTH, REPLIKA_VOCABULARY_LENGTH) = (vocab.len(), vocab.len()) */ 
    /* The outer product computes how much each element in fp.u contributed to the gradient grad_u */
    /*
        In this statement, the outer product is calculated between the intermediate gradient fp.u (from the forward pass) and grad_u (the calculated gradient with respect to the predicted probabilities). This suggests that the gradient of W2 is being influenced by how much the intermediate gradient fp.u contributes to the gradient of the predicted probabilities.
     */
    double* grad_W2 = numc_obj.outer<double>(fp.u, grad_u, vocab.len(), vocab.len());
    /*
        Shape of W2_T is (REPLIKA_VOCABULARY_LENGTH, REPLIKA_HIDDEN_SIZE), the matrix W2 has shape of (REPLIKA_HIDDEN_SIZE, REPLIKA_VOCABULARY_LENGTH)
     */ 
    double* W2_T = numc_obj.transpose_matrix<double>(reinterpret_cast<double*>(W2), SKIP_GRAM_HIDDEN_SIZE, vocab.len());
    /*
        H is the number of neurons in the hidden layer, where V is the vocabulary size. 
        //The grad_u has a shape of (1, V) and W2.T has a shape of (V, H)
         //The resulting matrix grad_h would have a shape of (1, H) or (1, REPLIKA_HIDDEN_SIZE)
         NEW
        The grad_u has a shape of (V, V) and W2.T has a shape of (V, H)
        //The resulting matrix grad_h would have a shape of (V, H) or (REPLIKA_VOCABULARY_LENGTH, REPLIKA_HIDDEN_SIZE)
     */
    double* grad_h = numc_obj.dot(grad_u, W2_T, {vocab.len(), 1, NULL, NULL}, {SKIP_GRAM_HIDDEN_SIZE, vocab.len(), NULL, NULL}, REPLIKA_PK_NUMC_YAXIS);
    /*
        The shape/dimensions of grad_W1 is (REPLIKA_VOCABULARY_LENGTH, REPLIKA_HIDDEN_SIZE)
     */
    double* grad_W1 = numc_obj.zeros<double>(2, vocab.len(), SKIP_GRAM_HIDDEN_SIZE);
    
    /*
        grad_W1[center_word_index] += grad_h

        grad_W1 has shape (REPLIKA_VOCABULARY_LENGTH, REPLIKA_HIDDEN_SIZE) or (len(vocab), SKIP_GRAM_HIDDEN_SIZE) same as W1
        grad_h has shape (1, REPLIKA_HIDDEN_SIZE) or (1, SKIP_GRAM_HIDDEN_SIZE) same as one row of W1 

        NEW 
        grad_W1 has shape (REPLIKA_VOCABULARY_LENGTH, REPLIKA_HIDDEN_SIZE) or (len(vocab), SKIP_GRAM_HIDDEN_SIZE) same as W1
        grad_h has shape (REPLIKA_VOCABULARY_LENGTH,, REPLIKA_HIDDEN_SIZE) or (REPLIKA_VOCABULARY_LENGTH,, SKIP_GRAM_HIDDEN_SIZE) same as one row of W1 
     */

    //SUM_OF_TWO_ARRAYS(reinterpret_cast<double(*)[SKIP_GRAM_HIDDEN_SIZE]>(grad_W1)[target_or_center_word_index_in_vocabulary], grad_h, reinterpret_cast<double(*)[SKIP_GRAM_HIDDEN_SIZE]>(grad_W1)[target_or_center_word_index_in_vocabulary], 1, SKIP_GRAM_HIDDEN_SIZE, double);    
    SUM_OF_TWO_ARRAYS(reinterpret_cast<double(*)[SKIP_GRAM_HIDDEN_SIZE]>(grad_W1)[target_or_center_word_index_in_vocabulary], reinterpret_cast<double(*)[SKIP_GRAM_HIDDEN_SIZE]>(grad_h)[target_or_center_word_index_in_vocabulary], reinterpret_cast<double(*)[SKIP_GRAM_HIDDEN_SIZE]>(grad_W1)[target_or_center_word_index_in_vocabulary], 1, SKIP_GRAM_HIDDEN_SIZE, double);

    /*
        # Backpropagation with intermediate gradients
        grad_h_with_respect_to_u = grad_h
        grad_h_with_respect_to_h = np.dot(grad_h_with_respect_to_u, W2.T)
        grad_h_with_respect_to_center_word = grad_h_with_respect_to_h
     */
    /*
        grad_h, has shape (REPLIKA_VOCABULARY_LENGTH,, REPLIKA_HIDDEN_SIZE)
        W2.T, Shape of W2_T is (REPLIKA_VOCABULARY_LENGTH, REPLIKA_HIDDEN_SIZE)
        grad_h_with_respect_to_center_word, shape is (REPLIKA_VOCABULARY_LENGTH,, REPLIKA_HIDDEN_SIZE)
     */
    double* grad_h_with_respect_to_center_word = numc_obj.dot(grad_h, W2_T, {vocab.len(), SKIP_GRAM_HIDDEN_SIZE, NULL, NULL}, {vocab.len(), SKIP_GRAM_HIDDEN_SIZE, NULL, NULL}, REPLIKA_PK_NUMC_YAXIS);

    alloc_obj.deallocate(reinterpret_cast<char*>(hot_one_array));
    alloc_obj.deallocate(reinterpret_cast<char*>(grad_u));
    alloc_obj.deallocate(reinterpret_cast<char*>(W2_T));
    alloc_obj.deallocate(reinterpret_cast<char*>(grad_h));

    return {grad_W1, grad_W2, grad_h_with_respect_to_center_word};
    /*
    for (int i = 0; i < SKIP_GRAM_HIDDEN_SIZE; i++)
    {
        //std::cout<<reinterpret_cast<double(*)[SKIP_GRAM_HIDDEN_SIZE]>(grad_W1)[target_or_center_word_index_in_vocabulary][i]<<" ";

        reinterpret_cast<double(*)[SKIP_GRAM_HIDDEN_SIZE]>(grad_W1)[target_or_center_word_index_in_vocabulary][i] = reinterpret_cast<double(*)[SKIP_GRAM_HIDDEN_SIZE]>(grad_W1)[target_or_center_word_index_in_vocabulary][i] + grad_h[i];
    }
     */

    /*
    for (int i = 0; i < SKIP_GRAM_HIDDEN_SIZE; i++)
    {
        std::cout<<reinterpret_cast<double(*)[SKIP_GRAM_HIDDEN_SIZE]>(grad_W1)[target_or_center_word_index_in_vocabulary][i]<<" ";

        //reinterpret_cast<double(*)[SKIP_GRAM_HIDDEN_SIZE]>(grad_W1)[target_or_center_word_index_in_vocabulary][i] = reinterpret_cast<double(*)[SKIP_GRAM_HIDDEN_SIZE]>(grad_W1)[target_or_center_word_index_in_vocabulary][i] + grad_h[i];
    }
    

    std::cout<<std::endl;
    */

    //std::cout<<vocab.get_size_of_context_line()<<std::endl;

    /*
        The ones in place of context indexes, shape of nes_in_place_of_context_indexes is (1, REPLIKA_HIDDEN_SIZE)
     */
    //double* ones_in_place_of_context_indexes = numc_obj.ones<double>(1, vocab.get_size_of_context_line());
    /*
        The shape of matrix(outer_of_grad_h_and_ones_in_place_of_context_indexes) is (REPLIKA_HIDDEN_SIZE, corpus_obj.get_size_of_context_line())
     */
    /*
        The shape of grad_h (1, REPLIKA_HIDDEN_SIZE), the shape of ones_in_place_of_context_indexes is (1, corpus_obj.get_size_of_context_line())
        The outer of grad_h and ones in place of context indexes and the shape of outer_of_grad_h_and_ones_in_place_of_context_indexes (REPLIKA_HIDDEN_SIZE, corpus_obj.get_size_of_context_line())
     */
    //double* outer_of_grad_h_and_ones_in_place_of_context_indexes = numc_obj.outer<double>(grad_h, ones_in_place_of_context_indexes, SKIP_GRAM_HIDDEN_SIZE, vocab.get_size_of_context_line());
    /*
        The transpose of outer_of_grad_h_and_ones_in_place_of_context_indexes and it has shape of (corpus_obj.get_size_of_context_line(), REPLIKA_HIDDEN_SIZE)
     */
    //double* transpose_of_outer_of_grad_h_and_ones_in_place_of_context_indexes = numc_obj.transpose_matrix<double>(outer_of_grad_h_and_ones_in_place_of_context_indexes, SKIP_GRAM_HIDDEN_SIZE, vocab.get_size_of_context_line());
    /*
        It is a two dimensional array... the grad_W1_subarray
        number of lines of this array is equal to the returned value of this funtion call corpus_obj.get_size_of_context_line() and columns equal to REPLIKA_HIDDEN_SIZE
     */
    //double (*grad_W1_subarray)[SKIP_GRAM_HIDDEN_SIZE];    
    //SUBARRAY(double, grad_W1_subarray, reinterpret_cast<double(*)[SKIP_GRAM_HIDDEN_SIZE]>(grad_W1), vocab.len(), SKIP_GRAM_HIDDEN_SIZE, vocab[context_word_index_in_vocabulary], vocab.get_size_of_context_line());

    //std::cout<<"Hello"<<std::endl;
    //std::cout<<vocab[context_word_index_in_vocabulary].c_str()<<std::endl;
}
#endif

#endif

