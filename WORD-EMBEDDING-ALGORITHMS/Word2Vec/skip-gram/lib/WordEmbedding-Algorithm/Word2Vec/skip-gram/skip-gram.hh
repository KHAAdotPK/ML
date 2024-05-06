/*
    skip-gram.hh
    Q@khaa.pk
 */

#ifndef SKIP_GRAM_SKIP_GRAM_HH
#define SKIP_GRAM_SKIP_GRAM_HH

#include "header.hh"

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
    Collective<T> m;
    try 
    {
        m = Numcy::max(a);
    }
    catch (ala_exception& e)
    {
        throw ala_exception(cc_tokenizer::String<char>("softmax() -> ") + cc_tokenizer::String<char>(e.what()));
    }
        
    // a_m, a minus m
    Collective<T> a_m;
    try 
    {
        a_m = Numcy::subtract(a, m);
    }
    catch (ala_exception& e)
    {
        throw ala_exception(cc_tokenizer::String<char>("softmax() -> ") + cc_tokenizer::String<char>(e.what()));
    }
    
    // e_a_m, exp over a_m
    Collective<T> e_a_m; 
    try
    {        
        e_a_m = Numcy::exp(a);
    }
    catch (ala_exception& e)
    {
        throw ala_exception(cc_tokenizer::String<char>("softmax() -> ") + cc_tokenizer::String<char>(e.what()));  
    }

    // Sum of e_a_m
    Collective<T> s_e_a_m;
    try
    {        
        s_e_a_m = Numcy::sum(e_a_m);
    }
    catch (ala_exception& e)
    {
        throw ala_exception(cc_tokenizer::String<char>("softmax() -> ") + cc_tokenizer::String<char>(e.what()));        
    }
    
    Collective<T> e_u_minus_max_divided_by_e_u_minus_max_sum;
    try
    {     
        e_u_minus_max_divided_by_e_u_minus_max_sum = Numcy::divide(e_a_m, s_e_a_m);
    }
    catch (ala_exception& e)
    {
        throw ala_exception(cc_tokenizer::String<char>("softmax() -> ") + cc_tokenizer::String<char>(e.what()));        
    }
            
    return e_u_minus_max_divided_by_e_u_minus_max_sum;
}

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

/*
    Performs part of the forward propagation step in a Skip-gram model
    
    @W1, embedding matrix (Collective object) 
    @W2, output layer weight matrix (Collective object)
    @vocab, instance of corpus class
    @pair, a pointer to a word pair object (containing center word index and context word indices) 
 */
template <typename T>
forward_propogation<T> forward(Collective<T>& W1, Collective<T>& W2, CORPUS_REF vocab, WORDPAIRS_PTR pair, bool verbose = false) throw (ala_exception)
{      
    if (pair->getCenterWord() > W1.getShape().getDimensionsOfArray().getNumberOfInnerArrays())
    {
        throw ala_exception("forward() Error: Index of center word is out of bounds of W1.");
    }

    T* h = NULL;

    try 
    {
        h = cc_tokenizer::allocator<T>().allocate(W1.getShape().getNumberOfColumns());
    }
    catch (ala_exception& e)
    {        
        throw ala_exception(cc_tokenizer::String<char>("forward() Error: ") + cc_tokenizer::String<char>(e.what()));
    }

    for (cc_tokenizer::string_character_traits<char>::size_type i = 0; i < W1.getShape().getNumberOfColumns(); i++)
    {
        *(h + i) = W1[W1.getShape().getNumberOfColumns()*(pair->getCenterWord() - INDEX_ORIGINATES_AT_VALUE) + i];
        
        /*if (_isnanf(h[i]))
        {        
            throw ala_exception(cc_tokenizer::String<char>("forward() Error: Hidden layer at ") + cc_tokenizer::String<char>("(W1 row) center word index ") +  cc_tokenizer::String<char>(pair->getCenterWord() - INDEX_ORIGINATES_AT_VALUE) + cc_tokenizer::String<char>(" and (column index) i -> ") + cc_tokenizer::String<char>(i) + cc_tokenizer::String<char>(" -> [ ") + cc_tokenizer::String<char>("_isnanf() was true") + cc_tokenizer::String<char>("\" ]"));            
        }*/ 
    }
    
    Collective<T> u;
    try 
    {
        u = Numcy::dot(Collective<T>{h, DIMENSIONS{W1.getShape().getNumberOfColumns(), 1, NULL, NULL}}, W2);
    }
    catch (ala_exception& e)
    {
        throw ala_exception(cc_tokenizer::String<char>("forward() -> ") + cc_tokenizer::String<char>(e.what()));
    }
    
    Collective<T> y_pred = softmax<T>(u);
    
    /*
        Dimensions of forward_propogation::hidden_layer_vector[1, SKIP_GRAM_EMBEDDNG_VECTOR_SIZE], collective is formed using pointer "h" to type T
        Dimensions of forward_propogation::predicted_probabilities[1, "len(vocab_unique)/W2.getShape().getNumberOfColumns()"]. Here it is collective y_pred
        Dimensions of forward_propogation::intermediate_activation[1, "len(vocab_unique)/W2.getShape().getNumberOfColumns()"]. Here it is collective u
     */                
    return forward_propogation<T>{Collective<T>{h, DIMENSIONS{W1.getShape().getNumberOfColumns(), 1, NULL, NULL}}, y_pred, u};
}

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

template <typename E = double>
backward_propogation<E> backward(Collective<E>& W1, Collective<E>& W2, CORPUS_REF vocab, forward_propogation<E>& fp, WORDPAIRS_PTR pair, bool verbose = false) throw (ala_exception)
{       
    /* The hot one array is row vector, and has shape (1, vocab.len = REPLIKA_VOCABULARY_LENGTH a.k.a no redundency) */
    Collective<E> oneHot;
    /*
        Creating a One-Hot Vector, using Numcy::zeros with a shape of (1, vocab.numberOfUniqueTokens()).
        This creates a zero-filled column vector with a length equal to the vocabulary size
     */
    try 
    {
        oneHot = Numcy::zeros(DIMENSIONS{vocab.numberOfUniqueTokens(), 1, NULL, NULL});    
    }
    catch (ala_exception& e)
    {
        throw ala_exception(cc_tokenizer::String<char>("backward() -> ") + cc_tokenizer::String<char>(e.what()));
    }
    
    /*
       The following code block, iterates through the context word indices (left and right) from the pair object.
       For each valid context word index (i), it sets the corresponding element in the oneHot vector to 1.
       This effectively creates a one-hot encoded representation of the context words.
     */
    try
    {        
        for (int i = SKIP_GRAM_WINDOW_SIZE - 1; i >= 0; i--)
        {       
            if (((*(pair->getLeft()))[i] - INDEX_ORIGINATES_AT_VALUE) < vocab.numberOfUniqueTokens())
            {
                oneHot[(*(pair->getLeft()))[i] - INDEX_ORIGINATES_AT_VALUE] = 1;
            }
        }
        for (int i = 0; i < SKIP_GRAM_WINDOW_SIZE; i++)
        {
            if (((*(pair->getRight()))[i] - INDEX_ORIGINATES_AT_VALUE) < vocab.numberOfUniqueTokens())
            {
                oneHot[(*(pair->getRight()))[i] - INDEX_ORIGINATES_AT_VALUE] = 1;
            }        
        }
    }
    catch (ala_exception& e)
    {
        throw ala_exception(cc_tokenizer::String<char>("backward() -> ") + cc_tokenizer::String<char>(e.what()));
    }
    
    /* The shape of grad_u is the same as y_pred (fp.predicted_probabilities) which is (1, len(vocab) without redundency) */
    Collective<E> grad_u;
    try 
    {          
        grad_u = Numcy::subtract<double>(fp.predicted_probabilities, oneHot);
    }
    catch (ala_exception& e)
    {
        throw ala_exception(cc_tokenizer::String<char>("backward() -> ") + cc_tokenizer::String<char>(e.what()));
    }
    
    /*
        Dimensions of grad_u is (1, len(vocab) without redundency)
        Dimensions of fp.intermediate_activation (1, len(vocab) without redundency)

        Dimensions of grad_W2 is (len(vocab) without redundency, len(vocab) without redundency)        
     */
    Collective<E> grad_W2;
    try 
    {        
        grad_W2 = Numcy::outer(fp.intermediate_activation, grad_u);
    }
    catch (ala_exception& e)
    {
        throw ala_exception(cc_tokenizer::String<char>("bacward() -> ") + cc_tokenizer::String<char>(e.what()));
    }

    /*
        Dimensions of W2 is (SKIP_GRAM_EMBEDDNG_VECTOR_SIZE, len(vocab) without redundency)
        Dimensions of W2_T is (len(vocab) without redundency, SKIP_GRAM_EMBEDDNG_VECTOR_SIZE)        
     */
    Collective<E> W2_T;
    try 
    {
        W2_T = Numcy::transpose(W2);
    }
    catch (ala_exception& e)
    { 
        throw ala_exception(cc_tokenizer::String<char>("bacward() -> ") + cc_tokenizer::String<char>(e.what()));
    }

    /*
       Dimensions of grad_u is (1, len(vocab) without redundency)
       Dimensions of W2_T is (len(vocab) without redundency, SKIP_GRAM_EMBEDDNG_VECTOR_SIZE)

       Dimensions of grad_h is (1, SKIP_GRAM_EMBEDDNG_VECTOR_SIZE)
     */
    Collective<E> grad_h;
    try
    {                
        grad_h = Numcy::dot(grad_u, W2_T);
    }
    catch (ala_exception& e)
    {
        throw ala_exception(cc_tokenizer::String<char>("bacward() -> ") + cc_tokenizer::String<char>(e.what()));
    }

    /*
        Dimensions of grad_W1 is (len(vocab) without redundency, SKIP_GRAM_EMBEDDNG_VECTOR_SIZE)
     */
    Collective<E> grad_W1;
    try
    {
        grad_W1 = Numcy::zeros(DIMENSIONS{SKIP_GRAM_EMBEDDNG_VECTOR_SIZE, vocab.numberOfUniqueTokens(), NULL, NULL});
    }
    catch(ala_exception& e)
    {
        throw ala_exception(cc_tokenizer::String<char>("bacward() -> ") + cc_tokenizer::String<char>(e.what())); 
    }

    /*
        Dimensions of grad_h is (1, SKIP_GRAM_EMBEDDNG_VECTOR_SIZE)
        Dimensions of grad_W1 is (len(vocab) without redundency, SKIP_GRAM_EMBEDDNG_VECTOR_SIZE)
     */
    for (cc_tokenizer::string_character_traits<char>::size_type i = 0; i < grad_W1.getShape().getNumberOfColumns(); i++)
    {
        grad_W1[(pair->getCenterWord() - INDEX_ORIGINATES_AT_VALUE)*SKIP_GRAM_EMBEDDNG_VECTOR_SIZE + i] += grad_h[i];
    }

    /*
        Dimensions of grad_W1 is (len(vocab) without redundency, SKIP_GRAM_EMBEDDNG_VECTOR_SIZE)
        Dimensions of grad_W2 is (len(vocab) without redundency, len(vocab) without redundency)
     */    
    return backward_propogation<E>{grad_W1, grad_W2};
}

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
    @t, data type. Used as argument to templated types and functions
    @verbose, when true puts more text on screen to help debug code    
 */
#define SKIP_GRAM_TRAINING_LOOP(epoch, W1, W2, el, vocab, pairs, lr, rs, t, verbose)\
{\
    for (cc_tokenizer::string_character_traits<char>::size_type i = 1; i <= epoch; i++)\
    {\
        el = 0;\
        if (verbose)\
        {\
            std::cout<< "Epoch# " << i << " of " << epoch << " epochs." << std::endl;\
        }\
        while (pairs.go_to_next_word_pair() != cc_tokenizer::string_character_traits<char>::eof())\
        {\
            /* We've a pair, a pair is LEFT_CONTEXT_WORD/S CENTER_WORD and RIGHT_CONTEXT_WORD/S */\
            WORDPAIRS_PTR pair = pairs.get_current_word_pair();\
            forward_propogation<t> fp;\
            backward_propogation<t> bp;\
            try\
            {\
                fp = forward<t>(W1, W2, vocab, pair);\
                bp = backward<t>(W1, W2, vocab, fp, pair);\
                /*std::cout<< "bp.grad_weights_input_to_hidden COLUMNS -> " << bp.grad_weights_input_to_hidden.getShape().getNumberOfColumns() << ", ROWS -> " << bp.grad_weights_input_to_hidden.getShape().getDimensionsOfArray().getNumberOfInnerArrays() << std::endl;*/\
                /*std::cout<< "bp.grad_weights_hidden_to_output COLUMNS -> " << bp.grad_weights_hidden_to_output.getShape().getNumberOfColumns() << ", ROWS -> " << bp.grad_weights_hidden_to_output.getShape().getDimensionsOfArray().getNumberOfInnerArrays() << std::endl;*/\
                /*std::cout<< "W2 COLUMNS -> " << W2.getShape().getNumberOfColumns() << ", ROWS -> " << W2.getShape().getDimensionsOfArray().getNumberOfInnerArrays() << std::endl;*/\
            }\
            catch (ala_exception& e)\
            {\
                std::cout<< "SKIP_GRAM_TRAINIG_LOOP -> " << e.what() << std::endl;\
            }\
            Collective<t> W2_reshaped;\
            try\
            {\
                W2_reshaped = Numcy::reshape(W2, bp.grad_weights_hidden_to_output);\
            }\
            catch(ala_exception& e)\
            {\
                std::cout<< "SKIP_GRAM_TRAINIG_LOOP -> " << e.what() << std::endl;\
            }\
            /*std::cout<< "W2_reshaped COLUMNS -> " << W2_reshaped.getShape().getNumberOfColumns() << ", ROWS -> " << W2_reshaped.getShape().getDimensionsOfArray().getNumberOfInnerArrays() << std::endl;*/\
            W1 -= (bp.grad_weights_input_to_hidden * lr);\
            W2_reshaped -= (bp.grad_weights_hidden_to_output * lr);\
            for (cc_tokenizer::string_character_traits<char>::size_type i = 0; i < W2.getShape().getDimensionsOfArray().getNumberOfInnerArrays(); i++)\
            {\
                for (cc_tokenizer::string_character_traits<char>::size_type j = 0; j < W2.getShape().getNumberOfColumns(); j++)\
                {\
                    W2[i*W2.getShape().getNumberOfColumns() + j] = W2_reshaped[i*W2_reshaped.getShape().getNumberOfColumns() + j];\
                }\
            }\
            el = el + (-1*log(fp.predicted_probabilities[pair->getCenterWord() - INDEX_ORIGINATES_AT_VALUE]));\
        }\
        std::cout<< "el = " << el/pairs.get_number_of_word_pairs() << std::endl;\
    }\
}\

#endif
