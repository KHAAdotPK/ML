/*
    Skip-Gram/src/main.cpp
    Written by, Q@khaa.pk
 */

#include "main.hh"

int main(int argc, char* argv[])
{
    ARG arg_corpus, arg_epoch, arg_help, arg_verbose;
    
    cc_tokenizer::allocator<char> alloc_obj;
    cc_tokenizer::csv_parser<cc_tokenizer::String<char>, char> argsv_parser(cc_tokenizer::String<char>(COMMAND));
    cc_tokenizer::String<char> data;
        
    FIND_ARG(argv, argc, argsv_parser, "?", arg_help);
    if (arg_help.i)
    {
        HELP(argsv_parser, arg_help, ALL);
        HELP_DUMP(argsv_parser, arg_help);

        return 0;
    }

    if (argc < 2)
    {        
        HELP(argsv_parser, arg_help, "help");                
        HELP_DUMP(argsv_parser, arg_help);                     
    }

    FIND_ARG(argv, argc, argsv_parser, "verbose", arg_verbose);

    FIND_ARG(argv, argc, argsv_parser, "corpus", arg_corpus);
    if (arg_corpus.i)
    {
        FIND_ARG_BLOCK(argv, argc, argsv_parser, arg_corpus);
        if (arg_corpus.argc)
        {            
            try 
            {
                data = cc_tokenizer::cooked_read<char>(argv[arg_corpus.i + 1]);
                if (arg_verbose.i)
                {
                    std::cout<<"Corpus: "<<argv[arg_corpus.i + 1]<<std::endl;
                }
            }
            catch (ala_exception e)
            {
                std::cout<<e.what()<<std::endl;
                return -1;
            }            
        }
        else
        { 
            ARG arg_corpus_help;
            HELP(argsv_parser, arg_corpus_help, "--corpus");                
            HELP_DUMP(argsv_parser, arg_corpus_help);

            return 0; 
        }                
    }
    else
    {
        try
        {        
            data = cc_tokenizer::cooked_read<char>(SKIP_GRAM_DEFAULT_CORPUS_FILE);
            if (arg_verbose.i)
            {
                std::cout<<"Corpus: "<<SKIP_GRAM_DEFAULT_CORPUS_FILE<<std::endl;
            }
        }
        catch (ala_exception e)
        {
            std::cout<<e.what()<<std::endl;
            return -1;
        }
    }

    /*        
        In the context of training a machine learning model, an epoch is defined as a complete pass over the entire training dataset during training.
        One epoch is completed when the model has made one update to the weights based on each training sample in the dataset.
        In other words, during one epoch, the model has seen every example in the dataset once and has made one update to the model parameters for each example.

        The number of epochs to train for is typically set as a hyperparameter, and it depends on the specific problem and the size of the dataset. 
        One common approach is to monitor the performance of the model on a validation set during training, and stop training when the performance 
        on the validation set starts to degrade.
     */
    unsigned long default_epoch = SKIP_GRAM_DEFAULT_EPOCH;    
    FIND_ARG(argv, argc, argsv_parser, "epoch", arg_epoch);
    if (arg_epoch.i)
    {
        FIND_ARG_BLOCK(argv, argc, argsv_parser, arg_epoch);

        if (arg_epoch.argc)
        {
            default_epoch = atoi(argv[arg_epoch.j]);
        }
        else
        {
            ARG arg_epoch_help;
            HELP(argsv_parser, arg_epoch_help, "e");                
            HELP_DUMP(argsv_parser, arg_epoch_help);

            return 0;
        }                
    }
    
    cc_tokenizer::csv_parser<cc_tokenizer::String<char>, char> data_parser(data);

    /*
        Create vocabulary        
     */
    class Corpus vocab(data_parser);
    //class numc numc_obj;

    /*
        Regularization: Neural networks are notorious for their overfitting issues and they tend to memorize the data without learning the underlying patterns.
        There are different regularization techniques including L1 and L2 regularization, dropout, and early stopping.
        Fundamentally, they involve applying some mathematical function to prevent the model from over-learning from the data.
        L1 and L2 regularization: add absolute values or squares of weights to the loss function
        Dropout: Randomly set some fraction of outputs in the layer to zero during training (prevents single neuron overlearning)
     */
    double regularizationStrength = 0.1;
    
    /*
    std::cout<<vocab.len(false)<<std::endl;
     */

    /*
        Generate skip-gram pairs
     */    
    /*
    SKIP_GRAM_PAIRS pairs;
     */
    SKIPGRAMPAIRS pairs(vocab/*, arg_verbose.i ? true : false*/);
    //GENERATE_SKIP_GRAM_PAIRS(pairs, vocab, false);

    generateNegativeSamples(vocab, pairs, 300);

    generateNegativeSamples_new(vocab, pairs, 30);

    /*
    try 
    {
        Numcy::Random::randn(DIMENSIONS{0, 0, NULL, NULL});
    }
    catch (ala_exception& e)
    {
        std::cout<< e.what() << std::endl;

        return 0;
    }
     */

    double epoch_loss = 0;
    
    /*
        For the neural network itself, Skip-gram typically uses a simple architecture. 

        Each row in W1 represents the embedding vector for one specific center word in your vocabulary(so in W1 word redendency is not allowed).
        During training, the central word from a word pair is looked up in W1 to retrieve its embedding vector.
        The size of embedding vector is hyperparameter(SKIP_GRAM_EMBEDDING_VECTOR_SIZE). It could be between 100 to 300 per center word.

        Each row in W2 represents the weight vector for predicting a specific context word (considering both positive and negative samples).
        The embedding vector of the central word (from W1) is multiplied by W2 to get a score for each context word.

        Hence the skip-gram variant takes a target word and tries to predict the surrounding context words.

        Why Predict Context Words?
        1. By predicting context words based on the central word's embedding, Skip-gram learns to capture semantic relationships between words.
        2. Words that often appear together in similar contexts are likely to have similar embeddings.
     */
    /*
        * Skip-gram uses a shallow architecture with two weight matrices, W1 and W2.

        * W1: Embedding Matrix
          - Each row in W1 is a unique word's embedding vector, representing its semantic relationship with other words.
          - The size of this embedding vector (SKIP_GRAM_EMBEDDING_VECTOR_SIZE) is a hyperparameter, typically ranging from 100 to 300.

        * W2: Output Layer (weights for predicting context words)
          - Each row in W2 represents the weight vector for predicting a specific context word (considering both positive and negative samples).
          - The embedding vector of the central word (from W1) is multiplied by W2 to get a score for each context word.

        * By predicting surrounding context words based on the central word's embedding, Skip-gram learns to capture semantic relationships between words with similar contexts.
     */
    Collective<double> W1;
    Collective<double> W2;
    //W1 = Collective<double>{NULL, DIMENSIONS{SKIP_GRAM_EMBEDDNG_VECTOR_SIZE, vocab.numberOfUniqueTokens(), NULL, NULL}};
    //W2 = Collective<double>(NULL, DIMENSIONS{vocab.numberOfUniqueTokens(), SKIP_GRAM_EMBEDDNG_VECTOR_SIZE, NULL, NULL});
    try
    {            
        W1 = Numcy::Random::randn(DIMENSIONS{SKIP_GRAM_EMBEDDNG_VECTOR_SIZE, vocab.numberOfUniqueTokens(), NULL, NULL});        
        W2 = Numcy::Random::randn(DIMENSIONS{vocab.numberOfUniqueTokens(), SKIP_GRAM_EMBEDDNG_VECTOR_SIZE, NULL, NULL});
    }
    catch (ala_exception& e)
    {
        std::cout<< e.what() << std::endl;

        return 0;
    }

    SKIP_GRAM_TRAINING_LOOP(default_epoch, W1, W2, epoch_loss, vocab, pairs, SKIP_GRAM_DEFAULT_LEARNING_RATE, regularizationStrength, double, arg_verbose.i ? true : false);

    /*
        Start training loop
     */
    /*
    if (arg_verbose.i)
    {
        SKIP_GRAM_TRAINING_LOOP(W1, W2, pairs, numc_obj, vocab, default_epoch, SKIP_GRAM_DEFAULT_LEARNING_RATE, regularizationStrength, true);
    }
    else
    {
        SKIP_GRAM_TRAINING_LOOP(W1, W2, pairs, numc_obj, vocab, default_epoch, SKIP_GRAM_DEFAULT_LEARNING_RATE, regularizationStrength, false);
    }
     */

    /*
    for (int i = 0; i < pairs.len(); i++)
    {
        for (int j = 0; j < SKIP_GRAM_DEFAULT_PAIR_SIZE; j++)
        {
            std::cout<<pairs[i][j]<<" ";            
        }

        std::cout<<std::endl;
    }
     */

    /*
        Initialize weights
     */
    /*
        W1 , w2. The size of the weight matrices depends on the number of input and output neurons in each layer
        shape of W1 is (len(vocab), SKIP_GRAM_HIDDEN_SIZE), W2 has shape of (SKIP_GRAM_HIDDEN_SIZE, len(vocab))
        This neural network has only one hidden layer. This hidden layer has SKIP_GRAM_HIDDEN_SIZE many neurons
        W1, This matrix represents the weights connecting the input layer (the one-hot vectors representing the context words) to the hidden layer.
        W2, This matrix represents the weights connecting the hidden layer to the output layer (the predicted one-hot vector for the target/context word).
     */
    /*
    double (*W1)[SKIP_GRAM_HIDDEN_SIZE] = reinterpret_cast<double (*)[SKIP_GRAM_HIDDEN_SIZE]>(numc_obj.RANDN(vocab.len(), SKIP_GRAM_HIDDEN_SIZE));    
    double* W2 = numc_obj.RANDN(SKIP_GRAM_HIDDEN_SIZE, vocab.len());
     */

    /*
        Training loop
     */
    //SKIP_GRAM_TRAINING_LOOP(W1, W2, pairs, numc_obj, vocab, default_epoch, SKIP_GRAM_DEFAULT_LEARNING_RATE, false);

    /*
        Start training loop
     */
    /*
    if (arg_verbose.i)
    {
        SKIP_GRAM_TRAINING_LOOP(W1, W2, pairs, numc_obj, vocab, default_epoch, SKIP_GRAM_DEFAULT_LEARNING_RATE, regularizationStrength, true);
    }
    else
    {
        SKIP_GRAM_TRAINING_LOOP(W1, W2, pairs, numc_obj, vocab, default_epoch, SKIP_GRAM_DEFAULT_LEARNING_RATE, regularizationStrength, false);
    }
     */

    /* ************************************************************************************************************** */
    /*                                         WE HAVE TRAINED SKIP-GRAM MODEL                                        */
    /* ************************************************************************************************************** */
    
    //#define WORDS "semantic AI Skip-Gram contextual analyze transformative technology mimic human intelligence and perform tasks"
    //#define WORDS "Artificial Intelligence AI has emerged as a transformative technology with the potential to revolutionize numerous fields and industries"

    /*
    #define WORDS "brown fox jumps"

    data = cc_tokenizer::String<char>(WORDS);
    data_parser = cc_tokenizer::csv_parser<cc_tokenizer::String<char>, char>(data);
    cc_tokenizer::string_character_traits<char>::size_type j = 0;
    
    data_parser.go_to_next_line();
    //std::cout<<"------> "<<data_parser.get_total_number_of_tokens()<<std::endl;
    double (*word_vectors)[SKIP_GRAM_HIDDEN_SIZE] = reinterpret_cast<double (*)[SKIP_GRAM_HIDDEN_SIZE]>(alloc_obj.allocate(sizeof(double)*data_parser.get_total_number_of_tokens()*SKIP_GRAM_HIDDEN_SIZE));
    while (data_parser.go_to_next_token() != cc_tokenizer::string_character_traits<char>::eof())
    {

        //std::cout<<"--> "<<data_parser.get_current_token().c_str()<<std::endl;

        for (unsigned long i = 0; i < SKIP_GRAM_HIDDEN_SIZE; i++)
        {
            CORPUS_PTR word = vocab[data_parser.get_current_token()];
            
            if (word == NULL)
            {
                std::cout<<"NULL -> "<<data_parser.get_current_token().c_str()<<std::endl;
                // TODO, deallocate and stop the program
            }
            
            word_vectors[j][i] = *(W1[vocab[data_parser.get_current_token()]->index - REPLIKA_PK_INDEX_ORIGINATES_AT] + i);
        }

        j++;
    }

    data_parser.reset(TOKENS);

    for (cc_tokenizer::string_character_traits<char>::size_type i = 0; i < data_parser.get_total_number_of_tokens(); i++)
    {
        for (j = i + 1; j < data_parser.get_total_number_of_tokens(); j++)
        {
            double sim = numc_obj.cosine_distance(word_vectors[i], word_vectors[j], SKIP_GRAM_HIDDEN_SIZE);

            printf("Similarity between %s and %s : %f", data_parser.get_token_by_number(i + 1).c_str(), data_parser.get_token_by_number(j + 1).c_str(), sim);
            std::cout<<std::endl;

            //CORPUS_PTR word = vocab[data_parser.get_token_by_number(i + 1)];
            //corpus::corpus_index_type* ptr = pairs[word->index];
            //std::cout<<"---------------------------> "<<vocab[ptr[SKIP_GRAM_PAIR_CONTEXT_INDEX]].c_str()<<std::endl;
            //std::cout<<"---------------------------> "<<vocab[ptr[SKIP_GRAM_PAIR_TARGET_INDEX]].c_str()<<std::endl;
        }
    }
            
    alloc_obj.deallocate(reinterpret_cast<char*>(W1));
    alloc_obj.deallocate(reinterpret_cast<char*>(W2));
    alloc_obj.deallocate(reinterpret_cast<char*>(word_vectors));
     */
                        
    return 0;
}