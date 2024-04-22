/*
    src/main.cpp
    Q@khaa.pk
 */

#include "main.hh"

int main(int argc, char* argv[])
{
    ARG arg_corpus, arg_epoch, arg_help, arg_learning_rate, arg_verbose, arg_window_size;

    cc_tokenizer::allocator<char> alloc_obj;
    cc_tokenizer::csv_parser<cc_tokenizer::String<char>, char> argsv_parser(cc_tokenizer::String<char>(COMMAND));
    cc_tokenizer::String<char> data;

    class numc numc_obj;
    double default_learning_rate = CBOW_DEFAULT_LEARNING_RATE;
    struct context_target_list cth = {0, NULL}; // context target head
    unsigned int default_window_size = CBOW_DEFAULT_WINDOW_SIZE;
    unsigned long default_epoch = CBOW_DEFAULT_EPOCH;

    FIND_ARG(argv, argc, argsv_parser, "?", arg_help);
    if (arg_help.i)
    {
        HELP(argsv_parser, arg_help, ALL);        
        HELP_DUMP(argsv_parser, arg_help);

        return 0;
    }

    FIND_ARG(argv, argc, argsv_parser, "verbose", arg_verbose);
    
    /*
        In the context of training a machine learning model, an epoch is defined as a complete pass over the entire training dataset during training.
        One epoch is completed when the model has made one update to the weights based on each training sample in the dataset.
        In other words, during one epoch, the model has seen every example in the dataset once and has made one update to the model parameters for each example.

        The number of epochs to train for is typically set as a hyperparameter, and it depends on the specific problem and the size of the dataset. 
        One common approach is to monitor the performance of the model on a validation set during training, and stop training when the performance 
        on the validation set starts to degrade.
     */    
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

    FIND_ARG(argv, argc, argsv_parser, "--lr", arg_learning_rate);
    if (arg_learning_rate.i)
    {
        FIND_ARG_BLOCK(argv, argc, argsv_parser, arg_learning_rate);

        if (arg_learning_rate.argc)
        {
            default_learning_rate = atoi(argv[arg_learning_rate.j]);
        }
        else
        {
            ARG arg_learning_rate_help;
            HELP(argsv_parser, arg_learning_rate_help, "lr");                
            HELP_DUMP(argsv_parser, arg_learning_rate_help);

            return 0;
        }                
    }

    FIND_ARG(argv, argc, argsv_parser, "ws", arg_window_size);
    if (arg_window_size.i)
    {
        FIND_ARG_BLOCK(argv, argc, argsv_parser, arg_window_size);

        if (arg_window_size.argc)
        {
            default_window_size = atoi(argv[arg_window_size.j]);
        }
        else
        {
            ARG arg_window_size_help;
            HELP(argsv_parser, arg_window_size_help, "--ws");                
            HELP_DUMP(argsv_parser, arg_window_size_help);

            return 0;
        }                
    }

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

    /* FINISHED COMMAND LINE ARGUMENT COLLECTION  */
    if (arg_verbose.i)
    {
        std::cout<<"Epoch is "<<default_epoch<<", Learning rate is "<<default_learning_rate<<", Window size is "<<default_window_size<<std::endl;
    }

    /*
        Create vocabulary, vocabulary is corpus minus any redundency        
     */
    cc_tokenizer::csv_parser<cc_tokenizer::String<char>, char> data_parser(data);
    class corpus vocab(data_parser);

    if (arg_verbose.i)
    {
        CORPUS_TO_CONTEXTS_AND_TARTGETS(data_parser, vocab, default_window_size, cth, true);
    }
    else
    {
        CORPUS_TO_CONTEXTS_AND_TARTGETS(data_parser, vocab, default_window_size, cth, false);
    }

    //cth.traverse();

    std::cout<<vocab.get_number_of_context_lines_in_vocabulary()<<std::endl;
    std::cout<<vocab.get_number_of_target_lines_in_vocabulary()<<std::endl;
    std::cout<<vocab.get_number_of_context_columns_in_vocabulary()<<std::endl;

    /*
        The size of the weight matrices depends on the number of input and output neurons in each layer
        Our neural network is simple, it only has one hidden layer. 
        Together, W1 andd W2 weight matrices define the neural network architecture and allow it to learn the word embeddings through training.

        In a neural network with two hidden layers, we would need to define three sets of weights: W1 connecting the input layer to the first hidden layer, 
        W2 connecting the first hidden layer to the second hidden layer, and W3 connecting the second hidden layer to the output layer.
        Each weight matrix would have a different shape, depending on the number of neurons in the layers it connects.

        The number of columns in W1 is equal to the number of rows in W2. This is necessary for the dot product operation.        
     */

    //std::cout<<"----> "<<vocab.get_number_of_context_lines_in_vocabulary()<<std::endl;
    
    /* Parameter size, it is just the number of weights in the matrix */
    double (*W1)[CBOW_HIDDEN_LAYER_SIZE_IN_NUMBER_OF_NEURONS] = reinterpret_cast<double (*)[CBOW_HIDDEN_LAYER_SIZE_IN_NUMBER_OF_NEURONS]>(numc_obj.RANDN(vocab.len(), CBOW_HIDDEN_LAYER_SIZE_IN_NUMBER_OF_NEURONS));    
    double* W2 = reinterpret_cast<double*>(numc_obj.RANDN(CBOW_HIDDEN_LAYER_SIZE_IN_NUMBER_OF_NEURONS, vocab.len()));
   
    //std::cout<<"------> "<<default_epoch<<std::endl;
    //CBOW_TRAINING_LOOP(default_epoch, cth, W1, W2, vocab);
    
    /* Fold back every thing */
    DEALLOCATE_CORPUS_TO_CONTEXT_AND_TARGETS(cth);


    //std::cout<<vocab.get_number_of_context_lines_in_vocabulary()<<std::endl;
    //std::cout<<vocab.get_number_of_target_lines_in_vocabulary()<<std::endl;

    //std::cout<<"----> "<<vocab.get_number_of_context_lines_in_vocabulary()<<std::endl;

    alloc_obj.deallocate(reinterpret_cast<cc_tokenizer::allocator<char>::pointer>(W1));
    alloc_obj.deallocate(reinterpret_cast<cc_tokenizer::allocator<char>::pointer>(W2));
        
    return 0;
}