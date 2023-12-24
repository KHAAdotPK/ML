/*
    src/main.cpp
    Q@khaa.pk
 */

/*
       THE GOAL DURING TRAINING IS FOR YOUR MODEL TO LEARN TO PREDICT THE TARGET SEQUENCE GIVEN THE INPUT SEQUENCE
    ------------------------------------------------------------------------------------------------------------------   
    The model's objective is to generate target sequences that closely match the true target sequences in the dataset.
 */

#include "main.hh"

int main(int argc, char *argv[])
{
    ARG arg_bs, arg_bs_line, arg_bs_para, arg_corpus, arg_dmodel, arg_epoch, arg_help, arg_verbose;

    cc_tokenizer::csv_parser<cc_tokenizer::String<char>, char> argsv_parser(cc_tokenizer::String<char>(COMMAND));
        
    if (argc < 2)
    {        
        HELP(argsv_parser, arg_help, "help");                
        HELP_DUMP(argsv_parser, arg_help); 

        return 0;                    
    }    
    
    FIND_ARG(argv, argc, argsv_parser, "?", arg_help);
    if (arg_help.i)
    {
        HELP(argsv_parser, arg_help, ALL);
        HELP_DUMP(argsv_parser, arg_help);

        return 0;
    }

    FIND_ARG(argv, argc, argsv_parser, "verbose", arg_verbose);

    FIND_ARG(argv, argc, argsv_parser, "bs", arg_bs);
    FIND_ARG_BLOCK(argv, argc, argsv_parser, arg_bs);
    FIND_ARG(argv, argc, argsv_parser, "bs_line", arg_bs_line);
    FIND_ARG(argv, argc, argsv_parser, "bs_paragraph", arg_bs_para);
    
    FIND_ARG(argv, argc, argsv_parser, "corpus", arg_corpus);
    FIND_ARG_BLOCK(argv, argc, argsv_parser, arg_corpus);
    FIND_ARG(argv, argc, argsv_parser, "--dmodel", arg_dmodel);
    FIND_ARG_BLOCK(argv, argc, argsv_parser, arg_dmodel);
    FIND_ARG(argv, argc, argsv_parser, "epoch", arg_epoch);
    FIND_ARG_BLOCK(argv, argc, argsv_parser, arg_epoch);    
    
    /* cc_tokenizer::String<char> data; */
    cc_tokenizer::String<char> input_sequence_data;
    cc_tokenizer::String<char> target_sequence_data;
    
    try 
    {
        if (arg_corpus.i && arg_corpus.argc)
        {
            ARG arg_input, arg_target;

            cc_tokenizer::csv_parser<cc_tokenizer::String<char>, char> argsv_parser(cc_tokenizer::String<char>(CORPUS_COMMAND));

            FIND_ARG((argv + arg_corpus.i), (arg_corpus.argc + 1), argsv_parser, "input", arg_input);
            FIND_ARG_BLOCK((argv + arg_corpus.i), arg_corpus.argc + 1, argsv_parser, arg_input);
            FIND_ARG((argv + arg_corpus.i) , (arg_corpus.argc + 1), argsv_parser, "target", arg_target);
            FIND_ARG_BLOCK((argv + arg_corpus.i), (arg_corpus.argc + 1), argsv_parser, arg_target);

            if (arg_input.argc)
            {
                input_sequence_data = cc_tokenizer::cooked_read<char>(argv[arg_corpus.i + arg_input.i + 1]);
            }

            if (arg_target.argc)
            {
                target_sequence_data = cc_tokenizer::cooked_read<char>(argv[arg_corpus.i + arg_target.i + 1]);
            }
        }
    }
    catch (ala_exception &e)
    {
        std::cerr <<"main(): "<<e.what()<<std::endl;
        return 0;  
    }
    
    /*
    try
    {
        if (arg_corpus.i && arg_corpus.argc)
        {                         
            data = cc_tokenizer::cooked_read<char>(argv[arg_corpus.i + 1]);            
        }
        else
        {
            data = cc_tokenizer::cooked_read<char>(DEFAULT_CORPUS_FOR_CODER_ENCODER_MODEL_USING_TRANSFORMERS_VOCAB_FILE);
        }
    }
    catch (ala_exception &e)
    {
        std::cerr <<"main(): "<<e.what()<<std::endl;

        return 0;
    }
     */

    cc_tokenizer::allocator<char> alloc_obj;

    /*
    cc_tokenizer::csv_parser<cc_tokenizer::String<char>, char> csv_parser(data); 
    class Corpus vocab(csv_parser);
     */

    cc_tokenizer::csv_parser<cc_tokenizer::String<char>, char> input_csv_parser(input_sequence_data); 
    cc_tokenizer::csv_parser<cc_tokenizer::String<char>, char> target_csv_parser(target_sequence_data); 

    class Corpus input_sequence_vocab;
    class Corpus target_sequence_vocab;

    // instance of parser and the size of corpus in number of lines
    try
    {       
        input_sequence_vocab = Corpus(input_csv_parser, 13);
        target_sequence_vocab = Corpus(target_csv_parser, 13);
    }
    catch (ala_exception& e)
    {
        std::cerr << e.what() << std::endl;                
        return 0;
    }

    /*
        std::cout<<"-->> "<<input_sequence_vocab.len(false)<<std::endl;
        std::cout<<"-->> "<<target_sequence_vocab.len(false)<<std::endl;
     */

    /*
    for (cc_tokenizer::string_character_traits<char>::size_type i = 0; i < target_sequence_vocab.len(false); i++)
    {
        std::cout<<target_sequence_vocab(i, false).c_str()<<" - "<<target_sequence_vocab(target_sequence_vocab(i, false))->index<<std::endl;
    }
     */
    
    /*
        d_model
        ---------
        In the original "Attention Is All You Need" paper that introduced the Transformer architecture, the hyperparameter is referred to as "d_model" (short for "dimension of the model").
        Commonly used values for d_model range from 128 to 1024, with 256 being a frequently chosen value.
        Each word is embedded into a vector of size "dimensionsOfTheModelHyperparameter". This embedding dimension determines how much information is captured in each vector representation of a word.
        Higher dimensions can potentially allow for more expressive representations, but they also increase the computational requirements of the model.
        Smaller dimensions might lead to more compact models but might struggle to capture complex patterns in the data.
     */
    cc_tokenizer::string_character_traits<char>::size_type dimensionsOfTheModelHyperparameter;
    if (arg_dmodel.argc)
    {
        dimensionsOfTheModelHyperparameter = std::atoi(argv[arg_dmodel.i + 1]);   
    }
    else
    {
        dimensionsOfTheModelHyperparameter = DEFAULT_DIMENTIONS_OF_THE_TRANSFORMER_MODEL_HYPERPARAMETER;
    }
    
    class Numcy nc;

    /*
        ------------------------------------
        | TRANSFORMERS AND PARALLELIZATION |
        ------------------------------------
        To solve the problem of parallelization, Transformers try to solve the problem by using encoders and decoders together with attention model.
     */

    /* ****************************************************************************************************************************************************** */
    /*   
        INPUTSEQUENCE, POSITION AND POSITIONENCODING, DIVISIONTERM PROVIDE EACH WORD THEIR OWN CORRESPONDING HIDDEN STATE THAT IS PASSED ALL THE WAY TO THE 
        DECODING STAGE.
        Positional encoding is crucial in the Transformer model because it doesn't have an inherent sense of order like recurrent or convolutional neural
        networks.
        
         encoder in an encoder-decoder model
        -------------------------------------
        The INPUTSEQUENCE, POSITION, and POSITIONENCODING, DIVISIONTERM Collectively form the foundation for the encoder's understanding of the input 
        sequence's content
        and structure. They enable the Transformer model to capture relationships between tokens and process sequences effectively, which is crucial for 
        tasks like machine translation, summarization, and more, where preserving the sequence order is important for generating meaningful output sequences.
     */
    /* ****************************************************************************************************************************************************** */     
    /*
        This is a crucial step in ensuring that the model can leverage both the token information and positional information.
     */
    struct Collective<float> encoderInput;
    struct Collective<float> decoderInput;

    /*
        The position Collective represents the positions of the tokens in the input sequence.
        In the context of the Transformer model, these positions are used to create positional encodings.
        Positional encodings are created by multiplying two vectors, the other vector is the Collective named divisionTerm("div_term").
     */
    //float *position = NULL;
    struct Collective<float> position;
    
    /*
        When training a transformer model, a typical dataset consists of pairs of input sequences and corresponding target sequences.
        For instance, an input sequence could be a sentence, and its corresponding target sequence would indicate whether the sentence is negative or positive.

        In this context:
        - The input sequence is the sequence of tokens provided to the model for processing during training.
        - During the training phase, the input sequence serves as the training data.

        Similarly, when interacting with a trained model like ChatGPT:
        - The input sequence refers to the question or query provided to the model.

        Here, we declare a structure, "Collective," parameterized by the data type (e.g., float) that represents the input sequence.

        Note: It's important to ensure that the data type specified (e.g., float) aligns with the requirements of your specific model and training data.
     */    
    struct Collective<float> inputSequence;     
    /*
        Positional Encoding...
        Positional encodings are added to the input embeddings to provide information about the order of tokens in the sequence.

        These encodings are often based on mathematical functions to capture the relative positions of tokens in a continuous vector space.
        The specific formula used for generating positional encodings can vary,
        but the basic idea is to encode position information that can be added to the token embeddings before feeding them into the model.
     */   
    struct Collective<float> positionEncoding;

    /*
        Random target sequence, the target sequence is a separate set of token IDs that the model learns to generate based on the input sequence.
        - The target sequences are what you want the model to learn to predict or generate.
     */
    struct Collective<float> targetSequence;

    /*
        div_term
        ----------
        It's a part of the positional encoding calculation used to provide the model with information about the positions of tokens in the input sequence.
        The purpose of "div_term" is to scale the frequencies of the sinusoidal functions. It does that by working as divisor when computing the sine and 
        cosine values for positional encodings.

        NOTE:- To understand sinusoidal concept...
        Read about macro "SCALING_FACTOR_CONSTANT"(lib/ML/NLP/Word2Vec/transformers/CoderEncoder/header.hh) 
     */
    struct Collective<float> divisionTerm;
                        
    //class Encoder encoder(DEFAULT_DIMENTIONS_OF_THE_TRANSFORMER_MODEL_HYPERPARAMETER, DEFAULT_NUMBER_OF_LAYERS_FOR_ENCODER_HYPERPARAMETER, DEFAULT_NUMBER_OF_ATTENTION_HEADS_HYPERPARAMETER, DEFAULT_DROP_OUT_RATE_HYPERPARAMETER);

    try {
        if (arg_epoch.argc)
        {
            if (arg_bs_line.i)
            {
                TRAINING_LOOP_LINE_BATCH_SIZE(input_csv_parser, target_csv_parser, encoderInput, decoderInput, dimensionsOfTheModelHyperparameter, std::atoi(argv[arg_epoch.i + 1]), input_sequence_vocab, target_sequence_vocab, position, divisionTerm, positionEncoding, inputSequence, targetSequence, float, arg_verbose.i ? MAKE_IT_VERBOSE_MAIN_HH : !MAKE_IT_VERBOSE_MAIN_HH);
            }
            else if (arg_bs_para.i)
            {

            }
            else if (arg_bs.argc)
            {

            }

            //TRAINING_LOOP(encoderInput, decoderInput, dimensionsOfTheModelHyperparameter, std::atoi(argv[arg_epoch.i + 1]), input_sequence_vocab, target_sequence_vocab, position, divisionTerm, positionEncoding, inputSequence, targetSequence, float, arg_verbose.i ? MAKE_IT_VERBOSE_MAIN_HH : !MAKE_IT_VERBOSE_MAIN_HH);
        }
        else
        {
            if (arg_bs_line.i)
            {
                TRAINING_LOOP_LINE_BATCH_SIZE(input_csv_parser, target_csv_parser, encoderInput, decoderInput, dimensionsOfTheModelHyperparameter, DEFAULT_EPOCH_HYPERPARAMETER, input_sequence_vocab, target_sequence_vocab, position, divisionTerm, positionEncoding, inputSequence, targetSequence, float, arg_verbose.i ? MAKE_IT_VERBOSE_MAIN_HH : !MAKE_IT_VERBOSE_MAIN_HH);
            }
            else if (arg_bs_para.i)
            {

            }
            else if (arg_bs.argc)
            {

            }
            //TRAINING_LOOP(encoderInput, decoderInput, dimensionsOfTheModelHyperparameter, DEFAULT_EPOCH_HYPERPARAMETER, input_sequence_vocab, target_sequence_vocab, position, divisionTerm, positionEncoding, inputSequence, targetSequence, float, arg_verbose.i ? MAKE_IT_VERBOSE_MAIN_HH : !MAKE_IT_VERBOSE_MAIN_HH);
        }
    }
    catch (ala_exception& e)
    {       
       std::cerr << e.what() << '\n'; 
    }

    /*
    try
    {
        DIMENSIONS foo = {0, 0, NULL, NULL};
        Numcy::ones(foo);
    }
    catch(ala_exception& e)
    {
        std::cerr << e.what() << '\n';
    }
     */

    try
    {
        /*DIMENSIONS dim3 = {10, 3, NULL, NULL};
        DIMENSIONS dim2 = {0, 10, &dim3, NULL};
        dim3.prev = &dim2;
        DIMENSIONS dim1 = {0, 78, &dim2, NULL};
        dim2.prev = &dim1;
        DIMENSIONS dim = {0, 9, &dim1, NULL};
        dim1.prev = &dim; */

        /*
        DIMENSIONSOFARRAY shape = dim.getDimensionsOfArray();

        std::cout<<"shape.n"<<shape.n<<std::endl;

        for (int i = 0; i < shape.n; i++)
        {
            std::cout<<shape.ptr[i]<<" - ";
        }

        std::cout<<std::endl;
        */

        /*
        DIMENSIONS foo = {1, 10, NULL, NULL};

        Numcy::triu<float>(NULL, foo);*/
    }
    catch(ala_exception& e)
    {
        std::cerr << e.what() << '\n';
    }
                
    /*
        TODO, write a note on why these uglies are here...
     */
    /*alloc_obj.deallocate(reinterpret_cast<char*>(divisionTerm.ptr));
    alloc_obj.deallocate(reinterpret_cast<char*>(encoderInput.ptr));
    alloc_obj.deallocate(reinterpret_cast<char*>(decoderInput.ptr));
    alloc_obj.deallocate(reinterpret_cast<char*>(inputSequence.ptr));
    alloc_obj.deallocate(reinterpret_cast<char*>(position.ptr));
    alloc_obj.deallocate(reinterpret_cast<char*>(positionEncoding.ptr));
    alloc_obj.deallocate(reinterpret_cast<char*>(targetSequence.ptr));*/
    
    divisionTerm.decrementReferenceCount();
    encoderInput.decrementReferenceCount();
    decoderInput.decrementReferenceCount();
    inputSequence.decrementReferenceCount();
    position.decrementReferenceCount();
    positionEncoding.decrementReferenceCount();
    targetSequence.decrementReferenceCount();
    
    input_sequence_vocab.decrementReferenceCount();
    target_sequence_vocab.decrementReferenceCount();
    
    /*divisionTerm.ptr = NULL;
    encoderInput.ptr = NULL;
    inputSequence.ptr = NULL;
    decoderInput.ptr = NULL;
    position.ptr = NULL;
    positionEncoding.ptr = NULL;
    targetSequence.ptr = NULL;*/
                                
    return 0;
}