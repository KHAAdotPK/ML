/*
    lib/NLP/transformers/encoder-decoder/model.hh
    Q@khaa.pk
 */

#include "./attention.hh"
#include "./encoderlayer.hh"
#include "./encoder.hh"

#ifndef NLP_ENCODER_DECODER_TRANSFORMER_MODEL_HH
#define NLP_ENCODER_DECODER_TRANSFORMER_MODEL_HH

/* 
    ----------------------------------  
    | UTILIZING INITIALIZER LISTS... | 
    ----------------------------------
    Here's an illustration of how initializer lists can enhance code readability and conciseness:
    Consider initializing a 'Collective<float>' object 'p' with the result of 'Numcy::arange' and a specified shape:
    Using an initializer list:
    Collective<float> p = {Numcy::arange((float)0.0, (float)is.shape[1], (float)1.0, {1, is.shape[1], NULL, NULL}), {1, is.shape[1], NULL, NULL}};
    Instead of the above, we could opt for a constructor approach:
    Collective<float> p = Collective<float>(Numcy::arange((float)0.0, (float)is.shape[1], (float)1.0, {1, is.shape[1], NULL, NULL}), {1, is.shape[1], NULL, NULL});

    It's worth noting that when dealing with private properties within the 'Collective' composite, using the specific overloaded constructor becomes necessary, while otherwise, it may not be required.
 */


/*
    @p, position an instance of Collective composite
    @is, input sequence
    @dt, division term
    @dm, dimensions of the model(d_model)
    @pe, position encoding
    @t, type
 */
#define BUILD_POSITION_ENCODING_FOR_LINE_BATCH_SIZE(p, is, dt, dm, pe, t) {\
p = Collective<t>{Numcy::arange<t, t>((t)0.0, (t)is.shape[NUMCY_DIMENSIONS_SHAPE_COLUMNS], (t)1.0, {1, is.shape[NUMCY_DIMENSIONS_SHAPE_COLUMNS], NULL, NULL}), {1, is.shape[NUMCY_DIMENSIONS_SHAPE_COLUMNS], NULL, NULL}};\
dt = Collective<t>{Numcy::exp<t>(Numcy::arange<t, t>((t)0.0, (t)dm, (t)2.0, {dm/2, 1, NULL, NULL}), dm/2), {dm/2, 1, NULL, NULL}};\
dt = dt * SCALING_FACTOR(SCALING_FACTOR_CONSTANT, dm);\
/*pe = Collective<t>{Numcy::zeros<t>({dm, is.shape[NUMCY_DIMENSIONS_SHAPE_COLUMNS], NULL, NULL}), {dm, is.shape[NUMCY_DIMENSIONS_SHAPE_COLUMNS], NULL, NULL}};*/\
pe = Numcy::zeros<t>({dm, is.shape[NUMCY_DIMENSIONS_SHAPE_COLUMNS], NULL, NULL});\
FILL_EVEN_INDICES_OF_POSITION_ENCODING(pe, Numcy::sin<t>(p * dt));\
FILL_ODD_INDICES_OF_POSITION_ENCODING(pe, Numcy::cos<t>(p * dt));\
}\

/*
    @v, vocab
    is, input sequence
    t, type
    for (cc_tokenizer::string_character_traits<char>::size_type i = 0; i < target_sequence_vocab.len(false); i++)
    {
        std::cout<<target_sequence_vocab(i, false).c_str()<<" - "<<target_sequence_vocab(target_sequence_vocab(i, false))->index<<std::endl;
    }
 */
#define BUILD_INPUT_SEQUENCE(v, is, t) {\
                                            cc_tokenizer::allocator<char> alloc_obj;\
                                            t* ptr = reinterpret_cast<t*>(alloc_obj.allocate(v.len(false)*sizeof(t)));\
                                            for (cc_tokenizer::string_character_traits<char>::size_type i = 0; i < v.len(false); i++)\
                                            {\
                                                /*std::cout<<v(i, false).c_str()<<" - "<<v(v(i, false))->index<<std::endl;*/\
                                                ptr[i] = v(v(i, false))->index;\
                                            }\
                                            is = {ptr, {v.len(false), 1, NULL, NULL}};\
                                       }\

/*
 * ---------------------------------------------------------
 * | BUILD INPUT SEQUENCE WHEN BATCH SIZE IS SET TO A LINE |
 * ---------------------------------------------------------    
 */
/* Temporary Solution to Address Compile-Time Error ("narrow conversion") */
/*
 * If you are confident that the 'int_type(int)' value can be safely accommodated within 'size_t' without loss of data,
 * you can use a 'static_cast' to perform the conversion. However, exercise caution when using this approach.
 */
/* TODO: Eliminate the Need for the Following "Narrow Conversion" */
/*
 * The return type of 'get_total_number_of_tokens()' is 'cc_tokenizer::string_character_traits<char>::int_type',
 * whereas the type of 'DIMENSIONS::columns' is 'cc_tokenizer::string_character_traits<char>::size_type'.
 * Converting a signed integer to its unsigned equivalent is considered a "narrow conversion,"
 * which may lead to unexpected behavior. It is advisable to avoid such conversions whenever possible.

 * In future iterations of the codebase, consider revising the design of the parser and related entities to ensure
 * that values of similar semantics share consistent data types. This will enhance code safety and maintainability.
 */
/*
    @is, input sequence
    @v, vocabulary, it is input vocabulary
    @icp, input csv parser
    @t, type
 */
#define BUILD_INPUT_SEQUENCE_FOR_LINE_BATCH_SIZE(is, v, p, t) {\
t* ptr = cc_tokenizer::allocator<t>().allocate(p.get_total_number_of_tokens());\
for (cc_tokenizer::string_character_traits<char>::size_type i = 0; i < p.get_total_number_of_tokens(); i++)\
{\
    CORPUS_PTR ret = v(p.get_token_by_number(i + 1));\
    ptr[i] = ret->index;\
}\
\
/* TODO: Eliminate the Need for Narrow Conversion */\
/* The return type of 'get_total_number_of_tokens()' is 'cc_tokenizer::string_character_traits<char>::int_type', */\
/* while 'DIMENSIONS::columns' is 'cc_tokenizer::string_character_traits<char>::size_type'. */\
/* Converting a signed to unsigned is a narrow conversion; it's recommended to avoid such conversions. */\
/* In future iterations, enhance code consistency by ensuring similar semantics share consistent data types.*/\
is = Collective<t>(ptr, {static_cast<cc_tokenizer::string_character_traits<char>::size_type>(p.get_total_number_of_tokens()), 1, NULL, NULL});\
/* Same thing with initializer lists. */\
/* is = {ptr, {static_cast<cc_tokenizer::string_character_traits<char>::size_type>(p.get_total_number_of_tokens()), 1, NULL, NULL}}; */\
}\

/*
 * ----------------------------------------------------------
 * | BUILD TARGET SEQUENCE WHEN BATCH SIZE IS SET TO A LINE |
 * ----------------------------------------------------------    
 */
/* Temporary Solution to Address Compile-Time Error ("narrow conversion") */

/* If you are confident that the 'int_type' value can be safely accommodated within 'size_t' without loss of data,
 * you can use a 'static_cast' to perform the conversion. However, exercise caution when using this approach.

/* TODO: Eliminate the Need for the Following "Narrow Conversion" */

/* 
 * The return type of 'get_total_number_of_tokens()' is 'cc_tokenizer::string_character_traits<char>::int_type',
 * whereas the type of 'DIMENSIONS::columns' is 'cc_tokenizer::string_character_traits<char>::size_type'.
 * Converting a signed integer to its unsigned equivalent is considered a "narrow conversion,"
 * which may lead to unexpected behavior. It is advisable to avoid such conversions whenever possible.

 * In future iterations of the codebase, consider revising the design of the parser and related entities to ensure
 * that values of similar semantics share consistent data types. This will enhance code safety and maintainability.
 */
/*
    @ts, target sequence
    @v, vocabulary, it is target vocabulary
    @tcp, target csv parser
    @t, type
 */
#define BUILD_TARGET_SEQUENCE_FOR_LINE_BATCH_SIZE(ts, v, p, t) {\
t* ptr = reinterpret_cast<t*>(cc_tokenizer::allocator<char>().allocate((p.get_total_number_of_tokens())*sizeof(t)));\
for (cc_tokenizer::string_character_traits<char>::size_type i = 0; i < p.get_total_number_of_tokens(); i++)\
{\
    CORPUS_PTR ret = v(p.get_token_by_number(i + 1));\
    ptr[i] = ret->index;\
}\
\
/* TODO: Eliminate the Need for Narrow Conversion */\
/* The return type of 'get_total_number_of_tokens()' is 'cc_tokenizer::string_character_traits<char>::int_type', */\
/* while 'DIMENSIONS::columns' is 'cc_tokenizer::string_character_traits<char>::size_type'. */\
/* Converting a signed to unsigned is a narrow conversion; it's recommended to avoid such conversions. */\
/* In future iterations, enhance code consistency by ensuring similar semantics share consistent data types.*/\
ts = Collective<t>(ptr, {static_cast<cc_tokenizer::string_character_traits<char>::size_type>(p.get_total_number_of_tokens()), 1, NULL, NULL});\
}\
                                                              
/*
    @v, vocab, it is input vocabulary
    @is, input sequence    
    @si, start index
    @ei, end index
    @t, type
 */
#define BUILD_INPUT_SEQUENCE_NEW(v, is, si, ei, t) {\
                                                        t* ptr = reinterpret_cast<t*>(cc_tokenizer::allocator<char>().allocate((ei - si)*sizeof(t)));\
                                                        for (cc_tokenizer::string_character_traits<char>::size_type i = si; i < ei; i++)\
                                                        {\
                                                            ptr[i] = v(v(i, REPLIKA_PK_USE_WHOLE))->index;\
                                                        }\
                                                        is = {ptr, {ei - si, 1, NULL, NULL}};\
                                                   }\

/*
    @v, vocab
    @ts, target sequence
    @t, type
 */                                    
#define BUILD_TARGET_SEQUENCE(v, ts, t) {\
                                            cc_tokenizer::allocator<char> alloc_obj;\
                                            t* ptr = reinterpret_cast<t*>(alloc_obj.allocate(v.len(false)*sizeof(t)));\
                                            for (cc_tokenizer::string_character_traits<char>::size_type i = 0; i < v.len(false); i++)\
                                            {\
                                                /*std::cout<<v(i, false).c_str()<<" - "<<v(v(i, false))->index<<std::endl;*/\
                                                ptr[i] = v(v(i, false))->index;\
                                            }\
                                            ts = {ptr, {v.len(false), 1, NULL, NULL}};\
                                        }\

/*
    @v, vocab, it is target vocabulary
    @ts, target sequence
    @si, start index
    @ei, end index
    @t, type
 */
#define BUILD_TARGET_SEQUENCE_NEW(v, ts, si, ei, t) {\
                                                        t* ptr = reinterpret_cast<t*>(cc_tokenizer::allocator<char>().allocate((ei - si)*sizeof(t)));\
                                                        for (cc_tokenizer::string_character_traits<char>::size_type i = si; i < ei; i++)\
                                                        {\
                                                            ptr[i] = v(v(i, REPLIKA_PK_USE_WHOLE))->index;\
                                                        }\
                                                        ts = {ptr, {v.len(REPLIKA_PK_USE_WHOLE), 1, NULL, NULL}};\
                                                    }\

/*
    @dm, d_model, dymensions of the model
    @p, position
    @dt, division term
    @pe, position encoding
    @is, input sequence

    position = {Numcy::arange((float)0.0, (float)inputSequence.shape[1], (float)1.0, {1, inputSequence.shape[1], NULL, NULL}), {1, inputSequence.shape[1], NULL, NULL}}; 
    divisionTerm = {Numcy::exp(nc.arange((float)0.0, (float)dimensionsOfTheModelHyperparameter, (float)2.0, {dimensionsOfTheModelHyperparameter/2, 1, NULL, NULL}), dimensionsOfTheModelHyperparameter/2), {dimensionsOfTheModelHyperparameter/2, 1, NULL, NULL}};        
 */ 
#define BUILD_POSITION_ENCODING(dm, p, dt, pe, is) {\
                                                        p = {Numcy::arange((float)0.0, (float)inputSequence.shape[1], (float)1.0, {1, inputSequence.shape[1], NULL, NULL}), {1, is.shape[1], NULL, NULL}};\
                                                        dt = {Numcy::exp(nc.arange((float)0.0, (float)dm, (float)2.0, {dm/2, 1, NULL, NULL}), dm/2), {dm/2, 1, NULL, NULL}};\
                                                        MULTIPLY_ARRAY_BY_SCALAR(dt.ptr, SCALING_FACTOR(SCALING_FACTOR_CONSTANT, dm), dm/2, dt.ptr, float);\
                                                        pe = {Numcy::zeros<float>({dm, is.shape[NUMCY_DIMENSIONS_SHAPE_COLUMNS], NULL, NULL}), {dm, is.shape[NUMCY_DIMENSIONS_SHAPE_COLUMNS], NULL, NULL}};\
                                                        /*std::cout<<"--> pe.columns*pe.rows = "<<pe.shape.columns*pe.shape.rows<<std::endl;*/\
                                                        /*FILL_EVEN_INDCES_OF_POSITION_ENCODING(pe, Numcy::sin(p * dt));*/\
                                                        Collective<float> temp = Numcy::sin(p * dt);\
                                                        FILL_EVEN_INDICES_OF_POSITION_ENCODING(pe, temp);\
                                                        cc_tokenizer::allocator<char>().deallocate(reinterpret_cast<char*>(temp.ptr));\
                                                        temp = {NULL, {0, 0, NULL, NULL}};\
                                                        /*FILL_ODD_INDICES_OF_POSITION_ENCODING(pe, Numcy::cos(p * dt));*/\
                                                        temp = Numcy::cos(p * dt);\
                                                        FILL_ODD_INDICES_OF_POSITION_ENCODING(pe, temp);\
                                                        cc_tokenizer::allocator<char>().deallocate(reinterpret_cast<char*>(temp.ptr));\
                                                        temp = {NULL, {0, 0, NULL, NULL}};\
                                                   }\

/*
    @pe, position encoding
    @s, instance of Collective
 */
 #define FILL_EVEN_INDICES_OF_POSITION_ENCODING(pe, s) {\
                                                            for (cc_tokenizer::string_character_traits<char>::size_type i = 0; i < s.shape.getN(); i++)\
                                                            {\
                                                                pe[i*2 + 0] = s[i];\
                                                            }\
                                                       }\

/*
    @pe, position encoding
    @c, instance of Collective
 */
#define FILL_ODD_INDICES_OF_POSITION_ENCODING(pe, c) {\
                                                         for (cc_tokenizer::string_character_traits<char>::size_type i = 0; i < c.shape.getN(); i++)\
                                                         {\
                                                             pe[i*2 + 1] = c[i];\
                                                         }\
                                                     }\

/*
    @icp, input csv parser
    @tcp, target csv parser
    @ei, encoder input
    @di, decoder input
    @dm, dimensions of the model(d_model)
    @es, epochs, 
    @iv, input sequence vocabulary
    @tv, target sequence vocabulary
    @p, position
    @dt, division term
    @pe, position encoding
    @is, input sequence
    @ts, target sequence
    @t, type
    @v, be verbose when true
 */
#define TRAINING_LOOP_LINE_BATCH_SIZE(icp, tcp, ei, di, dm, es, iv, tv, p, dt, pe, is, ts, t, v)\
{\
    for (cc_tokenizer::string_character_traits<char>::size_type i = 0; i < es; i++)\
    {\
        if (v == true)\
        {\
            std::cout << "Epoch " << (i + 1) <<", batch size set to a single line and total number of lines in input vocabulary is "<<iv.get_number_of_lines()<< " and total number of lines in target vocabulary is "<<tv.get_number_of_lines()<<std::endl;\
        }\
        for (cc_tokenizer::string_character_traits<char>::size_type j = 0; j < iv.get_number_of_lines(); j++)\
        {\
            icp.get_line_by_number(j + 1);\
            tcp.get_line_by_number(j + 1);\
            if (v == true)\
            {\
                std::cout << "Status of Forward Pass " << (j + 1) << ", input tokens# "<< icp.get_total_number_of_tokens() << ", target tokens# "<< tcp.get_total_number_of_tokens() << std::endl;\
            }\
            BUILD_INPUT_SEQUENCE_FOR_LINE_BATCH_SIZE(is, iv, icp, t);\
            BUILD_TARGET_SEQUENCE_FOR_LINE_BATCH_SIZE(ts, tv, tcp, t);\
            BUILD_POSITION_ENCODING_FOR_LINE_BATCH_SIZE(p, is, dt, dm, pe, t);\
            /* Encoder Input */\
            ei = Numcy::concatenate(pe, is);\
            /* Decoder Input */\
            di = Numcy::concatenate(pe, ts);\
            /* Masks */\
            /* The srcMask composite is used as masking matrix for the self-attention mechanism in the Transformer model.*/\
            /* This mask is applied to the attention scores during the self-attention computation to prevent attending to future positions in the sequence. */ \
            Collective<t> srcMask = Numcy::triu<t>(Numcy::ones<t>({is.shape[NUMCY_DIMENSIONS_SHAPE_COLUMNS], is.shape[NUMCY_DIMENSIONS_SHAPE_COLUMNS], NULL, NULL}), 1, false);\
            Collective<t> tgtMask = Numcy::triu<t>(Numcy::ones<t>({di.shape[NUMCY_DIMENSIONS_SHAPE_COLUMNS], di.shape[NUMCY_DIMENSIONS_SHAPE_COLUMNS], NULL, NULL}), 1, false);\
            \
            Encoder encoder(DEFAULT_DIMENTIONS_OF_THE_TRANSFORMER_MODEL_HYPERPARAMETER, DEFAULT_NUMBER_OF_LAYERS_FOR_ENCODER_HYPERPARAMETER, DEFAULT_NUMBER_OF_ATTENTION_HEADS_HYPERPARAMETER, DEFAULT_DROP_OUT_RATE_HYPERPARAMETER);\
            encoder.forward<t>(ei);\
            /* Reference counting, manual memory management */\
            /* is.decrementReferenceCount(); */\
            /*ts.decrementReferenceCount();*/\
            /*p.decrementReferenceCount();*/\
            /*dt.decrementReferenceCount();*/\
            /*pe.decrementReferenceCount();*/\
            /*  */\
            /*ei.decrementReferenceCount();*/\
            /*di.decrementReferenceCount();*/\
            /*onesForSrcMask.decrementReferenceCount();*/\
        }\
    }\
}\

/*
    @ei, encoder input
    @di, decoder input
    @dm, dimensions of the model(d_model)
    @es, epochs, 
    @iv, input sequence vocabulary
    @tv, target sequence vocabulary
    @p, position
    @dt, division term
    @pe, position encoding
    @is, input sequence
    @ts, target sequence
    @t, type
    @v, be verbose when true
 */
#define TRAINING_LOOP(ei, di, dm, es, iv, tv, p, dt, pe, is, ts, t, v) {\
                                                for (cc_tokenizer::string_character_traits<char>::size_type i = 0; i < es; i++)\
                                                {\
                                                    /* NUMBER OF FORWARD PASSES PER EPOCH */\
                                                    for (cc_tokenizer::string_character_traits<char>::size_type j = 0; j < iv.len(REPLIKA_PK_USE_WHOLE)/DEFAULT_BATCH_SIZE_HYPERPARAMETER(iv.len(REPLIKA_PK_USE_WHOLE)); j++)\
                                                    {\
                                                        cc_tokenizer::string_character_traits<char>::size_type start_index = j*DEFAULT_BATCH_SIZE_HYPERPARAMETER(iv.len(REPLIKA_PK_USE_WHOLE)), end_index = (j + 1)*DEFAULT_BATCH_SIZE_HYPERPARAMETER(iv.len(REPLIKA_PK_USE_WHOLE)) >= iv.len(REPLIKA_PK_USE_WHOLE) ? (j + 1)*iv.len(REPLIKA_PK_USE_WHOLE) : (j + 1)*DEFAULT_BATCH_SIZE_HYPERPARAMETER(iv.len(REPLIKA_PK_USE_WHOLE));\
                                                        if (v == true)\
                                                        {\
                                                            std::cout << "Epoch " << (i + 1) << ": Status of Forward Pass " << (j + 1) << std::endl;\
                                                            std::cout << "Start Index: " << start_index << "\n"\
                                                            << "End Index (exclusive): " << end_index << "\n";\
                                                        }\
                                                        /* Build input batch, for encoder */\
                                                        BUILD_INPUT_SEQUENCE_NEW(iv, is, start_index, end_index, t);\
                                                        /* Build target batch, for decoder */\
                                                        BUILD_TARGET_SEQUENCE_NEW(tv, ts, start_index, end_index, t);\
                                                        BUILD_POSITION_ENCODING(dm, p, dt, pe, is);\
                                                        /* Encoder Input */\
                                                        ei = Numcy::concatenate(is, pe);\
                                                        /* Decoder Input */\
                                                        di = Numcy::concatenate(ts, pe);\
                                                        /*float* ptr = Numcy::ones({is.shape[NUMCY_DIMENSIONS_SHAPE_COLUMNS], is.shape[NUMCY_DIMENSIONS_SHAPE_COLUMNS], NULL, NULL});*/\
                                                        Numcy::triu(Numcy::ones({is.shape[NUMCY_DIMENSIONS_SHAPE_COLUMNS], is.shape[NUMCY_DIMENSIONS_SHAPE_COLUMNS], NULL, NULL}), {is.shape[NUMCY_DIMENSIONS_SHAPE_COLUMNS], is.shape[NUMCY_DIMENSIONS_SHAPE_COLUMNS], NULL, NULL}, 0);\
                                                        if (v == true)\
                                                        {\
                                                            std::cout<< "Forward pass information has been recorded successfully." << std::endl;\
                                                        }\
                                                        cc_tokenizer::allocator<char>().deallocate(reinterpret_cast<char*>(is.ptr));\
                                                        is = {NULL, {0, 0, NULL, NULL}};\
                                                        cc_tokenizer::allocator<char>().deallocate(reinterpret_cast<char*>(ts.ptr));\
                                                        ts = {NULL, {0, 0, NULL, NULL}};\
                                                        cc_tokenizer::allocator<char>().deallocate(reinterpret_cast<char*>(p.ptr));\
                                                        p = {NULL, {0, 0, NULL, NULL}};\
                                                        cc_tokenizer::allocator<char>().deallocate(reinterpret_cast<char*>(dt.ptr));\
                                                        dt = {NULL, {0, 0, NULL, NULL}};\
                                                        cc_tokenizer::allocator<char>().deallocate(reinterpret_cast<char*>(pe.ptr));\
                                                        pe = {NULL, {0, 0, NULL, NULL}};\
                                                        cc_tokenizer::allocator<char>().deallocate(reinterpret_cast<char*>(ei.ptr));\
                                                        ei = {NULL, {0, 0, NULL, NULL}};\
                                                        cc_tokenizer::allocator<char>().deallocate(reinterpret_cast<char*>(di.ptr));\
                                                        di = {NULL, {0, 0, NULL, NULL}};\
                                                    }\
                                                }\
                                            }\

/*
    @e, epoch, 
    @iseq, input sequence
    @v, verbose,
 */
#define TRAINING_LOOP_old(e, iseq, v) {\
                                        for (cc_tokenizer::string_character_traits<char>::size_type i = 0; i < e; i++)\
                                        {\
                                            if (v == true)\
                                            {\
                                                std::cout<<"Epoch: "<<(i + 1)<<std::endl;\
                                            }\
                                            std::cout<<iseq.shape.getN()/DEFAULT_BATCH_SIZE_HYPERPARAMETER(iseq.shape.getN())<<std::endl;\
                                            /* Number of forward passes per epoch */\
                                            for (cc_tokenizer::string_character_traits<char>::size_type j = 0; j < iseq.shape.getN()/DEFAULT_BATCH_SIZE_HYPERPARAMETER(iseq.shape.getN()); j++)\
                                            {\
                                                cc_tokenizer::string_character_traits<char>::size_type start_index = j*DEFAULT_BATCH_SIZE_HYPERPARAMETER(iseq.shape.getN()), end_index = 0 /*(j + 1)*DEFAULT_BATCH_SIZE_HYPERPARAMETER(iseq.shape.getN()) >= iseq.shape.getN() ? j*iseq.shape.getN()  | (j + 1)*DEFAULT_BATCH_SIZE_HYPERPARAMETER(iseq.shape.getN())*/;\
                                                /* I SO DON'T LIKE THIS BLOCK */\
                                                if ((j + 1)*DEFAULT_BATCH_SIZE_HYPERPARAMETER(iseq.shape.getN()) >= iseq.shape.getN())\
                                                {\
                                                    end_index = (j + 1)*iseq.shape.getN();\
                                                }\
                                                else\
                                                {\
                                                    end_index = (j + 1)*DEFAULT_BATCH_SIZE_HYPERPARAMETER(iseq.shape.getN());\
                                                }\
                                                \
                                                std::cout<<"This forward pass is "<<j + 1<<" and \"start index\" is "<<start_index<<" and \"end index\" is "<<end_index<<"(exclusive)"<<std::endl;\
                                            }\
                                        }\
                                  }\

#endif