/*
    lib/NLP/transformers/encoder-decoder/attention.hh
    Q@khaa.pk
 */

#include "../../../../numcy/numcy.hh"
#include "./header.hh"

#ifndef NLP_ENCODER_DECODER_TRANSFORMER_MODEL_ATTENTION_HH
#define NLP_ENCODER_DECODER_TRANSFORMER_MODEL_ATTENTION_HH

/*
    Multi head attention.
 */
typedef class Attention
{
    cc_tokenizer::string_character_traits<char>::size_type dimensionsOfTheModel, numberOfAttentionHeads;

    public:
        Attention(void) : dimensionsOfTheModel(DEFAULT_DIMENTIONS_OF_THE_TRANSFORMER_MODEL_HYPERPARAMETER), numberOfAttentionHeads(DEFAULT_NUMBER_OF_ATTENTION_HEADS_HYPERPARAMETER)
        {   
            /*DIMENSIONS dim3 = {10, 3, NULL, NULL};
            DIMENSIONS dim2 = {0, 10, &dim3, NULL};
            dim3.prev = &dim2;
            DIMENSIONS dim1 = {0, 78, &dim2, NULL};
            dim2.prev = &dim1;
            DIMENSIONS dim = {0, 9, &dim1, NULL};
            dim1.prev = &dim;               
            Numcy::Random::randn(dim);*/
           //Numcy::Random::randn(DIMENSIONS{0, 0, NULL, NULL});
        }

        /*
            @d_model, name from the paper "Attention is all we need" we call it "dimensionsOfTheModel". 
            @num_heads, Number of attention heads.            
         */
        Attention(cc_tokenizer::string_character_traits<char>::size_type d_model, cc_tokenizer::string_character_traits<char>::size_type num_heads) : dimensionsOfTheModel(d_model), numberOfAttentionHeads(num_heads)
        { 
            /*DIMENSIONS dim3 = DIMENSIONS{10, 3, NULL, NULL};
            DIMENSIONS dim2 = DIMENSIONS{0, 10, &dim3, NULL};
            dim3.prev = &dim2;
            DIMENSIONS dim1 = DIMENSIONS{0, 78, &dim2, NULL};
            dim2.prev = &dim1;
            DIMENSIONS dim = DIMENSIONS{0, 9, &dim1, NULL};
            dim1.prev = &dim;*/

            //DIMENSIONS dim3(DIMENSIONS{10, 3, NULL, NULL});
            //DIMENSIONS dim2(DIMENSIONS{0, 10, &dim3, NULL});

            //cc_tokenizer::string_character_traits<char>::size_type *ptr = cc_tokenizer::allocator<cc_tokenizer::string_character_traits<char>::size_type>().allocate(5);            
            cc_tokenizer::string_character_traits<char>::size_type *ptr = reinterpret_cast<cc_tokenizer::string_character_traits<char>::size_type*>(cc_tokenizer::allocator<unsigned int>().allocate(5));
            ptr[0] = 9;
            ptr[1] = 78;
            ptr[2] = 10;
            ptr[3] = 3;
            ptr[4] = 10;

            DIMENSIONSOFARRAY dimensionsOfArray(ptr, 5);

            DIMENSIONS dim(dimensionsOfArray);

            std::cout << "------->>>>>>> " << dim.getDimensionsOfArray().getNumberOfInnerArrays() << std::endl;
                        
            Numcy::Random::randn(dim);

            std::cout<< "Yes..." << std::endl;           
        }

        /*
            @ei, encoder input
         */
        template <typename t = float>
        void forward(Collective<t>& ei)
        {
            std::cout<< "Columns = " << ei.getShape().getNumberOfColumns() << ", Rows = " << ei.getShape().getNumberOfRows().getNumberOfInnerArrays() << std::endl;
        }

        ~Attention()
        {                        
        }


} ATTENTION, MULTIHEADATTENTION;

#endif