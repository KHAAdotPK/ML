/*
    lib/NLP/transformers/encoder-decoder/attention.hh
    Q@khaa.pk
 */

#include "../../../../numcy/numcy.hh"
#include "./header.hh"

#ifndef NLP_ENCODER_DECODER_TRANSFORMER_MODEL_ATTENTION_HH
#define NLP_ENCODER_DECODER_TRANSFORMER_MODEL_ATTENTION_HH

/*
    Multi head attention
 */
typedef class Attention
{
    cc_tokenizer::string_character_traits<char>::size_type dimensionsOfTheModel, numberOfAttentionHeads;

    public:
        Attention(void) : dimensionsOfTheModel(DEFAULT_DIMENTIONS_OF_THE_TRANSFORMER_MODEL_HYPERPARAMETER), numberOfAttentionHeads(DEFAULT_NUMBER_OF_ATTENTION_HEADS_HYPERPARAMETER)
        {           
        }

        /*
            @d_model, name from the paper "Attention is all we need" we call it "dimensionsOfTheModel". 
            @num_heads, Number of attention heads.            
         */
        Attention(cc_tokenizer::string_character_traits<char>::size_type d_model, cc_tokenizer::string_character_traits<char>::size_type num_heads) : dimensionsOfTheModel(d_model), numberOfAttentionHeads(num_heads)
        {            
        }

        template <typename t = float>
        void forward(Collective<t>& ei)
        {
            
        }

        ~Attention()
        {                        
        }


} ATTENTION, MULTIHEADATTENTION;

#endif