/*
    lib/NLP/transformers/encoder-decoder/encoderlayer.hh
    Q@khaa.pk
 */

/*
    Transformer, has "encoder layers" (instead of one encoder we have few). Each "encoder layer" (or just an encoder) is very similar to other or all encoders have same architecture. 
    Each encoder or "encoder layer" consists of two layers: Self-attention and a feed Forward Neural Network. 

    As is the case in NLP applications in general, we begin by turning each input word into a vector.
    After embedding the words in input sequence, each of them flows through each of the two layers of the encoder.   
    The embedding only happens in the bottom most encoder, but in other encoders, it would be the output of the encoder that is directly below.
 */

#include "./attention.hh"
#include "./header.hh"

#ifndef NLP_ENCODER_DECODER_TRANSFORMER_MODEL_ENCODER_LAYER_HH
#define NLP_ENCODER_DECODER_TRANSFORMER_MODEL_ENCODER_LAYER_HH

/*
    Transformer, has "encoder layers" (instead of one encoder we have few). 
 */
typedef struct EncoderLayerList
{
    class EncoderLayer* ptr; 

    struct EncoderLayerList* next;
    struct EncoderLayerList* previous;
} ENCODERLAYERLIST;

typedef ENCODERLAYERLIST* ENCODERLAYERLIST_PTR;

/*
    def __init__(self, d_model, num_heads, dropout_rate):
        self.multihead_attention = MultiHeadAttention(d_model, num_heads)
        self.dropout_rate = dropout_rate
 */

typedef class EncoderLayer
{
    MULTIHEADATTENTION attention;
    cc_tokenizer::string_character_traits<char>::size_type dimensionsOfTheModel, numberOfAttentionHeads;
    float dropOutRate;

    public:
        EncoderLayer() : dimensionsOfTheModel(DEFAULT_DIMENTIONS_OF_THE_TRANSFORMER_MODEL_HYPERPARAMETER), numberOfAttentionHeads(DEFAULT_NUMBER_OF_ATTENTION_HEADS_HYPERPARAMETER), dropOutRate(DEFAULT_DROP_OUT_RATE_HYPERPARAMETER)
        {
            attention = MULTIHEADATTENTION();
        }
        /*
            @d_model, name from the paper "Attention is all we need" we call it "dimensionsOfTheModel". 
            @num_heads, Number of attention heads. 
            @dropout_rate, Dropout rate for regularization. The dropout_rate in the Transformer model is a regularization technique to prevent overfitting.
         */
        EncoderLayer(cc_tokenizer::string_character_traits<char>::size_type d_model, cc_tokenizer::string_character_traits<char>::size_type num_heads, float dropout_rate) : dropOutRate(dropout_rate)
        {
            // self.multihead_attention = MultiHeadAttention(d_model, num_heads)
            attention = MULTIHEADATTENTION(d_model, num_heads);
        }

        ~EncoderLayer()
        {            
        }

} ENCODERLAYER;

typedef ENCODERLAYER* ENCODERLAYER_PTR;

#endif