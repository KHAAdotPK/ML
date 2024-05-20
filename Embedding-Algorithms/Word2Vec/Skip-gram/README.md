## ~~Shallow~~ Neural Network
**Overview of feedforward(without cycles or loops, information moves in one direction) Skip-gram model/architecture**. 
It leverages a single hidden layer to capture the semantic representations of words and predict surrounding context words in a given corpus.
### Neural Network Structure
In terms of neural network terminology, here's how our implementation of skip-gram model fits in:
1. Input layer.
The input is a one-hot encoded vector representing the target word. If the vocabulary size is 𝑉, the input vector 𝑥 will have a size of 𝑉. 
~~What is a One-Hot Encoded Vector?
A one-hot encoded vector is a representation of a categorical variable (like a word) as a binary vector.~~
**Instead of one-hot vector this implementation uses a linkedlist of pairs**... 
The input vector x(it is linked list of pairs, each pair has a center/target word and corresponding context words on its left and right. Number of context words on each side of center/target words is determined by the macro SKIP_GRAM_WINDOW_SIZE. This macro is a hyperparameter). The size or the number of links in this linked list is equal to the number of unique(no redundency) words in the vocabulary, if number of such words is V then linkned list will have V many links as well. 
 ```C++
struct WordPairs 
{
    private:
        CONTEXTWORDS_PTR left;
        /* The index of center/target word into vocabulary makes the linked list of pairs(or vecto x) as one-hot encoded vector */
        cc_tokenizer::string_character_traits<char>::size_type centerWord;
        CONTEXTWORDS_PTR right;

        struct WordPairs* next;
        struct WordPairs* prev;
};
```
2. Hidden layer.
 The input layer is connected to the hidden layer through a weight matrix 𝑊1 of size 𝑉×𝑁.
 ```C++
#define SKIP_GRAM_EMBEDDNG_VECTOR_SIZE 100
Collective<double> W1 = Numcy::Random::randn<double>(DIMENSIONS{SKIP_GRAM_EMBEDDNG_VECTOR_SIZE, vocab.numberOfUniqueTokens(), NULL, NULL});
```
The hidden layer size is typically denoted by 𝑁 and represents the embedding dimensions. It is one of the hyperparameters. The input layer is connected to the hidden layer through a weight matrix 𝑊1 of size 𝑉×𝑁. Each row of 𝑊1 holds the embedding of one specific target/center word. The hidden layer represents one embedding vector (of the input/target/center word), computed as ℎ=𝑊1^T𝑥, out of the 𝑉 possible embeddings in 𝑊1. During training, this operation is performed for each target word in the input data, typically processed in batches that are smaller than the entire vocabulary, across all epochs(our vocabulary is small so our bachsize is V).
```C++
#define SKIP_GRAM_EMBEDDNG_VECTOR_SIZE 100
T* h_ptr = cc_tokenizer::allocator<T>().allocate(W1.getShape().getNumberOfColumns());
/* Epoch loop */
for (cc_tokenizer::string_character_traits<char>::size_type i = 1; i <= epoch; i++)
{
    /* Batch processing */
    while (pairs.go_to_next_word_pair() != cc_tokenizer::string_character_traits<char>::eof())
    {
        for (cc_tokenizer::string_character_traits<char>::size_type i = 0; i < W1.getShape().getNumberOfColumns(); i++)
        {
            *(h_ptr + i) = W1[W1.getShape().getNumberOfColumns()*(pair->getCenterWord() - INDEX_ORIGINATES_AT_VALUE) + i];
        }
        Collective<T> h = Collective<T>{h_ptr, DIMENSIONS{W1.getShape().getNumberOfColumns(), 1, NULL, NULL}};
    }
}
```
3. Output Layer.
The hidden layer is connected to the output layer through another weight matrix 𝑊2 of size 𝑁×𝑉.
```C++
Collective<double> W2 = W2 = Numcy::Random::randn(DIMENSIONS{vocab.numberOfUniqueTokens(), SKIP_GRAM_EMBEDDNG_VECTOR_SIZE, NULL, NULL});
```
+ The output layer produces a score for each word in the vocabulary, typically using the softmax function to convert scores into probabilities **scores=W2^T*h** and **probabilities=softmax(scores)**
```C++
/*
    Hidden layer and output layer operations:
    1. The hidden layer vector ℎ is prepared.
    2. The dot product computes the scores for each word in the vocabulary.
    3. The softmax function converts these scores into probabilities. 
 */
Collective<T> h = Collective<T>{h_ptr, DIMENSIONS{W1.getShape().getNumberOfColumns(), 1, NULL, NULL}};
Collective<T> u = Numcy::dot(h, W2);
Collective<T> y_pred = softmax(u);

/*
    One-Hot Encoding of Context Words:
 */
Collective<double> oneHot = Numcy::zeros(DIMENSIONS{vocab.numberOfUniqueTokens(), 1, NULL, NULL});
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
    if (((*(pair->getRight()))[i] - INDEX_ORIGINATES_AT_VALUE) < vocab.numberOfUniqueTokens())
    {
        oneHot[(*(pair->getRight()))[i] - INDEX_ORIGINATES_AT_VALUE] = 1;
    }        
}
/*
    Gradient Calculation and Weight Updates.
 */    
/*
    Gradient Calculation:
    1. grad_u computes the gradient of the loss with respect to the scores by subtracting the one-hot encoded vector from the predicted probabilities.
    2. grad_W2 calculates the gradient of the loss with respect to W2 using the outer product of the hidden layer activations and grad_u.
    3. grad_h is the gradient with respect to the hidden layer activations, obtained by multiplying grad_u with the transpose of 𝑊2.
 */    
Collective<double> grad_u = Numcy::subtract<double>(fp.predicted_probabilities, oneHot);
Collective<double> grad_W2 = Numcy::outer(fp.intermediate_activation, grad_u);
Collective<double> W2_T = Numcy::transpose(W2);
Collective<double> = grad_h = Numcy::dot(grad_u, W2_T);
/*
    Weight Updates:
    1. The loop updates grad_W1 by accumulating the gradients with respect to 𝑊1 for the center word.
 */
Collective<double> grad_W1 = Numcy::zeros(DIMENSIONS{SKIP_GRAM_EMBEDDNG_VECTOR_SIZE, vocab.numberOfUniqueTokens(), NULL, NULL});
for (cc_tokenizer::string_character_traits<char>::size_type i = 0; i < grad_W1.getShape().getNumberOfColumns(); i++)
{
    grad_W1[(pair->getCenterWord() - INDEX_ORIGINATES_AT_VALUE)*SKIP_GRAM_EMBEDDNG_VECTOR_SIZE + i] += grad_h[i];
}
```

