/*
    ML/NLP/unsupervised/Word2Vec/Skip-gram/header.hh
    Q@khaa.pk
 */

/*
    The skip-gram takes a target(center) word and tries to predict the surrounding context words.

    Skip-gram and Context Word Prediction.
    --------------------------------------
    1. Skip-gram aims to predict the surrounding context words (both positive and negative samples) based on the central word's embedding.
    2. During training, we iterate through word pairs in the corpus. The central word is the "target" we want to predict 
       surrounding words for.
    3. We retrieve the central word's embedding vector from W1 (the embedding layer).
    4. We then multiply this embedding vector with W2 (the output layer, although not a traditional one interms of NN).
       This multiplication essentially calculates a score for each potential context word (including negative samples) based on their
       relationship to the central word.
    5. Higher scores indicate a stronger predicted co-occurrence between the central word and the context word.

    Why Predict Context Words?
    1. By predicting context words based on the central word's embedding, Skip-gram learns to capture semantic relationships 
       between words.
    2. Words that often appear together in similar contexts are likely to have similar embeddings.
 */

//#ifndef R3PLICA_SKIP_GRAM_HEADER_HH
#include <random> // For random number generation
#include <unordered_set> // For efficient lookup of words

//#include "hyper-parameters.hh"
#ifndef R3PLICA_SKIP_GRAM_PAIRS_HH
//#include "skip-gram-pairs.hh"
#endif
#ifndef CC_TOKENIZER_REPLIKA_PK_SKIP_GRAM_SKIP_GRAM_H_HH
//#include "skip-gram.hh"
#endif
//#endif

#ifndef R3PLICA_SKIP_GRAM_HEADER_HH
#define R3PLICA_SKIP_GRAM_HEADER_HH

#include "hyper-parameters.hh"
#include "skip-gram-pairs.hh"
#include "skip-gram.hh"

/*
    To implement a function to generate negative samples for the Skip-gram model,
    we need to randomly select words from the vocabulary that do not appear within the context window of the central word.
    These negative samples are used to train the model to distinguish between words that are likely to appear together and those that are not.

    Is it ok to not find a single center word which is not in context words?
    It's perfectly normal and even expected in Skip-gram training not to find a single central word that doesn't appear in any context words.
    Here's why:
    1. Vocabulary Coverage: Typically, the vocabulary used for Skip-gram training includes most words encountered in your text corpus.
                            This means the chances of finding a central word that's completely absent from all context windows are very low.
    2. Context Window Size: Skip-gram focuses on words that appear close together within a defined window size.
                            It's highly likely that most words in your vocabulary will co-occur with at least one other word within that window in your corpus.
    3. Rare Words: Even for rare words, there's a chance they might co-occur with other rare words within the window size.
                   While the frequency of such co-occurrences might be low, it's still possible.                   

    @vocab,
    @pairs,
    @n, number of samples
 */
template <typename E = cc_tokenizer::string_character_traits<char>::size_type>
void generateNegativeSamples(CORPUS_REF vocab, SKIPGRAMPAIRS_REF pairs, E n = 10)
{
    // Initialize random number generator
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_int_distribution<E> distrib(0, vocab.numberOfUniqueTokens() - 1);

    // Set to store context words of each central word
    std::unordered_set<cc_tokenizer::string_character_traits<char>::size_type> context_words;

    while (pairs.go_to_next_word_pair() != cc_tokenizer::string_character_traits<char>::eof())
    {
        WORDPAIRS_PTR current_word_pair = pairs.get_current_word_pair();

        CONTEXTWORDS_PTR left_context = current_word_pair->getLeft();
        CONTEXTWORDS_PTR right_context = current_word_pair->getRight();
       
        for (E i = 0; i < SKIP_GRAM_WINDOW_SIZE; i++)
        {
            if (left_context->array[i] != INDEX_NOT_FOUND_AT_VALUE)
            {
                context_words.insert(left_context->array[i]);
            }

            if (right_context->array[i] != INDEX_NOT_FOUND_AT_VALUE)
            {
                context_words.insert(right_context->array[i]);
            }
        }
    }

    // Traverse the set using an iterator
    /*for (auto it = context_words.begin(); it != context_words.end(); ++it)
    {
        // Access the current element using the iterator
        cc_tokenizer::string_character_traits<char>::size_type current_word_index = *it;
        // Process the current word index
        std::cout << "Word index: " << current_word_index << std::endl;
    }*/
    
    // Generate negative samples
    for (E i = 0; i < n; i++)
    {
        // Randomly select a central word from the vocabulary, central words are all the unique words
        cc_tokenizer::string_character_traits<char>::size_type central_word_index = distrib(gen);
        cc_tokenizer::String<char> central_word = vocab[central_word_index + INDEX_ORIGINATES_AT_VALUE];

        //std::cout<< "count = " << context_words.size() << std::endl;

        // Check if the central word is not in the context words set
        if (context_words.find(central_word_index + INDEX_ORIGINATES_AT_VALUE) == context_words.end())
        {
            //std::cout<< "count = " << context_words.size() << std::endl;

            // This is a negative sample
            // You can process or store it as needed
            // For example:
            std::cout << "Negative sample: " << central_word.c_str() << std::endl;
        }
    }
}

/*
// Function to generate negative samples
std::vector<int> generateNegativeSamples_new(const std::unordered_map<std::string, int>& vocabulary, 
                                         int targetWordIndex, int numSamples, 
                                         const std::vector<double>& wordProbabilities) {
  std::random_device rd;
  std::mt19937 generator(rd());
  std::discrete_distribution<int> wordDist(wordProbabilities.begin(), wordProbabilities.end());

  std::vector<int> negativeSamples;
  for (int i = 0; i < numSamples; ++i) {
    // Sample a random word (excluding the target word) based on word probabilities
    int negativeSampleIndex = wordDist(generator);
    while (negativeSampleIndex == targetWordIndex) {
      // If the sampled word is the target word, re-sample
      negativeSampleIndex = wordDist(generator);
    }
    negativeSamples.push_back(negativeSampleIndex);
  }
  return negativeSamples;
}
 */
/*
    Negative Sampling
    -----------------
    Skip-gram uses negative sampling to generate negative word pairs that the model shouldn't
    predict as likely neighbors of the central word. You'll need to implement a function that samples
    words from the vocabulary with a probability proportional to their frequency
    (e.g., using a weighted random sampling approach).
 */
template <typename E = cc_tokenizer::string_character_traits<char>::size_type>
void generateNegativeSamples_new (CORPUS_REF vocab, SKIPGRAMPAIRS_REF pairs, E n = 10)
{
    std::vector<double> wordProbabilities;

    try
    {
        wordProbabilities = vocab.getWordProbabilities();
    }
    catch (ala_exception& e)
    {
        std::cout<< e.what() << std::endl;

        return;
    }

    // Set to store context words of each central word
    std::unordered_set<cc_tokenizer::string_character_traits<char>::size_type> context_words;

    while (pairs.go_to_next_word_pair() != cc_tokenizer::string_character_traits<char>::eof())
    {
        WORDPAIRS_PTR current_word_pair = pairs.get_current_word_pair();

        CONTEXTWORDS_PTR left_context = current_word_pair->getLeft();
        CONTEXTWORDS_PTR right_context = current_word_pair->getRight();
       
        for (E i = 0; i < SKIP_GRAM_WINDOW_SIZE; i++)
        {
            if (left_context->array[i] != INDEX_NOT_FOUND_AT_VALUE)
            {
                context_words.insert(left_context->array[i]);
            }

            if (right_context->array[i] != INDEX_NOT_FOUND_AT_VALUE)
            {
                context_words.insert(right_context->array[i]);
            }
        }
    }

    std::random_device rd;
    std::mt19937 generator(rd());
    std::discrete_distribution<E> wordDist(wordProbabilities.begin(), wordProbabilities.end());

    for (E i = 0; i < n; i++)
    {
        // Sample a random word (excluding the target word) based on word probabilities
        // "negativeSampleIndex" originates at zero, I checked that wordDist can also return index value of 0 as well
        E negativeSampleIndex = wordDist(generator);

        cc_tokenizer::String<char> central_word = vocab[negativeSampleIndex + INDEX_ORIGINATES_AT_VALUE];

        // UNCOMMENT THIS
        /*std::cout<< "E = " << negativeSampleIndex << " ";*/

        // Check if the central word is not in the context words set
        if (context_words.find(negativeSampleIndex + INDEX_ORIGINATES_AT_VALUE) == context_words.end())
        {
            //std::cout<< "count = " << context_words.size() << std::endl;

            // This is a negative sample
            // You can process or store it as needed
            // For example:
            std::cout << "Negative sample: " << central_word.c_str() << std::endl;
        }
    }

    std::cout<< std::endl;
}

#endif