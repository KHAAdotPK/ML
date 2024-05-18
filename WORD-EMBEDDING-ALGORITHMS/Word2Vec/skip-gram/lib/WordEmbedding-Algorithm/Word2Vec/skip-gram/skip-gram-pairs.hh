/*
    ML/NLP/unsupervised/Word2Vec/Skip-Gram/skip-gram-pairs.hh
    Q@khaa.pk
 */

/*
    TODO: Address handling for the last token/word of the corpus when it serves as the center/target word in a pair. 
    Given that there are no words/tokens to its right, the 'right' array contains values other than INDEX_NOT_FOUND_AT_VALUE. 
    For such a center/target word, it's crucial to verify the values in the 'right' array. 
    Specifically, we need to ensure that they don't exceed the total number of center/target words or all words (including redundancies).

    Conditions to check:
    
    1. Check if the first element of 'right' array points to a valid index in the vocabulary:
       if (vocab[(*(pair->getRight()))[0]].size() == 0) {}

    2. Check if the second element of 'right' array points to a valid index in the vocabulary:
       if (vocab[(*(pair->getRight()))[1]].size() == 0) {}

    3. Verify if the index referenced by the first element of 'right' array exceeds the total number of tokens:
       if ( (*(pair->getRight()))[0] > vocab.numberOfToken() ) {}

    4. Verify if the index referenced by the second element of 'right' array exceeds the total number of tokens:
       if ( (*(pair->getRight()))[1] > vocab.numberOfToken() ) {}    
 */

/*
    This should be moved to the respective methods being used here...
    std::cout<< vocab[pair->getCenterWord()].c_str() << std::endl;\
    std::cout<< pair->getLeft(0) << std::endl;\
    if ((*(pair->getRight()))[0] == INDEX_NOT_FOUND_AT_VALUE)\
    {\
        std::cout<<" YES YES YES YES YES YES YES YES YES YES " << std::endl;\
    }\
    std::cout<< vocab[(*(pair->getRight()))[0]].c_str() << std::endl;\
    std::cout<< vocab[(*(pair->getRight()))[1]].c_str() << std::endl;\
 */

#ifndef R3PLICA_SKIP_GRAM_PAIRS_HH
#define R3PLICA_SKIP_GRAM_PAIRS_HH

#include "header.hh"

struct SkipGramPairs;

typedef struct ContextWord
{   
    cc_tokenizer::string_character_traits<char>::size_type operator[](cc_tokenizer::string_character_traits<char>::size_type index) const
    {
        if (index >= SKIP_GRAM_WINDOW_SIZE)
        {            
            return INDEX_NOT_FOUND_AT_VALUE;
        }
        
        return array[index];   
    }

    cc_tokenizer::string_character_traits<char>::size_type array[SKIP_GRAM_WINDOW_SIZE];

} CONTEXTWORDS;

typedef CONTEXTWORDS* CONTEXTWORDS_PTR;

typedef struct WordPairs 
{
    // Declare a friend    
    friend struct SkipGramPairs;

    WordPairs(WordPairs& ref) : centerWord(ref.centerWord), left(ref.left), right(ref.right)
    {
        next = NULL;
        prev = NULL;
    } 

    WordPairs& operator=(const WordPairs& other)
    {
        if (this != &other) // Check for self-assignment
        { 
            this->centerWord = other.centerWord;
            this->left = other.left;
            this->right = other.right;
        }

        return *this;
    }

    /*
        (*(pair->getLeft()))[0]
         vocab[(*(pair->getLeft()))[0]].c_str()
     */
    cc_tokenizer::string_character_traits<char>::size_type getLeft(cc_tokenizer::string_character_traits<char>::size_type index)
    {
        if (index >= SKIP_GRAM_WINDOW_SIZE)
        {
            return INDEX_NOT_FOUND_AT_VALUE;
        }

        // The furthest context word from the center/target word is at index ((SKIP_GRAM_WINDOW_SIZE - 1) - index)
        return left->array[index];
    }

    CONTEXTWORDS_PTR getLeft(void) 
    {
        return left;
    }

    /*
        (*(pair->getRight()))[0]
        vocab[(*(pair->getRight()))[0]].c_str()
     */
    cc_tokenizer::string_character_traits<char>::size_type getRight(cc_tokenizer::string_character_traits<char>::size_type index)
    {   
        /*if (this->next == NULL)
        {
            return INDEX_NOT_FOUND_AT_VALUE;
        }*/

        if (index >= SKIP_GRAM_WINDOW_SIZE)
        {
            return INDEX_NOT_FOUND_AT_VALUE;
        }

        // The furthest context word from the center/target word is at index (SKIP_GRAM_WINDOW_SIZE - 1)
        return right->array[index];
    }

    CONTEXTWORDS_PTR getRight(void) 
    {
        return right;
    }

    cc_tokenizer::string_character_traits<char>::size_type getCenterWord(void)
    {
        return centerWord;    
    }

    private:
        CONTEXTWORDS_PTR left;
        cc_tokenizer::string_character_traits<char>::size_type centerWord;
        CONTEXTWORDS_PTR right;

        struct WordPairs* next;
        struct WordPairs* prev;

} WORDPAIRS;

typedef WORDPAIRS* WORDPAIRS_PTR;
/*
    In the context of word embedding models like Skip-gram,
    word pairs refer to a central word and the surrounding words within a specific window size (SKIP_GRAM_WINDOWS_SIZE).

    Skip-gram aims to predict the surrounding words based on the central word. 
    There are two main types of word pairs to consider in Skip-gram:
    1. Positive pairs: These pairs consist of the central word and its surrounding words within the window. 
       Since these words tend to co-occur frequently, they represent a positive relationship for the model to learn.
    2. Negative pairs: These are artificially generated pairs that don't typically appear together.
       They help the model distinguish between words with similar but not identical meanings.
       Negative pairs are often formed by randomly selecting words from the vocabulary that are not within the window of the central word.

    By processing a large corpus of text and analyzing these positive and negative word pairs, 
    Skip-gram learns the underlying relationships between words and builds a meaningful vector representation for each word.      
 */
typedef struct SkipGramPairs 
{    
    SkipGramPairs(CORPUS& vocab, bool verbose = false) : head(NULL), current_pair(NULL), n(0)
    {
        WORDPAIRS_PTR current_word_pair = NULL;

        /*
            At each i + INDEX_ORIGINATES_AT_VALUE we've the center word
            Context words are pair of words around this center word...
            SKIP_GRAM_WINDOW_SIZE center-word SKIP_GRAM_WINDOW_SIZE
            
            We must monitor redundancy as we construct the entire vocabulary from the corpus.
            It's crucial to maintain records of any redundant words, that is why the use of numberOfTokens() method instead of numberOfUniqueTokens()
         */
        for (cc_tokenizer::string_character_traits<char>::size_type i = 0; i < vocab.numberOfTokens(); i++)
        {
            if (head == NULL)
            {   
                try            
                {                
                    head = reinterpret_cast<WORDPAIRS_PTR>(cc_tokenizer::allocator<char>().allocate(sizeof(WORDPAIRS)));
                    head->next = NULL;
                    head->prev = NULL;

                    head->centerWord = vocab[vocab(i + INDEX_ORIGINATES_AT_VALUE, true)]; // i originated at 0, 
                                                                                          // vocabulary indices originate at INDEX_ORIGINATES_AT_VALUE
                                                                                          // We are traversing the vocabulary with redundency flag on. That is becuase of our choice for the outer for loop
                    // Later on find the surrounding context words
                    head->left = NULL;
                    head->right = NULL;

                    /*
                        Pointer "current_word_pair" is a displacement in to our linked list and it points to the word pair 
                        which will hold the center word and the all its context words
                     */
                    current_word_pair = head;

                    n = n + 1; // Increment number of word pair counter
                }
                catch (std::bad_alloc& e)
                {
                    head = NULL;
                    
                    throw ala_exception(e.what());
                }
            }
            else
            {                
                current_word_pair = head;

                // Get to the tail of the linked list, we already know that head is not null
                while (current_word_pair->next != NULL)
                {
                    current_word_pair = current_word_pair->next;
                }
                
                try
                {
                    // Add new link to the already existing linked list
                    current_word_pair->next = reinterpret_cast<WORDPAIRS_PTR>(cc_tokenizer::allocator<char>().allocate(sizeof(WORDPAIRS)));
                    current_word_pair->next->next = NULL;
                    current_word_pair->next->prev = current_word_pair;
                    /*
                        Pointer "current_word_pair" is a displacement in to our linked list and it points to the word pair 
                        which will hold the center word and the all its context words
                     */
                    current_word_pair = current_word_pair->next;

                    current_word_pair->centerWord = vocab[vocab(i + INDEX_ORIGINATES_AT_VALUE, true)]; // i originated at 0,
                                                                                                       // vocabulary indices originate at INDEX_ORIGINATES_AT_VALUE 
                                                                                                       // We traversing the vocabulary with redundency flag on. That is becuase of our choice for the outer for loop 
                    // Later on find the surrounding context words
                    current_word_pair->left = NULL;
                    current_word_pair->right = NULL;

                    n = n + 1; // Increment number of word pair counter
                }
                catch (std::bad_alloc& e)
                {
                     current_word_pair->next = NULL;

                     throw ala_exception(e.what()); 
                }                
            }

            // Find context words around the center word 
            // Pointer "current_word_pair" is a displacement in to our linked list and it points to the word pair which will hold the center word and the all its context words

            // context words on left, number of these at max would be SKIP_GRAM_WINDOW_SIZE
            CONTEXTWORDS_PTR left = NULL;
            try
            {                            
                left = reinterpret_cast<CONTEXTWORDS_PTR>(cc_tokenizer::allocator<char>().allocate(sizeof(CONTEXTWORDS)));
                memset(left->array, INDEX_NOT_FOUND_AT_VALUE, SKIP_GRAM_WINDOW_SIZE);
            }
            catch (std::bad_alloc& e)
            {
                left = NULL;

                throw ala_exception(e.what());
            }

            // Context words on the left of center word. Start on the left most and then move closer to the target/center word
            for (cc_tokenizer::string_character_traits<char>::size_type j = SKIP_GRAM_WINDOW_SIZE; j > 0; /*SKIP_GRAM_WINDOW_SIZE;*/ j--)
            {            
                if (vocab(((i + INDEX_ORIGINATES_AT_VALUE) - j), true).size())
                {
                    left->array[j - 1] = vocab[vocab(((i + INDEX_ORIGINATES_AT_VALUE) - j), true)];

                    if (verbose)
                    {
                        std::cout<< vocab(((i + INDEX_ORIGINATES_AT_VALUE) - j), true).c_str() << "  ";
                    }
                }
                else
                {
                    left->array[j - 1] =  INDEX_NOT_FOUND_AT_VALUE;

                    if (verbose)
                    {
                        std::cout<< "NONE" << "  ";
                    }
                }                
            }

            current_word_pair->left = left;

            left = NULL;

            if (verbose)
            {
                // The center word
                std::cout<< "[ " << vocab(i + INDEX_ORIGINATES_AT_VALUE, true).c_str() << " ] ";
            }

            CONTEXTWORDS_PTR right = NULL;
            try
            {
                right = reinterpret_cast<CONTEXTWORDS_PTR>(cc_tokenizer::allocator<char>().allocate(sizeof(CONTEXTWORDS)));
                memset(right->array, INDEX_NOT_FOUND_AT_VALUE, SKIP_GRAM_WINDOW_SIZE);
            }
            catch(std::bad_alloc& e)
            {
                right = NULL;

                throw ala_exception(e.what()); 
            }

            // Context words to the right of center word only when there are atleast one context word to the right
            // We need to stop this loop at the last word as target/center word and not look for context words on the right
            for (cc_tokenizer::string_character_traits<char>::size_type j = 1; j <= SKIP_GRAM_WINDOW_SIZE && (i + INDEX_ORIGINATES_AT_VALUE + j) /* Not go beyond the last (center/target) word */ <= vocab.numberOfTokens(); j++)
            {                
                if (vocab(((i + INDEX_ORIGINATES_AT_VALUE) + j), true).size())
                {
                    if (verbose)
                    {
                        std::cout<< vocab(((i + INDEX_ORIGINATES_AT_VALUE) + j), true).c_str() << "  ";

                        //std::cout<< "-------->>>>>>>> "  << vocab[vocab((i + INDEX_ORIGINATES_AT_VALUE) + j, true)] << std::endl;
                    }

                    right->array[j - 1] = vocab[vocab(((i + INDEX_ORIGINATES_AT_VALUE) + j), true)];
                }
                else
                {
                    right->array[j - i] = INDEX_NOT_FOUND_AT_VALUE;

                    if (verbose)
                    {
                        std::cout<< "NONE" << "  ";
                    }
                }
            }

            current_word_pair->right = right;

            right = NULL;

            if (verbose)
            {
                std::cout<< std::endl;            
            }
        }        
    }

    ~SkipGramPairs()
    {
        if (head == NULL)
        {
            return;
        }

        WORDPAIRS_PTR current_wordpair = head;

        while (current_wordpair->next != NULL)
        {
            current_wordpair = current_wordpair->next;
        }

        while (current_wordpair->prev != NULL)
        {
            current_wordpair = current_wordpair->prev;

            cc_tokenizer::allocator<char>().deallocate(reinterpret_cast<char*>(current_wordpair->next->left));
            cc_tokenizer::allocator<char>().deallocate(reinterpret_cast<char*>(current_wordpair->next->right));

            cc_tokenizer::allocator<char>().deallocate(reinterpret_cast<char*>(current_wordpair->next));
        }

        cc_tokenizer::allocator<char>().deallocate(reinterpret_cast<char*>(current_wordpair->left));
        cc_tokenizer::allocator<char>().deallocate(reinterpret_cast<char*>(current_wordpair->right));
        cc_tokenizer::allocator<char>().deallocate(reinterpret_cast<char*>(current_wordpair));        
    }

    /*
        PLEASE NOTE: current_pair gets NULL value at the end of loop on its own. 
     */
    cc_tokenizer::string_character_traits<char>::int_type go_to_next_word_pair(void) 
    {
        if (current_pair == NULL)
        {
            current_pair = head;
        }
        else 
        {
            current_pair = current_pair->next;
        }

        if (current_pair == NULL)
        {
            return cc_tokenizer::string_character_traits<char>::eof();
        }

        return ~cc_tokenizer::string_character_traits<char>::eof();
    }

    cc_tokenizer::string_character_traits<char>::size_type get_number_of_word_pairs(void)
    {
        return n;
    }

    WORDPAIRS_PTR get_current_word_pair(void) 
    {
        return current_pair;
    }

    /*
        Shuffling Word Pairs: Before iterating through word pairs. 
        It shuffles the pairs object (assumed to be an iterable containing word index pairs) using Numcy::Random::shuffle.
        This helps prevent the model from overfitting to the order of word pairs in the corpus.

        Both of the arguments originate at 0.
        The method swaps the values of two links with each other... 
        @a, zero originated link number
        @b, zero originated link number
     */
    void shuffle(cc_tokenizer::string_character_traits<char>::size_type a, cc_tokenizer::string_character_traits<char>::size_type b)
    {           
        cc_tokenizer::string_character_traits<char>::size_type i = 0;
        WORDPAIRS_PTR pair_a = NULL, pair_b = NULL;

        while (go_to_next_word_pair() != cc_tokenizer::string_character_traits<char>::eof())
        {            
            if (a == i)
            {
               pair_a = get_current_word_pair(); 
            }
            else if (b == i)
            {
                pair_b = get_current_word_pair();
            }

            if (pair_a != NULL && pair_b != NULL)
            {
                WORDPAIRS pair(*pair_a);
                *pair_a = *pair_b;
                *pair_b = pair;
                /*
                    We are breaking the loop with go_to_next_pair() abruptly(in the middle),
                    that is why we need to set the counter which keeps track of current word pair explicitly 
                 */
                current_pair = NULL;    
                break;          
            }

            i = i + 1;
        }
        
        
        /*
        if (head == NULL)
        {
            return;
        }

        cc_tokenizer::string_character_traits<char>::size_type i = 0;
        WORDPAIRS_PTR current_pair = head, pair_a = NULL, pair_b = NULL;
        
        do {
            if (a == i)
            {
               pair_a = current_pair; 
            }
            else if (b == i)
            {
                pair_b = current_pair;
            }

            if (pair_a != NULL && pair_b != NULL)
            {
                break;
            }

            i = i + 1;

            current_pair = current_pair->next;
            
        } while (current_pair != NULL);

        if (current_pair == NULL)
        {
            return;
        }
                         
        WORDPAIRS pair(*pair_a);
        *pair_a = *pair_b;
        *pair_b = pair;
         */        
    }

    private:
        WORDPAIRS_PTR head;
        WORDPAIRS_PTR current_pair;

        cc_tokenizer::string_character_traits<char>::size_type n;

} SKIPGRAMPAIRS;

typedef SKIPGRAMPAIRS* SKIPGRAMPAIRS_PTR;
typedef SKIPGRAMPAIRS& SKIPGRAMPAIRS_REF;
//typedef SkipGramPairs& SKIPGRAMPAIRS_REF;
typedef const SKIPGRAMPAIRS& SKIPGRAMPAIRS_CONST_REF;

#endif