#### Example code, showing how to use skip gram pairs class.
```C++
class Corpus vocab(data_parser);
SKIPGRAMPAIRS pairs(vocab/*, arg_verbose.i ? true : false*/);

std::cout<< pairs.get_number_of_word_pairs()
```

```C++
class Corpus vocab(data_parser);
SKIPGRAMPAIRS pairs(vocab/*, arg_verbose.i ? true : false*/);

while (pairs.go_to_next_word_pair() != cc_tokenizer::string_character_traits<char>::eof())
{
    WORDPAIRS_PTR pair = pairs.get_current_word_pair();

    // vocab instance of CORPUS class, indices originate at INDEX_ORIGINATES_AT_VALUE
    // This means the first word in the vocabulary has an index of INDEX_ORIGINATES_AT_VALUE, the second word has an index of INDEX_ORIGINATES_AT_VALUE + 1, and so on.
    // Therefore, we don't need to subtract any offset value (INDEX_ORIGINATES_AT_VALUE) when accessing words in the vocabulary using their indices.    
    std::cout<< vocab[pair->getCenterWord() /* - INDEX_ORIGINATES_AT_VALUE */].c_str() << ", ";
}
```

```C++
ARG arg_corpus;
cc_tokenizer::csv_parser<cc_tokenizer::String<char>, char> argsv_parser(cc_tokenizer::String<char>(COMMAND));

FIND_ARG(argv, argc, argsv_parser, "data", arg_corpus);

cc_tokenizer::String<char> data;

try 
{
    data = cc_tokenizer::cooked_read<char>(argv[arg_corpus.i + 1]);
}
catch (ala_exception e)
{
    std::cout<<e.what()<<std::endl;
    return -1;
}

CORPUS vocab(data);
SKIPGRAMPAIRS pairs(vocab);

while (pairs.go_to_next_word_pair() != cc_tokenizer::string_character_traits<char>::eof())
{
    WORDPAIRS_PTR pair = pairs.get_current_word_pair();
            
    for (cc_tokenizer::string_character_traits<char>::size_type i = 0; i < SKIP_GRAM_WINDOW_SIZE; i++)
    {
        std::cout<< vocab[pair->getLeft()->array[(SKIP_GRAM_WINDOW_SIZE - 1) - i]].c_str() << " "; 
    }

    std::cout<< " [ " << vocab[pair->getCenterWord()].c_str() << " ] ";

    for (cc_tokenizer::string_character_traits<char>::size_type i = 0; i < SKIP_GRAM_WINDOW_SIZE; i++)
    {
        std::cout<< vocab[pair->getRight()->array[i]].c_str() << " ";                
    }

    std::cout<< std::endl;
}        
```

```C++

```