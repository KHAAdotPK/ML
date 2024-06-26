/*
    usage/main.hh
    Q@khaa.pk
 */

// For argsv-c++, HELP macro 
#include <iostream>

#ifndef EMBEDDING_ALGORITHM_SKIP_GRAM_USAGE 
#define EMBEDDING_ALGORITHM_SKIP_GRAM_USAGE

#define SKIP_GRAM_DEFAULT_CORPUS_FILE "data\\corpus.txt"

#undef GRAMMAR_END_OF_TOKEN_MARKER
#undef GRAMMAR_END_OF_LINE_MARKER

#define GRAMMAR_END_OF_TOKEN_MARKER ' '
#define GRAMMAR_END_OF_LINE_MARKER '\n'

/*
    Note: The delimiter used to separate the elements in the COMMAND macro can be customized.
    The first definition uses commas (,) as delimiters, while the second definition uses whitespace. 
    If you wish to change the delimiter or adjust its size, you can modify the corresponding settings in the file...
    lib/csv/parser.h or in your CMakeLists.txt.
    Alternatively, you can undefine and redefine the delimiter after including the lib/argsv-cpp/lib/parser/parser.hh 
    file according to your specific requirements.

    Please note that the macros mentioned below are by default or originally defined in the file lib/csv/parser.h
    #define GRAMMAR_END_OF_TOKEN_MARKER ","
    #define GRAMMAR_END_OF_TOKEN_MARKER_SIZE 1
    #define GRAMMAR_END_OF_LINE_MARKER "\n"
    #define GRAMMAR_END_OF_LINE_MARKER_SIZE 1

    The following two macros are defined in file  lib\argsv-cpp\lib\parser\parser.hh
    #define HELP_STR_START    "("
    #define HELP_STR_END      ")"
 */
/*
    To change the default parsing behaviour of the CSV parser
        
    Snippet from CMakeLists.txt file
    # Add a definition for the GRAMMAR_END_OF_TOKEN_MARKER macro
    #add_definitions(-DGRAMMAR_END_OF_TOKEN_MARKER=" ")
    #add_definitions(-DGRAMMAR_END_OF_TOKEN_MARKER_SIZE=1)

    Snippet from CMakeLists.txt file
    # Add a definition for the GRAMMAR_END_OF_TOKEN_MARKER macro for the replika target
    #target_compile_definitions(replika PRIVATE GRAMMAR_END_OF_TOKEN_MARKER=" ")
    #target_compile_definitions(replika PRIVATE GRAMMAR_END_OF_TOKEN_MARKER_SIZE=1)
 */
#define COMMAND "h -h help --help ? /? (Displays help screen)\nv -v version --version /v (Displays version number)\ne epoch --epoch /e (Sets epoch or number of times the training loop would run)\ncorpus --corpus (Path to the file which has the training data)\nverbose --verbose (Display of output, verbosly)\nlr --lr (Learning rate)"

//#include "../lib/argsv-cpp/lib/parser/parser.hh"
//#include "../lib/sundry/cooked_read.hh"
//#include "../lib/corpus/corpus.hh"
#include "../lib/WordEmbedding-Algorithm/Word2Vec/skip-gram/header.hh"

#endif