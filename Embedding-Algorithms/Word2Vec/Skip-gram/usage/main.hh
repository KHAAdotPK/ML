/*
    usage/main.hh
    Q@khaa.pk
 */

#include <iostream>

#ifndef WORD_EMBEDDING_ALGORITHMS_SKIP_GRAM_USAGE_MAIN_HH
#define WORD_EMBEDDING_ALGORITHMS_SKIP_GRAM_USAGE_MAIN_HH

#define SKIP_GRAM_DEFAULT_CORPUS_FILE ".\\data\\corpus.txt"

#ifdef GRAMMAR_END_OF_TOKEN_MARKER
#undef GRAMMAR_END_OF_TOKEN_MARKER
#endif
#ifdef GRAMMAR_END_OF_LINE_MARKER
#undef GRAMMAR_END_OF_LINE_MARKER
#endif

#define GRAMMAR_END_OF_TOKEN_MARKER ' '
#define GRAMMAR_END_OF_LINE_MARKER '\n'

#include "../lib/argsv-cpp/lib/parser/parser.hh"
#include "../lib/sundry/cooked_read_new.hh"

#include "../ML/Embedding-Algorithms/Word2Vec/skip-gram/header.hh"

#define COMMAND "h -h help --help ? /? (Displays help screen)\nv -v version --version /v (Displays version number)\ne epoch --epoch /e (Sets epoch or number of times the training loop would run)\ncorpus --corpus (Path to the file which has the training data)\nverbose --verbose (Display of output, verbosly)\nlr --lr learningrate (Learning rate)\nrs -rs (Regularization strength)"

#endif