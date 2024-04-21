### Skip-gram Word Embedding Implementation in C++
---
__Description__
This repository implements the Skip-gram model for learning word embeddings from a text corpus. Skip-gram is a neural network architecture used to capture semantic relationships between words based on co-occurrence patterns. The learned word embeddings can be used for various natural language processing (NLP) tasks such as sentiment analysis, machine translation, and text classification.

__Features__
1. Implements the Skip-gram architecture with two weight matrices (embedding matrix and output layer)
2. Supports training on a text corpus
3. Uses negative sampling to improve training efficiency
4. (Optional) Add features you plan to implement in the future (e.g., different negative sampling methods, hyperparameter tuning)

__Dependencies__
1. C++ compiler with support for C++11 (e.g., g++ for GCC)
2. Numcy
3. Parser abstract class
3. CSV parser
4. String class
5. Command line argument parser
6. Sundry
7. Allocator class
8. Exception (ala_exception) class
9. Corpus class

__Getting Started__

*Clone the repository*
```BASH
```
*Clone dependecies*
```BASH
```
*Usage*
```BASH
```
1. Prepare your text corpus in a format suitable for the code (e.g., plain text file with one sentence per line).
2. Compile the code (instructions specific to your compiler).
3. Run the program with the following arguments (replace with actual argument descriptions):

```BASH
```

*Output*

The program will save the learned word embeddings to the specified output file (e.g., "word_embeddings.txt"). Each line in the file will represent a word and its corresponding embedding vector.

#### Further Development

This is a basic implementation of Skip-gram. Here are some potential areas for future development:

1. Explore different negative sampling methods
2. Implement hyperparameter tuning
3. Add support for different text pre-processing techniques
Integrate with other NLP tasks

#### Contributing
We welcome contributions to this project. Feel free to submit pull requests for bug fixes, improvements, and new features.

### License
This project is governed by a license, the details of which can be located in the accompanying file named 'LICENSE.' Please refer to this file for comprehensive information.