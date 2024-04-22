/*
    lib\ML\NLP\unsupervised\Word2Vec\CBOW\cbow.h
    Q@khaa.pk
*/

#include "../../../../../corpus/corpus.hh"
#include "../../../../../numc/numc.hh"

#ifndef CC_TOKENIZER_REPLIKA_PK_SKIP_GRAM_SKIP_GRAM_H_HH
#define CC_TOKENIZER_REPLIKA_PK_SKIP_GRAM_SKIP_GRAM_H_HH

/*
    An epoch refers to a complete pass through the training data during the training process.

    The number of epochs is an important hyperparameter in both CBOW and Skip-Gram models.
    It determines how many times the model will iterate over the entire training dataset.
    A larger number of epochs can potentially lead to better word embeddings as the model has more opportunities to learn from the data.
    However, setting the number of epochs too high may result in overfitting, 
    where the model becomes too specific to the training data and performs poorly on unseen data.

    The choice of the optimal number of epochs depends on various factors, such as the size of the training data,
    the complexity of the language being modeled, and the computational resources available.
    It is often determined through experimentation and validation on a held-out dataset to ensure the best performance of the word embeddings.
 */
#define CBOW_DEFAULT_EPOCH         100

/*
   The learning rate controls the step size at each iteration of the optimization process
 */
#define CBOW_DEFAULT_LEARNING_RATE 0.1

/*
    Technically, you can set the window size to 1 in the CBOW model, 
    but it would limit the amount of context information available for word prediction.
    With a window size of 1, only the immediately adjacent words to the target word would be considered as context words.
    This may result in a more limited understanding of word relationships and could potentially lead to less accurate word embeddings.

    A larger window size allows the model to capture more contextual information from a wider range of words surrounding the target word.
    This can often lead to better word embeddings, as the model has more context to learn from.
    However, a larger window size also increases the computational complexity and memory requirements of the model.
 */
#define CBOW_DEFAULT_WINDOW_SIZE 2

/*
    Number of neurons in the hidden layer and this represents the size of the hidden layer in the neural network.
    10 neurons is small size, suitable for small vocabulary.
    However, for larger vocabularies and more complex tasks, a larger hidden layer size may be required to capture more intricate relationships between the input and output. 
 */
#define CBOW_HIDDEN_LAYER_SIZE_IN_NUMBER_OF_NEURONS 10

template<typename E>
struct backward_propogation 
{
    /*
        Both arrays has shape which is (corpus::len(), REPLIKA_HIDDEN_SIZE) and (REPLIKA_HIDDEN_SIZE, corpus::len()) respectovely
     */
    E* grad_W1;
    E* grad_W2;    
};

template<typename E>
struct forward_propogation
{
    /*
        In the context of our CBOW model, h refers to the hidden layer vector obtained by averaging the embeddings of the context words.
        It is used in both the forward and backward passes of the neural network.

        The size of this array is (1, REPLIKA_HIDDEN_SIZE)
     */
    E* h;
    /*
        y_pred is a numc array of predicted probabilities of the output word given the input context. 
        In our implementation, it is the output of the forward propagation step.

        The size of this array is returned by "class corpus::len()" (1, copus::len())
     */
    E* y_pred;
};

struct context_target_link
{
    /*
        While parsing the corpus into context and target pair, 
        please make sure that context and target do not have a same value 
     */
    corpus::corpus_index_type context;
    corpus::corpus_index_type target;

    struct context_target_link* next;
    struct context_target_link* prev;    
};

struct context_target_list 
{
    corpus::corpus_index_type n;
    struct context_target_link* ptr;

    void traverse(void)
    {
        std::cout<<"Number of context and target lines = "<<n<<std::endl;

        if (!(n > 0))
        {
            return;
        }

        struct context_target_link link = *ptr;
        corpus::corpus_index_type i = 0, target = link.target;
        
        while (1)
        {            
            if (target != link.target)
            {
                std::cout<<" ----> "<<target;

                target = link.target;
                
                std::cout<<std::endl;
            }

            std::cout<<link.context<<" ";

            if (link.next == NULL)
            {
                std::cout<<" ----> "<<target;    

                break;
            }

            link = *link.next;
            
            i = i + 1;   
        }
    }

    corpus::corpus_index_type* get_context(corpus::corpus_index_type ln)
    {       
        cc_tokenizer::String<char>::size_type n = get_number_of_context_words(ln);

        if (n == 0)
        {
            return NULL;
        }

        cc_tokenizer::allocator<char> alloc_obj;
        struct context_target_link link = get_ctl(ln);

        corpus::corpus_index_type* context = reinterpret_cast<corpus::corpus_index_type*>(alloc_obj.allocate(n*sizeof(corpus::corpus_index_type)));

        for (cc_tokenizer::String<char>::size_type i = 0; i < n; i++)
        {
            context[i] = link.context;
            link = *link.next;
        }

        return context;
    }

    /* 
       The function get_ctl that retrieves a context_target_link struct from a linked list 
       based on the given index ln, returning a default struct if the index is out of bounds 
       or the list is empty
       @ln, link number is an index  into the linked list and it originates at 1  
     */    
    struct context_target_link get_ctl(corpus::corpus_index_type ln)
    {
        if (ptr == NULL || ln == 0)
        {
            return {0, 0, NULL, NULL};
        }

        struct context_target_link* ctl = NULL;
        
        do
        {
            if (ctl == NULL)
            {
                ctl = ptr;
            }
            else
            {
                ctl = ctl->next;
            }

            ln = ln - 1;
        }
        while (ln != 0);

        if (ln > 0)
        {
            return {0, 0, NULL, NULL};
        }
        else
        {        
            return *ctl;
        }
    }

    /*
        The function calculates the number of context words in a linked list.
        It starts by retrieving a link based on the given link number.
        If the link's context and target are the same, it returns 0.
        Otherwise, it iterates through the linked list,
        counting the number of linked elements until it encounters a different target or reaches the end.
        Finally, it returns the count of context words.
        @ln, link number is an index into the linked list and it originates at 1
     */
    cc_tokenizer::String<char>::size_type get_number_of_context_words(corpus::corpus_index_type ln)
    {        
        struct context_target_link link = get_ctl(ln);

        if (link.context == link.target)
        {
            return 0;
        } 

        // We found the link...
        cc_tokenizer::String<char>::size_type ret = 0;

        corpus::corpus_index_type target = link.target;

        while (1)
        {
            ret = ret + 1;

            if (link.next == NULL)
            {
                break;
            }

            link = *link.next;

            if (link.target != target)
            {
                break;
            }
        }

        return ret;
    }

    /*
        The function calculates the number of context words between a given context and target word.
        It iterates through a linked list of context words until it encounters a different target word or reaches the end of the list,
        counting the number of context words encountered.
        The final count is returned as the result.

        @link, instance of struct context_target_link
     */
    cc_tokenizer::String<char>::size_type get_number_of_context_words(context_target_link link)
    {
        if (link.context == link.target)
        {
            return 0;
        } 

        // We found the link...
        cc_tokenizer::String<char>::size_type ret = 0;

        corpus::corpus_index_type target = link.target;

        while (1)
        {
            ret = ret + 1;

            if (link.next == NULL)
            {
                break;
            }

            link = *link.next;

            if (link.target != target)
            {
                break;
            }
        }

        return ret;
    }

    corpus::corpus_index_type* get_context(struct context_target_link &link)
    {


        return NULL;
    }

    /*
        Get all three in one call
        @ln, 
     */
    void get_link_target_and_context(corpus::corpus_index_type ln)
    {
        struct context_target_link link = get_ctl(ln);

        if (link.context == link.target)
        {
            return;
        }

        corpus::corpus_index_type* context = get_context(ln);
    }          
};

/*
    @a, array of type @tp and of size @r rows and @c columns
    @b, array of type @tp and size of @n rows and @c columns
    @r, number of rows in @a
    @c, number of columns in array @a and @b
    @v, row vector has @n number of columns        
    @tp, type of each element of aray @a, @b row vector @v
 */
#define SUM_OF_TWO_MATRICES_OF_DIFFERENT_NUMBER_OF_ROWS(a, b, r, c, v, n, tp) {\
                                                                                    for (int i = 0; i < n; i++)\
                                                                                    {\
                                                                                        /*std::cout<<i<<"--> ";*/\
                                                                                        /*for (int j = 0; j < c; j++)*/\
                                                                                        {\
                                                                                            /*std::cout<<b[i*c + j]<<" ";*/\
                                                                                        }\
                                                                                        /*std::cout<<std::endl;*/\
                                                                                        /*std::cout<<v[i] - REPLIKA_PK_INDEX_ORIGINATES_AT<<"--> ";*/\
                                                                                        for (int j = 0; j < c; j++)\
                                                                                        {\
                                                                                            /*std::cout<<a[(v[i] - REPLIKA_PK_INDEX_ORIGINATES_AT)*c + j]<<" ";*/\
                                                                                            a[(v[i] - REPLIKA_PK_INDEX_ORIGINATES_AT)*c + j] += b[i*c + j];\
                                                                                            /*std::cout<<a[(v[i] - REPLIKA_PK_INDEX_ORIGINATES_AT)*c + j]<<" ";*/\
                                                                                        }\
                                                                                        /*std::cout<<std::endl<<" ------ -- -- - -- - --- -- - -- - - -- - -- - -- - "<<std::endl;*/\
                                                                                    }\
                                                                              }\


/*
    @a, array of type @tp and size @n
    @s, divisor of @tparam
    @n, size of array @a of type @tparam
    @p, pointer to @tp, the new array of the same size as the @a. Each value of @a is divided by @s 
    @tp, type of divisor @s and dividend @a. The @s and @a has same type which is @tp
 */
#define DIVIDE_ARRAY_BY_SCALAR(a, s, n, p, tp) {\
                                                cc_tokenizer::allocator<char> alloc_obj;\
                                                p = reinterpret_cast<tp*>(alloc_obj.allocate(sizeof(tp)*n));\
                                                for(cc_tokenizer::string_character_traits<char>::size_type i = 0; i < n; i++)\
                                                {\
                                                    p[i] = a[i]/s;\
                                                }\
                                               }\


/*
   @tp, type of p
   @p, subarray pointer
   @n, sizeof(*p)
   @w, weight ndarray
   @wr, rows of w
   @wc, columns of w
   @c, context array
   @cc, c columns
 */
#define  SUBARRAY(tp, p, w, wr, wc, c, cc)    {\
                                                  cc_tokenizer::allocator<char> alloc_obj_local;\
                                                  p = reinterpret_cast<tp(*)[wc]>(alloc_obj_local.allocate(sizeof(**p)*cc*wc));\
                                                  for (cc_tokenizer::string_character_traits<char>::size_type i = 0; i < cc; i++)\
                                                  {\
                                                      /*std::cout<<c[i]<<" ";*/\
                                                      for (cc_tokenizer::string_character_traits<char>::size_type j = 0; j < wc; j++)\
                                                      {\
                                                        /*std::cout<<w[c[i]][j]<<" ";*/\
                                                        /* Each word of vocabulary is assigned an index. The index value begins at REPLIKA_PK_INDEX_ORIGINATES_AT */\
                                                        /* It can be any value other than 0 as well. */\
                                                        /* The w array has embeddings of the vocabulary words, it is a two dimentional array and in which each line is a single word embedding for a single vocabulary word.*/\
                                                        /* The is why we need to deduct we need to deduct REPLIKA_PK_INDEX_ORIGINATES_AT */\
                                                        p[i][j] = w[c[i] - REPLIKA_PK_INDEX_ORIGINATES_AT][j];\
                                                      }\
                                                      /*std::cout<<std::endl;*/\
                                                  }\
                                              }\

/*
  @a, array of type @tp and of size @n
  @s, scalar value of type @tp. This value gets substracted from each element of @a
  @n, number of elements in an array @a
  @p, pointer to @tp, the new array of the same size as the @a. Each value of @a is reduced by @s 
  @tp, type of each element of aray @a and type of scalar @s

  On completion this macro would return a new array of type @tp with number of elements equals to @n
 */
#define SUBTRACT_SCALAR(a, s, n, p, tp) {\
                                          cc_tokenizer::allocator<char> alloc_obj_local;\
                                          p = reinterpret_cast<tp*>(alloc_obj_local.allocate(sizeof(tp)*n));\
                                          for (cc_tokenizer::string_character_traits<char>::size_type i = 0; i < n; i++)\
                                          {\
                                            p[i] = a[i] - s;\
                                          }\
                                        }\

/*
  @a, array of size @n
  @n, number of elements in an array @a  
  @s, scalar @s is placeholder for the calculated sum
 */
#define ARRAY_SUM(a, n, s) {\
                              for (cc_tokenizer::string_character_traits<char>::size_type i = 0; i < n; i++)\
                              {\
                                  s += a[i];\
                              }\
                           }\


template<typename E>
E* one_hot(unsigned int y_true, size_t vocab_len, class numc& numc_obj) 
{
    E* ptr = numc_obj.zeros<E>(1, vocab_len);

    if (ptr != NULL)
    {
        ptr[y_true] = 1;
    }

    return ptr;
}

/*
  TODO, details about how does it work
   u = h * W2 
   h is a row vector with shape (1, REPLIKA_HIDDEN_SIZE) 
   W2 is vector and has shape (REPLIKA_HIDDEN_SIZE, corpus::len())
   u is a row vector as well and has shape (1, corpus::len())
 */
template<typename E>
E* softmax(E* u, class corpus& corpus_obj, class numc& numc_obj)
{   
    cc_tokenizer::allocator<char> alloc_obj;

    E* u_minus_max = NULL; // Stores all elements of a u, minus an scalar.
    //SUBTRACT_SCALAR(u, numc_obj.max<double>(u, REPLIKA_HIDDEN_SIZE), REPLIKA_HIDDEN_SIZE, u_minus_max, E);
    SUBTRACT_SCALAR(u, numc_obj.max<double>(u, corpus_obj.len()), corpus_obj.len(), u_minus_max, E);

    //E* e_u_minus_max = numc_obj.exp<double>(u_minus_max, REPLIKA_HIDDEN_SIZE);
    /*
        Each element of array is transformed into a base^power form
     */
    E* e_u_minus_max = numc_obj.exp<double>(u_minus_max, corpus_obj.len());

    E e_u_minus_max_sum = 0; 
    //ARRAY_SUM(e_u_minus_max, REPLIKA_HIDDEN_SIZE, e_u_minus_max_sum);
    ARRAY_SUM(e_u_minus_max, corpus_obj.len(), e_u_minus_max_sum);

    double* e_u_minus_max_divided_by_e_u_minus_max_sum;
    //DIVIDE_ARRAY_BY_SCALAR(e_u_minus_max, e_u_minus_max_sum, REPLIKA_HIDDEN_SIZE, e_u_minus_max_divided_by_e_u_minus_max_sum, double);
    DIVIDE_ARRAY_BY_SCALAR(e_u_minus_max, e_u_minus_max_sum, corpus_obj.len(), e_u_minus_max_divided_by_e_u_minus_max_sum, double);

    alloc_obj.deallocate(reinterpret_cast<char*>(u_minus_max));
    alloc_obj.deallocate(reinterpret_cast<char*>(e_u_minus_max));
    
    /*
       array of exp(u - max)/sum_of(exp(u - max)) and shape of this array is (1, REPLIKA_VOCABULARY_LENGTH )
     */
    return e_u_minus_max_divided_by_e_u_minus_max_sum;
}

/* **************************************************************************************** */

struct forward_propogation<double> forward(unsigned int* context, double (*W1)[CBOW_HIDDEN_LAYER_SIZE_IN_NUMBER_OF_NEURONS], double* W2, class corpus& corpus_obj, class numc& numc_obj)
{
    cc_tokenizer::allocator<char> alloc_obj;
    /*
        It is a two dimensional array... 
        number of lines of this array is equal to the reqturned value of this funtion call corpus_obj.get_size_of_context_line()
        The shape of this array (corpus_obj.get_size_of_context_line(), REPLIKA_HIDDEN_SIZE)
     */
    double (*w1_subarray)[CBOW_HIDDEN_LAYER_SIZE_IN_NUMBER_OF_NEURONS];

    /*
        This macro makes/returns part of an array. A line of context array has indices of the word sarrounding a target word. W1(weight matix) has 
        a weight of each word of a vocabulary. This macro makes a arrray out of each line of W1 matrix which matches the index of each word of context line.  
     */    
    SUBARRAY(double, w1_subarray, W1, corpus_obj.len(), CBOW_HIDDEN_LAYER_SIZE_IN_NUMBER_OF_NEURONS, context, corpus_obj.get_size_of_context_line());
    
    /*
        In the CBOW model code we have not used any activation function.
        The hidden layer vector h is simply a weighted sum of the input word vectors. However, we can choose to apply an activation function such as 
        ReLU or tanh on this vector if needed.
     */
    /*
        Returned array is single dimension array, size of returned array for REPLIKA_PK_NUMC_YAXIS is REPLIKA_HIDDEN_SIZE
        Small h is for hidden, h refers to the hidden layer vector obtained by averaging the embeddings of the context words.
        It is used in both the forward and backward passes of the neural network.

        Each context word array has its own h value... h has a shape (1, REPLIKA_HIDDEN_SIZE)
     */
    double* h = numc_obj.mean(reinterpret_cast<double*>(w1_subarray), {CBOW_HIDDEN_LAYER_SIZE_IN_NUMBER_OF_NEURONS, corpus_obj.get_size_of_context_line(), NULL, NULL}, /*REPLIKA_PK_NUMC_MEAN_AXIS::*/REPLIKA_PK_NUMC_YAXIS);
    /*
        Returned array is single dimension array, size of returned array for REPLIKA_PK_NUMC_YAXIS is corpus_obj.len() and this is due to multiplication of of two vectors 
        More precisly, the returned array has number of lines equals to the rows of "matrix one" and the returned array has number of columns equal to "matrix two"
        In our following case, u has this shape (1, corpus_obj.len() = REPLIKA_VOCABULARY_LENGTH)

        The u is a row vector, it is a product of h and W2
     */
    double* u = numc_obj.dot(h, reinterpret_cast<double*>(W2), {CBOW_HIDDEN_LAYER_SIZE_IN_NUMBER_OF_NEURONS, 1, NULL, NULL}, {corpus_obj.len(), CBOW_HIDDEN_LAYER_SIZE_IN_NUMBER_OF_NEURONS, NULL, NULL}, REPLIKA_PK_NUMC_YAXIS);

    /*
        The context words are first converted to one-hot vectors and then averaged to produce a single input vector
        This input vector is then passed through a linear layer (with weights W1) and a nonlinear activation function (such as ReLU or tanh) to produce a hidden representation.
        Finally, this hidden representation is passed through a second linear layer (with weights W2) to produce the output prediction.
     */
    /*        
        W1_subarray = W1[context] shape of W1_subarray is (REPLIKA_CONTEXT_LINE_SIZE, REPLIKA_HIDDEN_SIZE)
        h a row vector with shape (1, REPLIKA_HIDDEN_SIZE), value at each index is an average of the a column at the same index in W1_subarray
        u = h * W2 and has shape (1, REPLIKA_VOABULARY_LENGTH) 
     */

    /*
        The y_pred is the predicted output vector of the model, which is calculated during the forward pass of the neural network.
        In the context of Word2Vec, y_pred is the output vector that represents the target word in the context of the input words.
        The y_pred is a single dimension array, the size of the line of this array(number of columns) is corpus_obj.len()

        y_pred is e_u_minus_max_divided_by_e_u_minus_max_sum
     */
    double* y_pred = softmax<double>(u, corpus_obj, numc_obj);
    
    alloc_obj.deallocate(reinterpret_cast<char*>(u));
    alloc_obj.deallocate(reinterpret_cast<char*>(w1_subarray));
    
    /*
        h is row vector, (1, REPLIKA_HIDDEN_SIZE)  
        y_pred is a row vector as well, (1, REPLIKA_VOCABULARY_LENGTH)
        y_pred is e_u_minus_max_divided_by_e_u_minus_max_sum
     */
    /*
        y_pred is the predicted output vector of the model, which is calculated during the forward pass of the neural network.
        In the context of Word2Vec, y_pred is the output vector that represents the target word in the context of the input words.
    */
    return {h, y_pred};
}

/* **************************************************************************************** */ 
/*
    Backward propagation (also known as back propagation) refers to the process of computing the gradients of the model's parameters 
    with respect to a loss function, in order to update the parameters during training.
 */
/*
    @context, the surrounding words that are used to predict a target word
    @fpg, values returned by fuction forward propagation 
          fpg.h = h refers to the hidden layer vector obtained by averaging the embeddings of the context words
          fpg.y_pred = y_pred is the predicted output vector of the model, y_pred is the output vector that represents the target word in the context of the input words
    @y_true, the true label of the center word in the context window. True output relative to predicted(y_pred) output
             The goal of the model(CBOW) is to predict this center word given the surrounding words in the context window                
 */
struct backward_propogation<double> backward(unsigned int* context, struct forward_propogation<double>& fpg, unsigned int y_true, double (*W1)[CBOW_HIDDEN_LAYER_SIZE_IN_NUMBER_OF_NEURONS], double* W2, class corpus& corpus_obj, class numc& numc_obj)
{
    cc_tokenizer::allocator<char> alloc_obj;

    /*
        The true distribution(represented as a one-hot encoding of the target word)
        The hot is row vector, and has shape (1, corpus_obj.len = REPLIKA_VOCABULARY_LENGTH)

        y_true, index in to vocabularoy of the target word a.k.a center word, donot expect them in the series, like 0 1 2 3 4 5....
     */
    double* hot = one_hot<double>(y_true - REPLIKA_PK_INDEX_ORIGINATES_AT, corpus_obj.len(), numc_obj);    
    /*
        Here, "mumc::subtract_matrix() is the loss function with respect to the output vector u" in Word2Vec is typically the cross-entropy loss,
        which measures the difference between the predicted probability distribution(fpg.y_pred) over the vocabulary and the true 
        distribution(represented as a one-hot encoding of the target word).

        The cross-entropy loss is commonly used in multi-class classification problems
        The grad_u(the returned value of this fuction call) is the gradient of the loss function with respect to the output vector u.
        It is calculated as the difference between the predicted output vector y_pred and the one-hot encoded target vector y_true.

        The shape of row vector hot is (1, corpus_obj.len = REPLIKA_VOCABULARY_LENGTH), likewise the shape of grad_u is the same as y_pred which is (1, len(vocab))        
        The shape of grad_u is the same as y_pred which is (1, len(vocab))
        All three vectors involved are single row vectors
     */
    double* grad_u = numc_obj.subtract_matrices<double>(fpg.y_pred, hot, corpus_obj.len());
    /*
        Shape of W2_T is (REPLIKA_VOCABULARY_LENGTH, REPLIKA_HIDDEN_SIZE), the matrix W2 has shape of (REPLIKA_HIDDEN_SIZE, REPLIKA_VOCABULARY_LENGTH)
     */
    double* W2_T = numc_obj.transpose_matrix<double>(reinterpret_cast<double*>(W2), CBOW_HIDDEN_LAYER_SIZE_IN_NUMBER_OF_NEURONS, corpus_obj.len());

    /*
        H is the number of neurons in the hidden layer, where V is the vocabulary size. 
        The grad_u has a shape of (1, V) and W2.T has a shape of (V, H)
        The resulting matrix grad_h would have a shape of (1, H) or (1, REPLIKA_HIDDEN_SIZE)
     */
    double* grad_h = numc_obj.dot(grad_u, W2_T, {corpus_obj.len(), 1, NULL, NULL}, {CBOW_HIDDEN_LAYER_SIZE_IN_NUMBER_OF_NEURONS, corpus_obj.len(), NULL, NULL}, REPLIKA_PK_NUMC_YAXIS);
    
    /*
        The grad_W2 has shape/dimensions of (REPLIKA_HIDDEN_SIZE, REPLIKA_VOCABULARY_LENGTH)
        The dimensions/shape of fpg.h is (1, REPLIKA_HIDDEN_SIZE) and the dimensions/shape of grad_u is (1, len(vocab) = corpus_obj.len() = REPLIKA_VOCABULARY_LENGTH)
     */    
    double* grad_W2 = numc_obj.outer<double>(fpg.h, grad_u, CBOW_HIDDEN_LAYER_SIZE_IN_NUMBER_OF_NEURONS, corpus_obj.len());

    /*
        The shape/dimensions of grad_W1 is (REPLIKA_VOCABULARY_LENGTH, REPLIKA_HIDDEN_SIZE)
     */
    double* grad_W1 = numc_obj.zeros<double>(2, corpus_obj.len(), CBOW_HIDDEN_LAYER_SIZE_IN_NUMBER_OF_NEURONS);
    
    /*
        The ones in place of context indexes, shape of nes_in_place_of_context_indexes is (1, REPLIKA_HIDDEN_SIZE)
     */
    double* ones_in_place_of_context_indexes = numc_obj.ones<double>(1, corpus_obj.get_size_of_context_line());
    /*
        The shape of matrix(outer_of_grad_h_and_ones_in_place_of_context_indexes) is (REPLIKA_HIDDEN_SIZE, corpus_obj.get_size_of_context_line())
     */
    /*
        The shape of grad_h (1, REPLIKA_HIDDEN_SIZE), the shape of ones_in_place_of_context_indexes is (1, corpus_obj.get_size_of_context_line())
        The outer of grad_h and ones in place of context indexes and the shape of outer_of_grad_h_and_ones_in_place_of_context_indexes (REPLIKA_HIDDEN_SIZE, corpus_obj.get_size_of_context_line())
     */
    double* outer_of_grad_h_and_ones_in_place_of_context_indexes = numc_obj.outer<double>(grad_h, ones_in_place_of_context_indexes, CBOW_HIDDEN_LAYER_SIZE_IN_NUMBER_OF_NEURONS, corpus_obj.get_size_of_context_line());
    /*
        The transpose of outer_of_grad_h_and_ones_in_place_of_context_indexes and it has shape of (corpus_obj.get_size_of_context_line(), REPLIKA_HIDDEN_SIZE)
     */
    double* transpose_of_outer_of_grad_h_and_ones_in_place_of_context_indexes = numc_obj.transpose_matrix<double>(outer_of_grad_h_and_ones_in_place_of_context_indexes, CBOW_HIDDEN_LAYER_SIZE_IN_NUMBER_OF_NEURONS, corpus_obj.get_size_of_context_line());
    
    /*
        It is a two dimensional array... the grad_W1_subarray
        number of lines of this array is equal to the returned value of this funtion call corpus_obj.get_size_of_context_line() and columns equal to REPLIKA_HIDDEN_SIZE
     */
    double (*grad_W1_subarray)[CBOW_HIDDEN_LAYER_SIZE_IN_NUMBER_OF_NEURONS];
    SUBARRAY(double, grad_W1_subarray, reinterpret_cast<double(*)[CBOW_HIDDEN_LAYER_SIZE_IN_NUMBER_OF_NEURONS]>(grad_W1), corpus_obj.len(), CBOW_HIDDEN_LAYER_SIZE_IN_NUMBER_OF_NEURONS, context, corpus_obj.get_size_of_context_line());

    /*
    for (int i = 0; i < corpus_obj.get_size_of_context_line(); i++)
    {
        for (int j = 0; j < REPLIKA_HIDDEN_SIZE; j++)
        {
            std::cout<<grad_W1_subarray[i][j]<<" ";
        }

        std::cout<<std::endl;
    }
    std::cout<<" ------- ------ ------- ------ ------- ----- --- ------- ---- --- --------- --- --- "<<std::endl;
     */

    /*
        grad_W1_subarray has shape (corpus_obj.get_size_of_context_line(), REPLIKA_HIDDEN_SIZE)
        grad_W1[context] 

        transpose_of_outer_of_grad_h_and_ones_in_place_of_context_indexes has shape (corpus_obj.get_size_of_context_line(), REPLIKA_HIDDEN_SIZE)
        np.outer(grad_h, np.ones((len(context),))).T
     */

    SUM_OF_TWO_MATRICES_OF_DIFFERENT_NUMBER_OF_ROWS(grad_W1, transpose_of_outer_of_grad_h_and_ones_in_place_of_context_indexes, corpus_obj.len(), CBOW_HIDDEN_LAYER_SIZE_IN_NUMBER_OF_NEURONS, context, corpus_obj.get_size_of_context_line(), double);
    /*
    for (cc_tokenizer::String<char>::size_type i = 0; i < corpus_obj.len(); i++)
    {
        for (cc_tokenizer::String<char>::size_type j = 0; j < REPLIKA_HIDDEN_SIZE; j++)
        {
            std::cout<<grad_W1[i*REPLIKA_HIDDEN_SIZE + j]<<" ";
        }
        std::cout<<std::endl;
    }
     */
                
    alloc_obj.deallocate(reinterpret_cast<char*>(grad_h));    
    alloc_obj.deallocate(reinterpret_cast<char*>(grad_u));
    //alloc_obj.deallocate(reinterpret_cast<char*>(grad_W1));
    alloc_obj.deallocate(reinterpret_cast<char*>(grad_W1_subarray));
    //alloc_obj.deallocate(reinterpret_cast<char*>(grad_W2));
    alloc_obj.deallocate(reinterpret_cast<char*>(hot));
    alloc_obj.deallocate(reinterpret_cast<char*>(ones_in_place_of_context_indexes));
    alloc_obj.deallocate(reinterpret_cast<char*>(outer_of_grad_h_and_ones_in_place_of_context_indexes));

    return {grad_W1, grad_W2};
}
/* **************************************************************************************** */ 

/*      
    In the given code block, the variable "window_size" is used to determine the size of the context window around the target word.
    The context for a target word consists of the words that appear within the window surrounding it.

    When iterating through the words in a sentence, the word at index "i" will be the target word,
    and the context will be formed by selecting "window_size" number of words to the left and "window_size" number of words to the 
    right of the target word. Therefore, the starting index for the context is "i - window_size," and the ending index for the context
    is "i + window_size + 1."

    However, since the target word itself should not be included in the context, we need to exclude it.
    To achieve this, we use the condition "if j != i" to exclude the target word from the context.
    Thus, the context list is constructed by iterating over "j" in the range from "i - window_size" to "i + window_size + 1,"
    excluding the "ith" index.

    That's why we subtract "window_size" from "i" when iterating over the range to obtain the starting index of the context,
    and we add "window_size + 1" to "i" when iterating over the range to obtain the ending index of the context.

    The process.
    --------------
    The "context window" or "sliding window" method in natural language processing, specifically 
    for generating context and target word pairs in models like Continuous Bag-of-Words (CBOW) or Skip-gram.
    In this process, the text corpus is traversed, and at each position, a window of fixed size is moved across the text.
    The center word or words in the window are considered as the target word(s), and the surrounding words within the
    window are treated as the context words. The window slides through the text, generating multiple context and target word pairs.
    
    @dp, data parser, instance of data parser class
    @vocab,  vocabulary, instance of corpus class      
    @ws, window size
    @cth, context target head, an instance of "struct context_target_list"
    @verbose, macro does'nt stay silent anf put more words on the screen when verbose is true 
 */
#define CORPUS_TO_CONTEXTS_AND_TARTGETS(dp, vocab, ws, cth, verbose)  {\
                                                        dp.reset(LINES);\
                                                        dp.reset(TOKENS);\
                                                        cc_tokenizer::allocator<char> alloc_obj;\
                                                        struct context_target_link* ctl = NULL; /* context target link */\
                                                        while(dp.go_to_next_line() != cc_tokenizer::string_character_traits<char>::eof())\
                                                        {\
                                                            /* Why is this if statement here? The parser is functioning correctly. */\
                                                            /* If 'corpus::corpus_index_type' had been a signed type, it would have worked, and I wouldn't have needed to include that if statement. */\
                                                            /* If there had been no empty line at the end of the corpus, there would have been no need for that if statement. */\
                                                            /* Okay, I admit that in the case of an empty line in the corpus, the parser treats it as a line with one token. */\
                                                            /* This is incorrect. When I have time, I should work on the parser code and eliminate this behavior.*/\
                                                            if (dp.get_total_number_of_tokens() >= ws)\
                                                            {\
                                                                for (corpus::corpus_index_type i = ws; i < (dp.get_total_number_of_tokens() - ws); i++)\
                                                                {\
                                                                    vocab.increase_number_of_target_lines_in_corpus();\
                                                                    vocab.increase_number_of_context_lines_in_corpus();\
                                                                    for (corpus::corpus_index_type j = i - ws; j < (i + ws + 1); j++)\
                                                                    {\
                                                                        if (i != j)\
                                                                        {\
                                                                            if (verbose)\
                                                                            {\
                                                                                std::cout<<dp.get_token_by_number(j + 1).c_str();\
                                                                                std::cout<<" -- "<<dp.get_token_by_number(i + 1).c_str()<<std::endl;\
                                                                            }\
                                                                            if (cth.ptr == NULL)\
                                                                            {\
                                                                                ctl = reinterpret_cast<struct context_target_link*>(alloc_obj.allocate(sizeof(struct context_target_link)));\
                                                                                ctl->next = NULL;\
                                                                                ctl->prev = NULL;\
                                                                                cth.n = 1;\
                                                                                cth.ptr = ctl;\
                                                                            }\
                                                                            else\
                                                                            {\
                                                                                ctl->next = reinterpret_cast<struct context_target_link*>(alloc_obj.allocate(sizeof(struct context_target_link)));\
                                                                                ctl->next->prev = ctl;\
                                                                                ctl->next->next = NULL;\
                                                                                ctl = ctl->next;\
                                                                                cth.n = cth.n + 1;\
                                                                            }\
                                                                            ctl->context = vocab[dp.get_token_by_number(j + 1)]->index;\
                                                                            ctl->target = vocab[dp.get_token_by_number(i + 1)]->index;\
                                                                        }\
                                                                    }\
                                                                }\
                                                            }\
                                                            else\
                                                            {\
                                                                if (verbose)\
                                                                {\
                                                                    std::cout<<"Number of tokens are less than window size for line number "<<dp.get_current_line_number();\
                                                                    std::cout<<". Number of tokens are "<<dp.get_total_number_of_tokens()<<" and window size is "<<ws<<std::endl;\
                                                                }\
                                                            }\
                                                        }\
                                                        vocab.set_number_of_columns_in_context(cth.n);\
                                                    }\

/*
    @e, epoch the number of complete pass through the training data
    @cth, context target head
    @w1, 
    @w2,
 */
#define CBOW_TRAINING_LOOP(e, cth, w1, w2, vcoab) {\
                                                        cc_tokenizer::allocator<char> alloc_obj;\
                                                        corpus::corpus_index_type* context = NULL;\
                                                        corpus::corpus_index_type y_pred = 0;\
                                                        unsigned long k_maxIterations = std::numeric_limits<unsigned long>::max();\
                                                        unsigned long k_iterations = std::min(e, k_maxIterations);\
                                                        corpus::corpus_index_type i_maxIterations = std::numeric_limits<corpus::corpus_index_type>::max();\
                                                        corpus::corpus_index_type i_iterations = std::min(cth.n, i_maxIterations);\
                                                        DIMENSIONS dim_context;\
                                                        for (unsigned long k = 0; k < k_iterations; k++)\
                                                        {\
                                                            corpus::corpus_index_type i = 0;\
                                                            while (i < i_iterations)\
                                                            {\
                                                                struct context_target_link link = cth.get_ctl(i + 1);\
                                                                if (link.target != link.context)\
                                                                {\
                                                                    cc_tokenizer::String<char>::size_type n = cth.get_number_of_context_words(i + 1);\
                                                                    context = cth.get_context(i + 1);\
                                                                    /*std::cout<<"n = "<<n<<std::endl;*/\
                                                                    /*std::cout<<"Target = "<<link.target<<std::endl;*/\
                                                                    if (context != NULL)\
                                                                    {\
                                                                        for (cc_tokenizer::String<char>::size_type j = 0; j < n; j++)\
                                                                        {\
                                                                            /*std::cout<<context[j]<<" ";*/\
                                                                        }\
                                                                        /*std::cout<<std::endl;*/\
                                                                        alloc_obj.deallocate(reinterpret_cast<char*>(context));\
                                                                        context = NULL;\
                                                                        /*std::cout<<"---UFO----"<<std::endl;*/\
                                                                    }\
                                                                    i = i + n;\
                                                                }\
                                                                else\
                                                                {\
                                                                    i++;\
                                                                }\
                                                            }\
                                                            std::cout<<"--------------------------------------------------------------"<<std::endl;\
                                                        }\
                                                        std::cout<<"HOL HOLA"<<std::endl;\
                                                  }\

#define GET_NEXT_TARGET_AND_ITS_CONTEXT_DIMENSIONS(cth, i, context, dim_context, alloc_obj, y_pred)\
                                                                        {\
                                                                            dim_context = {0, 0, NULL, NULL};\
                                                                            if (cth.ptr != NULL && i <= cth.n)\
                                                                            {\
                                                                                dim_context.columns = 1;\
                                                                                struct context_target_link link = cth.get_ctl(dim_context.columns + i);\
                                                                                corpus::corpus_index_type target = link.target;\
                                                                                y_pred = target;\
                                                                                while (target == link.target)\
                                                                                {\
                                                                                    /* Use old link here */\
                                                                                    dim_context.columns = dim_context.columns + 1;\
                                                                                    link = cth.get_ctl(dim_context.columns + i);\
                                                                                }\
                                                                                /* dim_context.columns is now atlesase 1 */\
                                                                                if (context == NULL)\
                                                                                {\
                                                                                    context = reinterpret_cast<corpus::corpus_index_type*>(alloc_obj.allocate(sizeof(corpus::corpus_index_type)*dim_context.columns));\
                                                                                }\
                                                                                link = cth.get_ctl(i + 1);\
                                                                                corpus::corpus_index_type j = 0;\
                                                                                while (/*target == link.target*/ j < dim_context.columns)\
                                                                                {\
                                                                                    context[j] = /*link.context;*/ cth.get_ctl(j + i + 1).context;\
                                                                                    /*std::cout<<context[j]<<" ";*/\
                                                                                    link = /**link.next;*/cth.get_ctl(j + i + 1);\
                                                                                    /*j++;*/\
                                                                                    /*link = cth.get_ctl(i + j + 1);*/\
                                                                                    j++;\
                                                                                }\
                                                                            }\
                                                                        }\

/*
    PLEASE NOTE:- Use this macro in macro CBOW_TRAINING_LOOP macro
    ----------------------------------------------------------------
    @cth, context target head
    @ctln, context target link number    
 */
#define NEXT_TARGET_AND_ITS_CONTEXT(cth, ctln, vocab, context, dim_context, y_pred) {\
                                                                    /*cc_tokenizer::allocator<char> alloc_obj;*/\
                                                                    \
                                                                    if (cth.ptr != NULL && ctln <= cth.n)\
                                                                    {\
                                                                        corpus::corpus_index_type j = 0;\
                                                                        corpus::corpus_index_type target = 0;\
                                                                        struct context_target_link* ctl = cth.ptr, *head_ctl = NULL;\
                                                                        while (1)\
                                                                        {\
                                                                            if (ctln == j)\
                                                                            {\
                                                                                head_ctl = ctl;\
                                                                                target = head_ctl->target;\
                                                                                break;\
                                                                            }\
                                                                            \
                                                                            j = j + 1;\
                                                                            ctl = ctl->next;\
                                                                            if (ctl == NULL)\
                                                                            {\
                                                                                break;\
                                                                            }\
                                                                        }\
                                                                        j = 0;\
                                                                        while (ctl != NULL)\
                                                                        {\
                                                                            if (ctl->target != target)\
                                                                            {\
                                                                                break;\
                                                                            }\
                                                                            j = j + 1;\
                                                                            /*std::cout<<ctl->context<<" "<<vocab(ctl->context).c_str()<<" ";*/\
                                                                            ctl = ctl->next;\
                                                                        }\
                                                                        ctln = ctln + j;\
                                                                        context = reinterpret_cast<corpus::corpus_index_type*>(alloc_obj.allocate(sizeof(corpus::corpus_index_type)*j));\
                                                                        j = 0;\
                                                                        while(1)\
                                                                        {\
                                                                            if (head_ctl->target != target)\
                                                                            {\
                                                                                y_pred = target;\
                                                                                break;\
                                                                            }\
                                                                            context[j] = head_ctl->context;\
                                                                            head_ctl = head_ctl->next;\
                                                                            /*j = j - 1*/;\
                                                                            j = j + 1;\
                                                                        }\
                                                                        dim_context.columns = j;\
                                                                        dim_context.rows = 1;\
                                                                    }\
                                                                }\

/*
   @cth, context target head, an instance of "struct context_target_list"
*/
#define DEALLOCATE_CORPUS_TO_CONTEXT_AND_TARGETS(cth) {\
                                                         if (cth.n != 0)\
                                                         {\
                                                            cc_tokenizer::allocator<char> alloc_obj;\
                                                            struct context_target_link* ctl = cth.ptr;\
                                                            while (1)\
                                                            {\
                                                                if (ctl->next == NULL)\
                                                                {\
                                                                    break;\
                                                                }\
                                                                \
                                                                ctl = ctl->next;\
                                                            }\
                                                            while(1)\
                                                            {\
                                                                if (ctl->prev == NULL)\
                                                                {\
                                                                    alloc_obj.deallocate(reinterpret_cast<cc_tokenizer::allocator<char>::pointer>(ctl));\
                                                                    ctl = NULL;\
                                                                    break;\
                                                                }\
                                                                ctl = ctl->prev;\
                                                                alloc_obj.deallocate(reinterpret_cast<cc_tokenizer::allocator<char>::pointer>(ctl->next));\
                                                                ctl->next = NULL;\
                                                            }\
                                                            cth = {0, NULL};\
                                                         }\
                                                      }\

#endif