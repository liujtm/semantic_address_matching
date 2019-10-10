# A Deep Learning Architecture for Semantic Address Matching


Codes in this repository is for the paper **A Deep Learning Architecture for Semantic Address Matching (Lin *et al.* 2019)**.
Cite as **Lin, Yue, & Kang, Mengjun. (2019, October 8). yuelinnnnnnn/semantic_address_matching: Semantic address matching (Version v1.0). Zenodo. http://doi.org/10.5281/zenodo.3476673**

Below is an overview of each file in this repository.

  - `geo_config.py` Hyperparameter settings for the ESIM
  - `geo_data_prepare.py` Tokenize the corpus and convert each address element into index
  - `geo_data_processor.py` Process the labeled address dataset and divide it into training, development and test sets
  - `geo_ESIM.py` Implementation of the enhanced sequential inference model (ESIM)
  - `geo_similarity.py` Calculate statistical characteristics of the labeled address dataset
  - `geo_test.py` Output predictive results of the ESIM on the test set
  - `geo_token.py` Tokenize with the Jieba library
  - `geo_train.py` Train the ESIM and evaluate its accuracy on the development set
  - `geo_word2vec.py` Train word vectors of address elements
  - `other_CRF.py` Tokenize using CRF **[Comber and Arribas-Bel (2019)]** 
  - `other_crf_w2v.py` Train word vectors of address elements (CRF tokenizer)
  - `other_string.py` String similarity-based address matching methods: measure the string relevance
  - `other_w2v_cls.py` Use word2vec embeddings directly for classification: calculat cosine similarity
