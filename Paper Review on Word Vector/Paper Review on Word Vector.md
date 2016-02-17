# Paper Review on Word Vector

## Learning Word Vectors for Sentiment Analysis (2011 / 294)
### Abstract:
Unsupervised vector-based approaches to semantics can model rich lexical meanings, but they largely fail to capature sentiment information that is central to many word meanings and important for a wide range of NLP tasks. We present a model that uses a mix of unsupervised and supervised techniques to learn word vectors capturing semantic term-document information as well as rich sentiment content. The proposed model can leverage both continuous and multi-dimensional sentiment information as well as non-sentiment annotations. We instantiate the model to unitlize the document-level sentiment polarity annotations present in many online documents. We evaluate the model using small, widely used sentiment and subjectivity copora and find it out-performs several previously introduced methods for sentiment classification. We also introduce a large dataset of movie reviews to serve as a more robust benchmark for work in this area.

### Dataset: 
- Pang and Lee Movei Review Dataset (2004)
- IMDB (Internet Movie Database) with balanced positive and negative classes

### Method:
- Word vectors without traditional stopword removal to preserve sentiment components. Stemming not applied, and non-word tokens kept.
- Model word probabilities conditioned on topic mixture variable.
- MLE (maximum likelihood estimate) for unlabeled documents and MAP (maximum a priori) for topic mixture variable. (unsupervised)
- Logistic regression for sentiment classification (supervised)
- 10-fold cross-validation
- Final objective funciton
![Screen Shot 2016-02-15 at 10.19.34 PM.png](quiver-image-url/8DBA9149032AEB42EF43277775BC7280.png)

### Evaluation:
- Both models (w/wo sentiment term) perform better than LSA.
- Improvement over the bag-of-word baseline.
![Screen Shot 2016-02-15 at 10.21.24 PM.png](quiver-image-url/E4FDE2909A7C62E1AA30F2A5272C91FA.png)

## (REF) Emotions from text machine learnig for text-based emotion prediction (2005/78)
### Abstract: 
In addtion to information text contains attitudinal, and more specifically, emotional content. This paper explores the text-based emotion prediction problem empirically, using supervised machine learnig with the SNoW learning architecture. The goal is to classify the emotional affinity of sentences in the narrative domain of children's fairy tales, for subsequent usage in appropriate expressive rendering of text-to-speech synthesis. In initial experiments on a preliminary data set of 22 fairy tales show encouraging results over a naive baseline and BOW approach for classifcation of emotional versus non-emotional contents, with some dependency on parameter tuning. We also discuss rsults for a tripartie model which covers emotional valence, as well as feature set alternations. In addition, we present plans for a more cognitively sound sequential model, taking into consideration a larger set of basic emotions.

### NO WORD VECTOR

## (REF) Mining WordNet for Fuzzy Sentiment: Sentiment Tag Extraction from WordNet Glosses (2006/285)
### Abstract:
Many of the tasks required for semantic tagging of phrases and texts rely on a list of words annotated with some semaantic features. We present a method for extractin sentiment-bearing adjectives from WordNet using the Sentiment Tag Extraction Program (STEP). We did 58 STEP runs on unique non-intersecting seed lists drawn from manually annotated list of positive and negative adjectives and evaluated the results against other manually annotated lists. The 58 runs were then collapsed into a single set o f7813 unique words. For each word we computed a Net Overlap Score by subtracting the total number of runs assigning this word a negative sentiment from the total of the runs that consider it positive. We demonstrate that Net Overlap Score can be used as a measure of the words degree of membership in the fuzzy category of sentiment: the core adjectives, which had the highest Net Overlap Scores, were identified most accurately both by STEP and by human annotators, while the words on the periphery of the category had the lowest scores and were associated with low rates of inter-annotator agreement.

### NO WORD VECTOR

## (REF) A neural probabilistic language model (2003/145)
### Abstract:
A goal of statistical language modeling is to learn the joint probability function of sequences of words in a language. This is intrinsically difficult because of the curse of dimensionality: a word sequence on which the model will be tested is likely to be different from all the word sequences seen during training. Traditional but very successful approaches based on n-grams obtain generalization by concatenating very short overlapping seqquences seen in the training set. We propose to fing the curse of dimensionality by __learning a distributed representation__ for words which allows each training sentence to inform the model about an exponential number o fsemantically neighboring sentences. The model learns siultaneously (1) a distibuted representation for each word along with (2) the probability function for word sequences, expressed in terms of these representations.  Generalization is obtaineed because a sequence of words that has never been seen before gets high probability if it is made of words that are similar (in the sense of having a nearby representation) to words forming an already seen sentence. Training such large models (with millions of parameters) within a reasonable time is iteself a significant challenge. We report on experiments using __neural networks for the probability function__, showing on two text corpora that the proposed approach significantly __improves on state-of-the-art n-gram models__, and that the proposed approach allows to take advantage of longer contexts.

### TO BE CONTINUED...

## (REF) Latent dirichlet allocation (2003/2933)
### Abstract: 
We describe latent Dirichlet allocation (LDA), a generative probabilistic model for collections of discrete data such as text corpora. LDA is a three-level __hierarchical Bayesian model__, in which each item of a collection is modeled as a finite mixture over an underlying set of topics. Each topic is, in turn, modeled as an infinite mixture over an underlying set of topic probabilities. In the context of text modeling, the topic probabilities provide an explicit representation of a document. We present efficient approximate inference techniques based on variational methods and an EM algorithm for empirical Bayes parameter estimation. We report results in document modeling, text classification, and collaborative filtering, comparing to a mixture of unigrams model and the probabilistic LSI model.

### Model:
![Screen Shot 2016-02-15 at 11.18.22 PM.png](quiver-image-url/19497D7EBE5A910B4A6FA59876B4F900.png)

### TO BE CONTINUED...

## (REF) A unified architecture for natural language processing: deep neural networks with multitask learning (2008/124)
### Abstract:
We describe a single convolutional neural network architecture that, given a sentence, outputs a host of language processing predictions: part-of-speech tages, chunks, named entity tags, semantic roles, semantically similar words and the likelihood that the sentence makes sense (grammatically and semantically) using a language model. The entire network is trained jointly on all these tasks using weight-sharing, an instance of multitask learning. All the tasks use labeled data except the language model which is learnt from unlabeled text and represents a novel form of semi-supervised learning for the shared tasks. We show how both multitask learning and semi-supervised learning improve the generalization of the shared tasks, resulting in state-of-the-art performance.

### Dataset: 
 Sections 02-21 of the PropBank dataset version 1 (about 1 million words) for training and Section 23 for testing as standard in all SRL experiments. POS and chunking tasks use the same data split via the Penn TreeBank. NER labeled data was obtained by running the Stanford Named Entity Recognizer on Wikipedia.

### Method:
- Part-Of-Speech Tagging; Chunking, labeling sentence segments or phrases; Named Entity Recognition; Semantic role labeling; 
- Word vectors with neural network for multitasking learning

### Evaluation:
![Screen Shot 2016-02-15 at 11.59.02 PM.png](quiver-image-url/B34D973C1D23661B0137C6446AF8C0D6.png)

## (REF) Joint parsing and named entity recognition (2009/32)
### Abstract:
For many language technology applications, such as question answering, the overall system ruuns several independent processors over the data (such as a named entity recognizer, a coreference system, and a parser). This easily results in inconsistent annotations, which are harmful to the performance of the aggregate system. We begin to address this problem with __a joint model of aprsing and named entity recognition, based on a discriminative feature-based consttuency parser__. Our model produced a consistent output, where the named entity spans do not conflict with the phrasal spans of the parse tree. The joint representation also allows the information from each type of annotation to improve performance on the other, and, in experiments with the OntoNotes corpus, we found improvements of up to 1.36% absolute F1 for parsing, and up to 9.0% F1 for named entity recognition.

### JUST CHUNKING AND TAGGING

## (REF) Seeing stars when there aren't many stars: graph-based semi-supervised learning for sentiment categorization (2006/56)
### Abstract: 
We present a graph-based semi-supervised learning algorithm to address the sentiment analysis task of rating inference. Given a set of documents (e.g. movie reviews) and accompanying ratings, the task calls for inferring numerical ratings for unlabeled documents based on the perceived sentiment expressed by their text. In particular, we are interested in the __situation where labeled data is scarce__. We place this task in the semi-supervised setting and demonstrate that considering unlabeled reviews in the learning process can improve rating-inference performance. We do so by __creating a graph on both labeled and unlabeled data to encode certain assumptions__ for this task. We then solve an optimization problem to obtain a smooth rating function over the whole graph. When only limited labeled data is available, this method achieves significantly better predictive accuracy over othe rmethods that ignore the unlabeled examples during training.

## (REF) Joint sentiment/topic model for sentiment analysis (2009/97)
### Abstract:
Sentiment analysis or opinion mining aims to use automated tools to detect subjective information such as opinions, attitudes, and feelings expressed in text. This paper proposes a novel probabilistic modeling framework based on LDA, called joint sentiment/topic model (JST), which detects sentiment and topic simultaneously from text. Unlike other amchine learning approaches to sentiment classification which often require albeled corpora for classifier training, the proposed JST model is fully unsupervised. The model has been evaluated on the movie review dataset to classify the review sentiment polarity and minimum prior information have also been explored to further improve the sentiment classification accuracy. Preliminary experiments have shown promising results achieved by JST.

## (REF) Thumbs Up?: sentiment classification using machine learning techniques (2002/890)
### Abstract:
We consider the problem of classifying documents not by topic, but by overall sentiment, e.g., determing whether a review is positive or negative. Using __movie reviews as data__, we find that standard machine learning techniques definitively outperform human-produced baselines. However, the three machine learning methods we employed (__Naive Bayes, maximum entropy classification, and support vector machines) do not perform as well on sentiment classification as on traditional topic-based categorization__. We conclude by examining factors that make the sentiment classification problem more challenging.

### TO BE CONTINUED...

## (REF) Word representations: a simple and general method for semi-supervised learning (2010/87)
### Abstract:
If we take an existing supervised NLP system, a simple and gernal way to improve accuracy is to use unsupervised word representations as extra word features. We evaluate Brown clusters, Collobert and Weston (2008) embeddings, and HLBL (Mnih & Hinton, 2009) embeddings of words on both NER and chunking. We use near state-of-the-art supervised baselines, and find that each of the three word representations improves the accuracy of these baselines. We find further improvements by combinng different word representations. 

### TO BE CONTINUED...

## Baseline and Bigrams: Simple, Good Sentiment and Topic Classification (2012/17)
### Abstract:
Variants fo Naive Bayes (NB) and Support Vector Machines (SVM) are often used as baseline methods for text classification, but their performance varies greatly depending on the model variant, features used and task/dataset. We show that: (i) the __inclusion of word bigram features gives consistent gains on sentiment analysis tasks__; (ii) for short snippet sentiment tasks, NB actually does better than SVMs (while for longer documents the opposite result holds); (iii) a simple but novel __SVM variant using NB log-count ratios as feature values__ consistently performs well across tasks and datasets. Based on these observations, we identify simple NB and SVM variants which outperform most published results on sentiment analysis dataesets, sometimes providing a new state-of-the-art performance level.

### TO BE CONTINUED...

## End-to-End text recognition with convolutional neural networds (2012/121)
### Abstract:
Full end-to-end text recognition in nautural images is a challenging problem that has received much attention recently. Traditional systems in this area have relied on elaborate models incorporating carefully hand-engineered features or large amounts of prior knowledge. In this paper, we take a different route and combine the representational power of large, multilayer neural networks together with recent developments in unsupervised feature learning, which allows us to use a common framework to train highly-accurate text detector and character recognizer modules. Then, using only simple off-the-shelf methods, we integrate these two modules into a full end-to-end, lexicon-driven, scene text recognition system that achieves state-of-the-art performance on standard benchmarks, namely Street View Text and ICDAR 2003.

## Learning Sentiment-Specific Word Embedding for Twitter Sentiment Classification (2014/64)
### Abstract:
We present a method that learns word embedding for Twitter sentiment classification in this paper. Most existing algorithms for learning continuous word representations typically only model the syntactic context of words but ignore the sentiment of text. This is problematic for sentiment analysis as they usually map words with similar syntactic context but opposite sentiment polarity such as good and bad, to neighboring word vectors. We address this issue by learning __sentiment specific word embedding (SSWE)__, which encodes sentiment information in the continuous representation of words. Specifically, we develop three __neural networks__ to effectively incorporate the supervision from sentiment polarity of text in their loss functions. To obtain large scale training corpora, we learn the sentiment-specific word embedding from mamssive distant-supervised tweets collected by postive and negative emotions. Experiments onn applying SSWE to a benchmark Twitter sentiment classification dataset in SemEval 2013 show that (1) the SSWE feature preforms comparably with hand-crafted freatrures in the top-performed system; (2) the performance is futhre improved by concatenating SSWE with existing feature set.

## *Evaluation of Word Vector Representations by Subspace Alignment (2015/3)
### Abstract:
Unsupervisedly learned word vectors have proven to provide exceptionally effective features in many NLP tasks. Most common intrinsic evaluations of vector quality measure correlation with similarity judgements. However, thesee often correlate poorly with how well the learned represetnations perform as features in downstream evaluation tasks. We present QVEC--a computationally inexpensive intrinsic evaluation measure of the quality of word embeddings based on alignment to a matrix of features extracted from manually crafted lexical resources--that obrains strong correlation with performance of the vectors in a battery of downstream semantic evaluation tasks.

### Dataset: 
- an existing semantic resource: SemCor(Miller et al. 1993)
- WordNet (Fellbaum 1998)
- WS-353 dataset (Finkelstein et al., 2001), MEN dataset (Bruni et al., 2012), SimLex-999 (Hill et al., 2014) for word similarity test.
- 20 Newsgroups (20NG) dataset for text classification

### Method:
- Construct Word Vectors with annotation dismensions
![Screen Shot 2016-02-16 at 12.44.57 PM.png](quiver-image-url/D8F0D2659119E9B7AD9C206E7349C804.png)
- Word Vector Evaluatio Model
![Screen Shot 2016-02-16 at 12.47.23 PM.png](quiver-image-url/E0A803341048E3A19624A2AB4DD7451F.png)

### Evaluation:
![Screen Shot 2016-02-16 at 12.54.04 PM.png](quiver-image-url/B238885AD602E5C98928158270040CF3.png)

### TO BE CONTINUED...

## *Generating Overview Summaries of Ongoing Email Thread Discussions (2004/65)
### Abstract:
The tedious task of responding to a blacklog of email is one which is familiar to many researchers. As a subset of email management, we address the problem of constructing a summary of email discussios. Specifically, we examine ongoing discussions which will untilmately culminate in a consensus in a decsion-making process. Our summary provides a snapshot of the current state-of-affairs of the discussio and facilitates a speedy response fromthe user, who might be the bottleneck in some matter being resolved. We present a method which uses the sructure of the thread dialogue and word vector techniques to determine which sentence in the thread shoud be extracted as the main issue. Our solution successfully identifies the sentence containing the issue of the thread being discussed, potentially more informative that subject line.

### Dataset:
Archives of Columbia Unversity ACM Student Chapter Committee

### Method:
- Combination of traditional vector space techniques and Singular Value Decomposition (SVD).
![Screen Shot 2016-02-16 at 1.07.53 PM.png](quiver-image-url/A8787AB08E4A4B294048097242190E09.png)

### TO BE CONTINUED...

## *Distributed Representations of Words and Phrases and their Compositionality (2013/1133)
### Abstract:
The recently introduced continuous Skip-gram model is an efficient method for learning hgih-quality distributed vector representations that capture a large number of precise syntatic and semantic word relationships. In this paper we present several extensions that improve both the quality of the vectors and the training speed. By subsampling of the frequent words we obtain significant speedup and also learn more regular word represetntations. We also describe a simple alternative to the hierarchical softmax called negative sampling. An inherent limitation of word representations is their indifference to word order and their inability to represent idiomatic phrases. For example, the meanings of Canada and Air cannot be easily combined to obtain Air Canada. Motivated by this example, we present a simple method for finding phrases in text, and show that learning good vector represenations for millions of phrases is possible.

### Dataset:
Google New articles

### Method:
- Skip-gram model (Advantage over previous neural network: not involve dense matrix multiplication).
- Subsampling to counter the imbalance between the rare and frequent words.
- Hierarchical softmax.

### Evaluation:
- Large amount of training data is crucial to increase the accuracy.
- A big Skip-gram model outperform all previously published word representation methods.

### TO BE CONTINUED...

## *Efficient Estimation of Word Representations in Vector Space (2013/1205)
### Abstract:
We propose two novel moel architectures for computing continuous vector representations of words from very large data sets. The quality of these representations is measured in a word similarity task, and the results are compared to the previously best performing techninques based on different types of neural networks. We observe large improvements in accuracy at much lower computational cost, i.e. it takes less than a day to learn high quality word vectors from a 1.6 billion words data set. Furthermore, we show that these vectors provide state-of-art performance on our test set for measuring syntactic and semantic word similarities.

### Dataset:
Google New corpus

### Models:
![Screen Shot 2016-02-16 at 11.21.30 PM.png](quiver-image-url/33E039B2CEE46380A230C3E8E968F0F2.png)

### TO BE CONTINUED..



