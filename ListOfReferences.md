| Year | Cite | Title                                                                                                        | Data                                                                                                                                                                                        | Method                                                                                                                                                                                                                                | Evaluation                                                                                                                                                                                 |
|------|------|--------------------------------------------------------------------------------------------------------------|---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| 2011 | 294  | Learning Word Vectors for Sentiment Analysis                                                                 | Pang and Lee Movei Review Dataset (2004);IMDB (Internet Movie Database) with balanced positive and negative classes                                                                         | Word vectors without traditional stopword removal to preserve sentiment components. Stemming not applied, and non-word tokens kept. Model word probabilities conditioned on topic mixture variable.                                   | Both models (w/wo sentiment term) perform better than LSA. Improvement over the bag-of-word baseline.                                                                                      |
| 2003 | 145  | A neural probabilistic language model                                                                        | the Brown corpus; Associated Press (AP) News from 1995 and 1996.                                                                                                                            | The model learns simultaneously (1) a distibuted representation for each word along with (2) the probability function for word sequences, expressed in terms of these representations.                                                | improves on state-of-the-art n-gram models, and that the proposed approach allows to take advantage of longer contexts.                                                                    |
| 2003 | 2933 | Latent dirichlet allocation                                                                                  | TREC AP corpus (Harman, 1992)                                                                                                                                                               | Hierarchical Bayesian model in which each item of a collection is modeled as a finite mixture over an underlying set of topics. Each topic is, in turn, modeled as an infinite mixture over an underlying set of topic probabilities. |                                                                                                                                                                                            |
| 2008 | 124  | A unified architecture for natural language processing: deep neural networks with multitask learning         | PropBank dataset                                                                                                                                                                            | Part-Of-Speech Tagging; Chunking, labeling sentence segments or phrases; Named Entity Recognition; Semantic role labeling; Word vectors with neural network for multitasking learning                                                 | Improve the SRL(Semantic Role Labeling) performance                                                                                                                                        |
| 2009 | 32   | Joint parsing and named entity recognition                                                                   | LDC2008T04 OntoNotes Release 2.0 corpus                                                                                                                                                     | a discriminative feature-based constituency parser                                                                                                                                                                                    | improvements of up to 1.36% absolute F1 for parsing, and up to 9.0% F1 for named entity recognition                                                                                        |
| 2006 | 56   | Seeing stars when there aren't many stars: graph-based semi-supervised learning for sentiment categorization | Movie revews used in (Pang and Lee, 2005)                                                                                                                                                   | Semi-supervised, KNN, Graph with each document at one node                                                                                                                                                                            | Achieved better performance than all other methods in all four author corpora                                                                                                              |
| 2009 | 97   | Joint sentiment/topic model for sentiment analysis                                                           | Movie Review Dataset                                                                                                                                                                        | a joint sentiment/topic (JST) model by adding an additional sentiment layer between the docu- ment and the topic layer.                                                                                                               | Unsupervised. Accuracy is lower but close to other listed methods.                                                                                                                         |
| 2013 | 1205 | Efficient Estimation of Word Representations in Vector Space                                                 | Google New corpus                                                                                                                                                                           | Continuous Bag-of-Words model; Continuous Skip-gram model                                                                                                                                                                             | large improvements in accuracy at much lower computational cost                                                                                                                            |
| 2013 | 1133 | Distributed Representations of Words and Phrases and their Compositionality                                  | Google New articles                                                                                                                                                                         | Skip-gram model; Subsampling; Hierarchical softmax                                                                                                                                                                                    | Large amount of training data is crucial to increase the accuracy. A big Skip-gram model outperform all previously published word representation methods.                                  |
| 2004 | 65   | Generating Overview Summaries of Ongoing Email Thread Discussions                                            | Archives of Columbia Unversity ACM Student Chapter Committee                                                                                                                                | Combination of traditional vector space techniques and Singular Value Decomposition (SVD).                                                                                                                                            | a combination of simple word vector approaches with singular value decomposition approaches do well at extracting discussion issues.                                                       |
| 2015 | 3    | Evaluation of Word Vector Representations by Subspace Alignment                                              | WS-353 dataset (Finkelstein et al., 2001), MEN dataset (Bruni et al., 2012), SimLex-999 (Hill et al., 2014) for word similarity test. 20 Newsgroups (20NG) dataset for text classification. | Construct Word Vectors with annotation dismensions.Word Vector Evaluation Models                                                                                                                                                      | Pearson correlation for intrinsic and extrinsic score r = 0.87                                                                                                                             |
| 2014 | 64   | Learning Sentiment-Specific Word Embedding for Twitter Sentiment Classification                              | latest Twitter sentiment clas- sification benchmark dataset in SemEval 2013 (Nakov et al., 2013)                                                                                            | sentiment specific word embedding (SSWE) which encodes sentiment information in the continuous representation of words. Neural networks for classification.                                                                           | (1) the SSWE feature preforms comparably with hand-crafted freatrures in the top-performed system; (2) the performance is futhre improved by concatenating SSWE with existing feature set. |
| 2012 | 17   | Baseline and Bigrams: Simple, Good Sentiment and Topic Classification                                        | Movie Reviews, Customer Reviews, News group dataset                                                                                                                                         | Linear classifier and log-count ratio. Multinominal NB and SVM.                                                                                                                                                                       | NB better at sentiment snippet task; SVM better at full-length review.                                                                                                                     |