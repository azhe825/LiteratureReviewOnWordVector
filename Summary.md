
### Learning Word Vectors for Sentiment Analysis (2011 / 294)
#### Method:
- Word vectors without traditional stopword removal to preserve sentiment components. Stemming not applied, and non-word tokens kept.
- Model word probabilities conditioned on topic mixture variable.
- MLE (maximum likelihood estimate) for unlabeled documents and MAP (maximum a priori) for topic mixture variable. (unsupervised)
- Logistic regression for sentiment classification (supervised)
- Final objective funciton
 
D is unlabeled document, $\theta$ is the topic mixture, R is word representation matrix, $\psi$ is the regression weight. $s_k$ is the sentiment label.
