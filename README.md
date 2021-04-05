# Sentiment Analysis and Topic Modeling on COVID-19 Vaccine Tweets
## Introduction
Since the pandemic started, various research teams have worked on analyzing the sentiment of COVID-19 specific tweets. For example, Chakraborty and her team found through a deep learning classifier that though most tweets regarding COVID-19 were positive, many re-tweets contained little to no useful words but disseminated useless information just as the pandemic had spread into humanity (Chakraborty et al., 2020, p. 106754). There is also research that focuses more on the machine learning perspective than the societal perspective: Sethi and his team analyzed bi-class and multi-class setting over n-gram feature set along with a cross-dataset evaluation of different machine learning techniques, achieving a maximum accuracy of about 93% in detecting the actual sentiment behind a tweet related to COVID-19 (Sethi et al., 2020, pp. 1–3).

Despite all the previous work, little is done on analyzing the public's perception of the COVID-19 vaccine. For months, scientists around the world have been racing to produce an effective vaccine and governments have been throwing billions at pharmaceutical companies to be first in line for access. But despite the severe disruption caused by the pandemic, a significant minority of people say they don’t want to register the vaccine even when one becomes available. As a result, this project analyzes the tweets with the #CovidVaccine hashtag as an attempt to acquire more insights into the public's perception and the root cause of skepticism towards COVID vaccines. 

Website Repository: https://github.com/zhiyichenGT/zhiyichenGT.github.io

## Problem Definition
This project tackles a few problems: what is the public’s perception and acceptance of the COVID-19 vaccine. If any skepticism exists, what are the major factors contributing to the skepticism. To address the problem, we will use supervised learning to classify the sentiments of the tweets and utilize unsupervised learning to discovered inherent topics upon which the vaccine skepticism might have grown. Aside from that, we plan to incorporate the geolocations of the users as well as their profile descriptions as features into our models.

## Methods
We plan to divide the project into three stages: Exploratory Data Analysis (EDA), Supervised Learning, and Unsupervised Learning.

### Datasets
For this project, we will be using two datasets:
- [Covid Vaccine Tweets](https://www.kaggle.com/kaushiksuresh147/covidvaccine-tweets)
  - This is a collection of tweets with the #CovidVaccine hashtag. It started on January 8, 2020, and is updated on a daily basis. For our project, we will be utilizing the data collected by February 28, 2021.
- [Stanford Sentiment Treebank](https://nlp.stanford.edu/sentiment/)
  - Introduced in Recursive Deep Models for Semantic Compositionality Over a Sentiment Treebank (Socher et al., 2013, p. 1637). It includes fine-grained sentiment labels for 215,154 phrases in the parse trees of 11,855.

### Exploratory Data Analysis (EDA)
In the EDA, we plan to explore the geographical distributions of the tweets as well as other information such as the number of friends and followers of the users. We expect this information to provide us with some insights about the user such as his or her location and scale of social networks on the internet.

### Supervised Learning
For supervised learning, we plan to divide it into two parts: (1) dictionary-based sentiment analysis on the tweets using [SentiWords](https://hlt-nlp.fbk.eu/technologies/sentiwords) and [VADER](https://github.com/cjhutto/vaderSentiment) (Valence Aware Dictionary for Sentiment Reasoning); (2) training a model using the Stanford Sentiment Treebank (SST) and testing on the Covid Vaccine Tweets (CVT) dataset. 

#### Dictionary-based Sentiment Analysis
For this approach, we plan to calculate the sentiment scores based on different sentiment dictionaries and aggregate the word-level sentiment scores to obtain a document-level sentiment level on each tweet. We will compare the results of using different sentiment dictionaries as well as the model trained using the SST dataset as described below.

#### Neural Network-based Sentiment Analysis
For this part of the project, we will use the SST as our training data and the CVT as our testing data. This can also be divided into two parts. The first part will be fine-tuning pre-trained LSTM-based and Transformer-based models such as BERT using the SST dataset. The second part is to build an LSTM model as a baseline to compare the performances of these two models as well as the vanilla dictionary-based models described above.

To evaluate the result, we will use the F1 score and confusion matrix.

### Unsupervised Learning
#### Topic Modeling
Aside from building a supervised classification model, we will check the nature of the data using unsupervised learning, performing experiments using different topic modeling algorithms, such as LSA (Latent Semantic Analysis), PLSA (Probabilistic Latent Semantic Analysis), LDA (Latent Dirichlet Allocation), and lda2Vec (a deep learning approach), to uncover the inherent topic upon which the approval or skepticism of COVID vaccine is built. This task mainly evolves around the tweets.
Common evaluation metrics such as silhouette score will be used. We will also use PCA (the first two principal components) and tSNE to visualize our high dimensional clustering result.

## References
- Boyon, N. (2020, September 1). Three in four adults globally say they would get a vaccine for COVID-19. Ipsos. https://www.ipsos.com/en-us/news-polls/WEF-covid-vaccine-global
- Chakraborty, K., Bhatia, S., Bhattacharyya, S., Platos, J., Bag, R., & Hassanien, A. E. (2020). Sentiment Analysis of COVID-19 tweets by Deep Learning Classifiers—A study to show how popularity is affecting accuracy in social media. Applied Soft Computing, 97, 106754. https://doi.org/10.1016/j.asoc.2020.106754
- Guàrdia, A. B., & Hirsch, C. (2020, September 18). Coronavirus vaccine skepticism — by the numbers. POLITICO. https://www.politico.eu/article/coronavirus-vaccine-skepticism-by-the-numbers/
- Sethi, M., Pandey, S., Trar, P., & Soni, P. (2020, July). Sentiment Identification in COVID-19 Specific Tweets. 2020 International Conference on Electronics and Sustainable Communication Systems (ICESC). https://doi.org/10.1109/icesc48915.2020.9155674
- Socher, R., Perelygin, A., Wu, J., Chuang, J., Manning, C. D., Ng, A., & Potts, C. (2013, October). Recursive Deep Models for Semantic Compositionality Over a Sentiment Treebank. In D. Yarowsky, T. Baldwin, A. Korhonen, K. Livescu, & S. Bethard (Eds.), Proceedings of the 2013 Conference on Empirical Methods in Natural Language Processing (pp. 1631–1642). Association for Computational Linguistics.


