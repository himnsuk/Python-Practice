### What Does Cross-Validation Mean?
Cross-validation is a technique that is used for the assessment of how the results of statistical analysis generalize to an independent data set. Cross-validation is largely used in settings where the target is prediction and it is necessary to estimate the accuracy of the performance of a predictive model. The prime reason for the use of cross-validation rather than conventional validation is that there is not enough data available for partitioning them into separate training and test sets (as in conventional validation). This results in a loss of testing and modeling capability.

Cross-validation is also known as rotation estimation.

### Explain the steps for a Data analytics project

![Data Analytics Steps](LifeCycleRefresh_FINAL-2.png)

1. Understand the business issue
2. Understand your data set
3. Prepare the data
4. Perform exploratory analysis and modeling
5. Validate your data
6. Visualize and present your findings


### Discuss Artificial Neural Networks

An artificial neural network (ANN) is the piece of a computing system designed to simulate the way the human brain analyzes and processes information. It is the foundation of artificial intelligence (AI) and solves problems that would prove impossible or difficult by human or statistical standards.

### What is Back Propagation?

backward propagation of errors

> Backpropagation, short for "backward propagation of errors," is an algorithm for supervised learning of artificial neural networks using gradient descent. ... Partial computations of the gradient from one layer are reused in the computation of the gradient for the previous layer.


Back-propagation is just a way of propagating the total loss back into the neural network to know how much of the loss every node is responsible for, and subsequently updating the weights in such a way that minimizes the loss by giving the nodes with higher error rates lower weights and vice versa.

### What is a Random Forest?

![Random Forrest](1_VHDtVaDPNepRglIAv72BFg.jpeg)

Random forest, like its name implies, consists of a large number of individual decision trees that operate as an ensemble. Each individual tree in the random forest spits out a class prediction and the class with the most votes becomes our model’s prediction

### What is the importance of having a selection bias?
Selection Bias occurs when there is no appropriate randomization acheived while selecting individuals, groups or data to be analysed.Selection bias implies that the obtained sample does not exactly represent the population that was actually intended to be analyzed.Selection bias consists of Sampling Bias, Data, Attribute and Time Interval.

### Explain the difference between Data Science and Data Analytics

Data analytics is more specific and concentrated than data science. Data analytics focuses more on viewing the historical data in context while data science focuses more on machine learning and predictive modeling. ... On the other hand, data analytics involves a few different branches of broader statistics and analysis.

### What are the important libraries of Python that are used in Data Science?

* TensorFlow.
* NumPy.
* SciPy.
* Matplotlib.
* Pandas.
* Keras.
* SciKit-Learn.
* Statsmodels.
* Plotly
* Seaborn
  
### What is Interpolation and Extrapolation?
Estimating a value from 2 known values from a list of values is Interpolation. Extrapolation is approximating a value by extending a known set of values or facts. ... Interpolation is an estimation of a value within two known values in a sequence of values.

### How can the outlier values be treated?
### How often should an algorithm be updated?
It never happens that the final ML model deployed once takes away the worries forever and keeps giving accurate predictions.

Why is that? Lets figure out below:

1. Model Drift:
To understand this, let us recall one of the most critical assumptions in ML modelling — train and test dataset should belong to similar distribution. And, the model will be good if the new data is similar to the data observed in the past on which the model was trained on.
So, we understand that if test data distribution deviates from that of train data, the model will not hold good. But what could be the possible reasons for such deviation. Well, it can be attributed to many reasons depending on the business case, e.g. change in consumer preferences, fast moving competitive space, geographic shift, economic conditions etc.
Hence, the drifting data distribution calls for an ongoing process of periodically checking the validity of old model. In short, it is critical to keep your machine learning model updated; but the key is when? We will discuss this and a lot more as we proceed, so stay tuned.

2. Robustness:
   People/entities that get affected by the outcome of the ML models may deliberately alter their response in order to send spurious input to the model, thereby escaping the impact of the model predictions. For example, the models such as fraud detection, cyber-security etc receive manipulated and distorted inputs which cause model to output misclassified predictions. Such type of adversaries also drives down the model performance.
3. When ground truth is not available at the time of model training:
   In most of the machine learning models, the ground truth labels are not available to train the model. For example, target variable which captures the response of the end user is not known. In that case, your best bet could be to mock the user action based on certain set of rules coming from business understanding or leverage the open source dataset to initiate model training. But, this model might not necessarily represent the actual data and hence will not perform well until a burn-in period where it starts picking (aka learning) the true actions of the end user. 

What all comes under the scope of model retraining?
* Updating the model parameters?
* Reiterating over the hyper-parameter search space
* Re-running the model selection pipeline across the candidate pool of algorithms
* If that also does not uptick the model performance, then do we need to introduce new features into the model, maybe re-do the feature engineering and selection pipeline?

Lets look at some of the ways to assess the distribution drift:

* **Histogram**: A quick way to visualize the comparison is to draw the histogram — the degree of overlap between the two histograms gives a measure of similarity.
* **K-S statistic**: To check if the upcoming new data belongs to the same distribution as that of training data.
* **Target Distribution**: One quick way to check the consistent predictive power of the ML model is to examine the distribution of the target variable. For example, if your training dataset is imbalanced with 99% data belonging to class 1 and remaining 1% to class 0. And, the predictions reflect this distribution to be around 90%-10%, then it should be treated as an alert for further investigation.
* **Correlation**: Monitoring pairwise correlations between individual predictors will help bring out the underlying drift

### Define the term deep learning
Deep learning is a type of machine learning and artificial intelligence (AI) that imitates the way humans gain certain types of knowledge. Deep learning is an important element of data science, which includes statistics and predictive modeling.

### What is Normal Distribution

![Normal Distribution](normal_distribution.png)
In probability theory, a normal distribution is a type of continuous probability distribution for a real-valued random variable

### Which language is best for text analytics? R or Python?
Python
### Explain the benefits of using statistics by Data Scientists
Get meaningful insight from data
### Name various types of Deep Learning Frameworks
* TensorFlow. Google's open-source platform TensorFlow is perhaps the most popular tool for Machine Learning and Deep Learning. ...
* PyTorch. PyTorch is an open-source Deep Learning framework developed by Facebook. ...
* Keras. ...
* Sonnet. ...
* MXNet. ...
* Swift for TensorFlow. ...
* Gluon. ...
* DL4J.

### What is skewed Distribution & uniform distribution?
Skewed distribution is a condition when one side (either right or left) of the graph has more dataset in comparison to the other side. Uniform distribution is a condition when all the observations in a dataset are equally spread across the range of distribution.

### What is reinforcement learning?
Reinforcement learning is an area of Machine Learning. It is about taking suitable action to maximize reward in a particular situation. ... In the absence of a training dataset, it is bound to learn from its experience. Example: The problem is as follows: We have an agent and a reward, with many hurdles in between

### What is precision?

Precision is defined as the number of true positives divided by the number of true positives plus the number of false positives. ... While recall expresses the ability to find all relevant instances in a dataset, precision expresses the proportion of the data points our model says was relevant actually were relevant.

### Do you prefer Python or R for text analytics?

Python

### What is Cluster Sampling?

Cluster sampling is a probability sampling method in which you divide a population into clusters, such as districts or schools, and then randomly select some of these clusters as your sample. ... In single-stage sampling, you collect data from every unit within the selected clusters.
