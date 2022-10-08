LINK => [LDA](https://www.machinelearningplus.com/nlp/topic-modeling-python-sklearn-examples/#:~:text=The%20most%20important%20tuning%20parameter,be%20%3E%201)

### Preprocessing
Stop-word elimination: removal of the most common words in a language that are not helpful and in general unusable in text mining like prepositions, numbers, and words that do not contain applicable information for the study. In fact, in NLP, there is no particular general list of stop words used by all developers who choose their list based on their goal to improve the recommendation system performance.

• Stemming: the conversion of words into their root, using stemming algorithms such as Snowball Stemmer.

• Lemmatizing: used to enhance the system's accuracy by returning the base or dictionary form of a word.

• Tokenizing: dividing a text input into tokens like phrases, words, or other meaningful elements (tokens). The outcome of tokenization is a sequence of tokens.

• Identifying n-gram procedure such as bigram (phrases containing two words) and trigram (phrases containing three words) words and consider them as one word.

After the preprocessing step, we applied a commonly used term-weighting method called TF-IDF, which is a pre-filtering stage with all the included TM methods. TF-IDF is a numerical statistic measure used to score the importance of a word (term) 

## Description of Assumptions 

- Documents with similar topics will use similar group of words
- Document Definition/Modeling
  - Documents are probability distribution over latent topic
  - Topics are probability distribution over words

LDA takes a number of documents. It assumes that the words are in each document are related. It then tries to figure out the "recipe" for how each document could have been created. We just need to tell the model how many topics to construct and it uses that "recipe" to generate topic and word distributions over a corpus. Based on that output, we can identify similar documents within the corpus.


**Advantages**
 - LDA is an effective tool for topic modeling.
 - Easy to understand conceptually
 - Has been shown to produce good results over many domains.
 - New application

**Limitations**

-  Must know the number of topics K in advance
- Dirichlet topic distribution cannot capture correlations among topics


### Hyperparameter Tuning in LDA
First, let’s differentiate between model hyperparameters and model parameters :

__Model hyperparameters__ can be thought of as settings for a machine learning algorithm that are tuned by the data scientist before training. Examples would be the number of trees in the random forest, or in our case, number of topics K

__Model parameters__ can be thought of as what the model learns during training, such as the weights for each word in a given topic
Now that we have the baseline coherence score for the default LDA model, let’s perform a series of sensitivity tests to help determine the following model hyperparameters:

1. Number of Topics (K)
1. Dirichlet hyperparameter alpha: Document-Topic Density
1. Dirichlet hyperparameter beta: Word-Topic Density

## Load the packages


```python
# !pip install pyLDAvis
```


```python
# !python3 -m spacy download en
```


```python
# Run in terminal or command prompt
# python3 -m spacy download en

import numpy as np
import pandas as pd
import re, nltk, spacy, gensim

# Sklearn
from sklearn.decomposition import LatentDirichletAllocation, TruncatedSVD
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.model_selection import GridSearchCV
from pprint import pprint

# Plotting tools
import pyLDAvis
import pyLDAvis.sklearn
import matplotlib.pyplot as plt
%matplotlib inline
```

    /usr/local/lib/python3.7/dist-packages/past/types/oldstr.py:5: DeprecationWarning: Using or importing the ABCs from 'collections' instead of from 'collections.abc' is deprecated since Python 3.3,and in 3.9 it will stop working
      from collections import Iterable



```python

```

## Import Newsgroups Text Data


```python
# Import Dataset
df = pd.read_json('https://raw.githubusercontent.com/selva86/datasets/master/newsgroups.json')
print(df.target_names.unique())
```

    ['rec.autos' 'comp.sys.mac.hardware' 'comp.graphics' 'sci.space'
     'talk.politics.guns' 'sci.med' 'comp.sys.ibm.pc.hardware'
     'comp.os.ms-windows.misc' 'rec.motorcycles' 'talk.religion.misc'
     'misc.forsale' 'alt.atheism' 'sci.electronics' 'comp.windows.x'
     'rec.sport.hockey' 'rec.sport.baseball' 'soc.religion.christian'
     'talk.politics.mideast' 'talk.politics.misc' 'sci.crypt']



```python
df.head(15)
```





  <div id="df-a4c36cfb-26a6-4086-9f06-e4b9c901edb7">
    <div class="colab-df-container">
      <div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>content</th>
      <th>target</th>
      <th>target_names</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>From: lerxst@wam.umd.edu (where's my thing)\nS...</td>
      <td>7</td>
      <td>rec.autos</td>
    </tr>
    <tr>
      <th>1</th>
      <td>From: guykuo@carson.u.washington.edu (Guy Kuo)...</td>
      <td>4</td>
      <td>comp.sys.mac.hardware</td>
    </tr>
    <tr>
      <th>2</th>
      <td>From: twillis@ec.ecn.purdue.edu (Thomas E Will...</td>
      <td>4</td>
      <td>comp.sys.mac.hardware</td>
    </tr>
    <tr>
      <th>3</th>
      <td>From: jgreen@amber (Joe Green)\nSubject: Re: W...</td>
      <td>1</td>
      <td>comp.graphics</td>
    </tr>
    <tr>
      <th>4</th>
      <td>From: jcm@head-cfa.harvard.edu (Jonathan McDow...</td>
      <td>14</td>
      <td>sci.space</td>
    </tr>
    <tr>
      <th>5</th>
      <td>From: dfo@vttoulu.tko.vtt.fi (Foxvog Douglas)\...</td>
      <td>16</td>
      <td>talk.politics.guns</td>
    </tr>
    <tr>
      <th>6</th>
      <td>From: bmdelane@quads.uchicago.edu (brian manni...</td>
      <td>13</td>
      <td>sci.med</td>
    </tr>
    <tr>
      <th>7</th>
      <td>From: bgrubb@dante.nmsu.edu (GRUBB)\nSubject: ...</td>
      <td>3</td>
      <td>comp.sys.ibm.pc.hardware</td>
    </tr>
    <tr>
      <th>8</th>
      <td>From: holmes7000@iscsvax.uni.edu\nSubject: WIn...</td>
      <td>2</td>
      <td>comp.os.ms-windows.misc</td>
    </tr>
    <tr>
      <th>9</th>
      <td>From: kerr@ux1.cso.uiuc.edu (Stan Kerr)\nSubje...</td>
      <td>4</td>
      <td>comp.sys.mac.hardware</td>
    </tr>
    <tr>
      <th>10</th>
      <td>From: irwin@cmptrc.lonestar.org (Irwin Arnstei...</td>
      <td>8</td>
      <td>rec.motorcycles</td>
    </tr>
    <tr>
      <th>11</th>
      <td>From: david@terminus.ericsson.se (David Bold)\...</td>
      <td>19</td>
      <td>talk.religion.misc</td>
    </tr>
    <tr>
      <th>12</th>
      <td>From: rodc@fc.hp.com (Rod Cerkoney)\nSubject: ...</td>
      <td>4</td>
      <td>comp.sys.mac.hardware</td>
    </tr>
    <tr>
      <th>13</th>
      <td>From: dbm0000@tm0006.lerc.nasa.gov (David B. M...</td>
      <td>14</td>
      <td>sci.space</td>
    </tr>
    <tr>
      <th>14</th>
      <td>From: jllee@acsu.buffalo.edu (Johnny L Lee)\nS...</td>
      <td>6</td>
      <td>misc.forsale</td>
    </tr>
  </tbody>
</table>
</div>
      <button class="colab-df-convert" onclick="convertToInteractive('df-a4c36cfb-26a6-4086-9f06-e4b9c901edb7')"
              title="Convert this dataframe to an interactive table."
              style="display:none;">

  <svg xmlns="http://www.w3.org/2000/svg" height="24px"viewBox="0 0 24 24"
       width="24px">
    <path d="M0 0h24v24H0V0z" fill="none"/>
    <path d="M18.56 5.44l.94 2.06.94-2.06 2.06-.94-2.06-.94-.94-2.06-.94 2.06-2.06.94zm-11 1L8.5 8.5l.94-2.06 2.06-.94-2.06-.94L8.5 2.5l-.94 2.06-2.06.94zm10 10l.94 2.06.94-2.06 2.06-.94-2.06-.94-.94-2.06-.94 2.06-2.06.94z"/><path d="M17.41 7.96l-1.37-1.37c-.4-.4-.92-.59-1.43-.59-.52 0-1.04.2-1.43.59L10.3 9.45l-7.72 7.72c-.78.78-.78 2.05 0 2.83L4 21.41c.39.39.9.59 1.41.59.51 0 1.02-.2 1.41-.59l7.78-7.78 2.81-2.81c.8-.78.8-2.07 0-2.86zM5.41 20L4 18.59l7.72-7.72 1.47 1.35L5.41 20z"/>
  </svg>
      </button>

  <style>
    .colab-df-container {
      display:flex;
      flex-wrap:wrap;
      gap: 12px;
    }

    .colab-df-convert {
      background-color: #E8F0FE;
      border: none;
      border-radius: 50%;
      cursor: pointer;
      display: none;
      fill: #1967D2;
      height: 32px;
      padding: 0 0 0 0;
      width: 32px;
    }

    .colab-df-convert:hover {
      background-color: #E2EBFA;
      box-shadow: 0px 1px 2px rgba(60, 64, 67, 0.3), 0px 1px 3px 1px rgba(60, 64, 67, 0.15);
      fill: #174EA6;
    }

    [theme=dark] .colab-df-convert {
      background-color: #3B4455;
      fill: #D2E3FC;
    }

    [theme=dark] .colab-df-convert:hover {
      background-color: #434B5C;
      box-shadow: 0px 1px 3px 1px rgba(0, 0, 0, 0.15);
      filter: drop-shadow(0px 1px 2px rgba(0, 0, 0, 0.3));
      fill: #FFFFFF;
    }
  </style>

      <script>
        const buttonEl =
          document.querySelector('#df-a4c36cfb-26a6-4086-9f06-e4b9c901edb7 button.colab-df-convert');
        buttonEl.style.display =
          google.colab.kernel.accessAllowed ? 'block' : 'none';

        async function convertToInteractive(key) {
          const element = document.querySelector('#df-a4c36cfb-26a6-4086-9f06-e4b9c901edb7');
          const dataTable =
            await google.colab.kernel.invokeFunction('convertToInteractive',
                                                     [key], {});
          if (!dataTable) return;

          const docLinkHtml = 'Like what you see? Visit the ' +
            '<a target="_blank" href=https://colab.research.google.com/notebooks/data_table.ipynb>data table notebook</a>'
            + ' to learn more about interactive tables.';
          element.innerHTML = '';
          dataTable['output_type'] = 'display_data';
          await google.colab.output.renderOutput(dataTable, element);
          const docLink = document.createElement('div');
          docLink.innerHTML = docLinkHtml;
          element.appendChild(docLink);
        }
      </script>
    </div>
  </div>




## Remove emails and newline characters


```python
# Convert to list
data = df.content.values.tolist()

# Remove Emails
data = [re.sub('\S*@\S*\s?', '', sent) for sent in data]

# Remove new line characters
data = [re.sub('\s+', ' ', sent) for sent in data]

# Remove distracting single quotes
data = [re.sub("\'", "", sent) for sent in data]

pprint(data[:1])
```

    <>:5: DeprecationWarning: invalid escape sequence \S
    <>:8: DeprecationWarning: invalid escape sequence \s
    <>:5: DeprecationWarning: invalid escape sequence \S
    <>:8: DeprecationWarning: invalid escape sequence \s
    <>:5: DeprecationWarning: invalid escape sequence \S
    <>:8: DeprecationWarning: invalid escape sequence \s
    <ipython-input-6-10af9153bd18>:5: DeprecationWarning: invalid escape sequence \S
      data = [re.sub('\S*@\S*\s?', '', sent) for sent in data]
    <ipython-input-6-10af9153bd18>:8: DeprecationWarning: invalid escape sequence \s
      data = [re.sub('\s+', ' ', sent) for sent in data]


    ['From: (wheres my thing) Subject: WHAT car is this!? Nntp-Posting-Host: '
     'rac3.wam.umd.edu Organization: University of Maryland, College Park Lines: '
     '15 I was wondering if anyone out there could enlighten me on this car I saw '
     'the other day. It was a 2-door sports car, looked to be from the late 60s/ '
     'early 70s. It was called a Bricklin. The doors were really small. In '
     'addition, the front bumper was separate from the rest of the body. This is '
     'all I know. If anyone can tellme a model name, engine specs, years of '
     'production, where this car is made, history, or whatever info you have on '
     'this funky looking car, please e-mail. Thanks, - IL ---- brought to you by '
     'your neighborhood Lerxst ---- ']


## Tokenize and Clean-up using gensim’s simple_preprocess()


```python
def sent_to_words(sentences):
    for sentence in sentences:
        yield(gensim.utils.simple_preprocess(str(sentence), deacc=True))  # deacc=True removes punctuations

data_words = list(sent_to_words(data))

print(data_words[:1])
```

    [['from', 'wheres', 'my', 'thing', 'subject', 'what', 'car', 'is', 'this', 'nntp', 'posting', 'host', 'rac', 'wam', 'umd', 'edu', 'organization', 'university', 'of', 'maryland', 'college', 'park', 'lines', 'was', 'wondering', 'if', 'anyone', 'out', 'there', 'could', 'enlighten', 'me', 'on', 'this', 'car', 'saw', 'the', 'other', 'day', 'it', 'was', 'door', 'sports', 'car', 'looked', 'to', 'be', 'from', 'the', 'late', 'early', 'it', 'was', 'called', 'bricklin', 'the', 'doors', 'were', 'really', 'small', 'in', 'addition', 'the', 'front', 'bumper', 'was', 'separate', 'from', 'the', 'rest', 'of', 'the', 'body', 'this', 'is', 'all', 'know', 'if', 'anyone', 'can', 'tellme', 'model', 'name', 'engine', 'specs', 'years', 'of', 'production', 'where', 'this', 'car', 'is', 'made', 'history', 'or', 'whatever', 'info', 'you', 'have', 'on', 'this', 'funky', 'looking', 'car', 'please', 'mail', 'thanks', 'il', 'brought', 'to', 'you', 'by', 'your', 'neighborhood', 'lerxst']]


## Lemmatization


```python
def lemmatization(texts, allowed_postags=['NOUN', 'ADJ', 'VERB', 'ADV']):
    """https://spacy.io/api/annotation"""
    texts_out = []
    for sent in texts:
        doc = nlp(" ".join(sent)) 
        texts_out.append(" ".join([token.lemma_ if token.lemma_ not in ['-PRON-'] else '' for token in doc if token.pos_ in allowed_postags]))
    return texts_out

# Initialize spacy 'en' model, keeping only tagger component (for efficiency)
# Run in terminal: python3 -m spacy download en
nlp = spacy.load('en_core_web_sm', disable=['parser', 'ner'])

# Do lemmatization keeping only Noun, Adj, Verb, Adverb
data_lemmatized = lemmatization(data_words, allowed_postags=['NOUN', 'ADJ', 'VERB', 'ADV'])

print(data_lemmatized[:2])
```

    ['s thing subject car nntp post host college park line wonder out there enlighten car see other day door sport car look late early call door really small addition front bumper separate rest body know tellme model name engine spec year production car make history info funky look car mail thank bring neighborhood lerxst', 'subject clock poll final call summary final call clock report keyword acceleration clock upgrade article line nntp post host fair number brave soul upgrade clock oscillator share experience poll send brief message detail experience procedure top speed attain cpu rate speed add card adapter heat sink hour usage day floppy disk functionality floppy especially request summarize next day so add network knowledge base do clock upgrade answer poll thank']



```python
data_lemmatized[:2]
```




    ['s thing subject car nntp post host college park line wonder out there enlighten car see other day door sport car look late early call door really small addition front bumper separate rest body know tellme model name engine spec year production car make history info funky look car mail thank bring neighborhood lerxst',
     'subject clock poll final call summary final call clock report keyword acceleration clock upgrade article line nntp post host fair number brave soul upgrade clock oscillator share experience poll send brief message detail experience procedure top speed attain cpu rate speed add card adapter heat sink hour usage day floppy disk functionality floppy especially request summarize next day so add network knowledge base do clock upgrade answer poll thank']



## Create the Document-Word matrix


```python
vectorizer = CountVectorizer(analyzer='word',       
                             min_df=10,                        # minimum reqd occurences of a word 
                             stop_words='english',             # remove stop words
                             lowercase=True,                   # convert all words to lowercase
                             token_pattern='[a-zA-Z0-9]{3,}',  # num chars > 3
                             # max_features=50000,             # max number of uniq words
                            )

data_vectorized = vectorizer.fit_transform(data_lemmatized)
```

## Check the Sparsicity


```python
# Materialize the sparse data
data_dense = data_vectorized.todense()

# Compute Sparsicity = Percentage of Non-Zero cells
print("Sparsicity: ", ((data_dense > 0).sum()/data_dense.size)*100, "%")
```

    Sparsicity:  0.8259859421715849 %


## Build LDA model with sklearn


```python
# Build LDA Model
lda_model = LatentDirichletAllocation(n_components=20,               # Number of topics
                                      max_iter=10,               # Max learning iterations
                                      learning_method='online',   
                                      random_state=100,          # Random state
                                      batch_size=128,            # n docs in each learning iter
                                      evaluate_every = -1,       # compute perplexity every n iters, default: Don't
                                      n_jobs = -1,               # Use all available CPUs
                                     )
lda_output = lda_model.fit_transform(data_vectorized)

print(lda_model)  # Model attributes
```

    LatentDirichletAllocation(learning_method='online', n_components=20, n_jobs=-1,
                              random_state=100)


#### Output of above code

```python
LatentDirichletAllocation(batch_size=128, doc_topic_prior=None,
             evaluate_every=-1, learning_decay=0.7,
             learning_method='online', learning_offset=10.0,
             max_doc_update_iter=100, max_iter=10, mean_change_tol=0.001,
             n_components=10, n_jobs=-1, n_topics=20, perp_tol=0.1,
             random_state=100, topic_word_prior=None,
             total_samples=1000000.0, verbose=0)
```

## Diagnose model performance with perplexity and log-likelihood


```python
# Log Likelyhood: Higher the better
print("Log Likelihood: ", lda_model.score(data_vectorized))

# Perplexity: Lower the better. Perplexity = exp(-1. * log-likelihood per word)
print("Perplexity: ", lda_model.perplexity(data_vectorized))

# See model parameters
pprint(lda_model.get_params())
```

    Log Likelihood:  -8153640.003229906
    Perplexity:  1721.0030040794254
    {'batch_size': 128,
     'doc_topic_prior': None,
     'evaluate_every': -1,
     'learning_decay': 0.7,
     'learning_method': 'online',
     'learning_offset': 10.0,
     'max_doc_update_iter': 100,
     'max_iter': 10,
     'mean_change_tol': 0.001,
     'n_components': 20,
     'n_jobs': -1,
     'perp_tol': 0.1,
     'random_state': 100,
     'topic_word_prior': None,
     'total_samples': 1000000.0,
     'verbose': 0}


## How to GridSearch the best LDA model?

The most important tuning parameter for LDA models is n_components (number of topics). In addition, I am going to search learning_decay (which controls the learning rate) as well.

Besides these, other possible search params could be learning_offset (downweigh early iterations. Should be > 1) and max_iter. These could be worth experimenting if you have enough computing resources.

Be warned, the grid search constructs multiple LDA models for all possible combinations of param values in the param_grid dict. So, this process can consume a lot of time and resources.


```python
# Define Search Param
search_params = {'n_components': [10, 15], 'learning_decay': [.5, .7, .9]}
# search_params = {'n_components': [10, 15, 20, 25, 30], 'learning_decay': [.5, .7, .9]}

# Init the Model
lda = LatentDirichletAllocation()

# Init Grid Search Class
model = GridSearchCV(lda, param_grid=search_params)

# Do the Grid Search
model.fit(data_vectorized)
```




    GridSearchCV(estimator=LatentDirichletAllocation(),
                 param_grid={'learning_decay': [0.5, 0.7, 0.9],
                             'n_components': [10, 15]})



#### Output of above cell

```python
GridSearchCV(cv=None, error_score=nan,
             estimator=LatentDirichletAllocation(batch_size=128,
                                                 doc_topic_prior=None,
                                                 evaluate_every=-1,
                                                 learning_decay=0.7,
                                                 learning_method='batch',
                                                 learning_offset=10.0,
                                                 max_doc_update_iter=100,
                                                 max_iter=10,
                                                 mean_change_tol=0.001,
                                                 n_components=10, n_jobs=None,
                                                 perp_tol=0.1,
                                                 random_state=None,
                                                 topic_word_prior=None,
                                                 total_samples=1000000.0,
                                                 verbose=0),
             iid='deprecated', n_jobs=None,
             param_grid={'learning_decay': [0.5, 0.7, 0.9],
                         'n_components': [10, 15]},
             pre_dispatch='2*n_jobs', refit=True, return_train_score=False,
             scoring=None, verbose=0)
```

#### Output of the above cell

```python
GridSearchCV(cv=None, error_score='raise',
       estimator=LatentDirichletAllocation(batch_size=128, doc_topic_prior=None,
             evaluate_every=-1, learning_decay=0.7, learning_method=None,
             learning_offset=10.0, max_doc_update_iter=100, max_iter=10,
             mean_change_tol=0.001, n_components=10, n_jobs=1,
             n_topics=None, perp_tol=0.1, random_state=None,
             topic_word_prior=None, total_samples=1000000.0, verbose=0),
       fit_params=None, iid=True, n_jobs=1,
       param_grid={'n_topics': [10, 15, 20, 25, 30], 'learning_decay': [0.5, 0.7, 0.9]},
       pre_dispatch='2*n_jobs', refit=True, return_train_score='warn',
       scoring=None, verbose=0)
```

## How to see the best topic model and its parameters?


```python
# Best Model
best_lda_model = model.best_estimator_

# Model Parameters
print("Best Model's Params: ", model.best_params_)

# Log Likelihood Score
print("Best Log Likelihood Score: ", model.best_score_)

# Perplexity
print("Model Perplexity: ", best_lda_model.perplexity(data_vectorized))
```

    Best Model's Params:  {'learning_decay': 0.7, 'n_components': 10}
    Best Log Likelihood Score:  -1718429.7833504635
    Model Perplexity:  1623.2083379818696



```python
model.best_score_
```




    -1718429.7833504635



## Compare LDA Model Performance Scores

Plotting the log-likelihood scores against num_topics, clearly shows number of topics = 10 has better scores. And learning_decay of 0.7 outperforms both 0.5 and 0.9.

This makes me think, even though we know that the dataset has 20 distinct topics to start with, some topics could share common keywords. For example, ‘alt.atheism’ and ‘soc.religion.christian’ can have a lot of common words. Same with ‘rec.motorcycles’ and ‘rec.autos’, ‘comp.sys.ibm.pc.hardware’ and ‘comp.sys.mac.hardware’, you get the idea.

To tune this even further, you can do a finer grid search for number of topics between 10 and 15. But I am going to skip that for now.

So the bottom line is, a lower optimal number of distinct topics (even 10 topics) may be reasonable for this dataset. I don’t know that yet. But LDA says so. Let’s see.


```python
model.cv_results_
```




    {'mean_fit_time': array([59.5211206 , 70.6821795 , 57.17428579, 68.53499546, 58.91123142,
            69.47348566]),
     'std_fit_time': array([1.16909945, 3.43938694, 1.11527192, 0.88950909, 1.22535391,
            1.40551226]),
     'mean_score_time': array([1.35757103, 1.41992121, 1.16648078, 1.45628572, 1.19785829,
            1.46151004]),
     'std_score_time': array([0.33321529, 0.0512939 , 0.05206522, 0.05800298, 0.03772571,
            0.08270252]),
     'param_learning_decay': masked_array(data=[0.5, 0.5, 0.7, 0.7, 0.9, 0.9],
                  mask=[False, False, False, False, False, False],
            fill_value='?',
                 dtype=object),
     'param_n_components': masked_array(data=[10, 15, 10, 15, 10, 15],
                  mask=[False, False, False, False, False, False],
            fill_value='?',
                 dtype=object),
     'params': [{'learning_decay': 0.5, 'n_components': 10},
      {'learning_decay': 0.5, 'n_components': 15},
      {'learning_decay': 0.7, 'n_components': 10},
      {'learning_decay': 0.7, 'n_components': 15},
      {'learning_decay': 0.9, 'n_components': 10},
      {'learning_decay': 0.9, 'n_components': 15}],
     'split0_test_score': array([-1661864.28936651, -1681440.45352782, -1661794.34965946,
            -1680467.77684547, -1667521.24032579, -1681832.46554077]),
     'split1_test_score': array([-1821173.09478781, -1839333.52668648, -1820550.24187333,
            -1838078.27695413, -1821410.68938757, -1835496.07837538]),
     'split2_test_score': array([-1738160.62309147, -1759126.7702443 , -1734442.5654753 ,
            -1763485.16115413, -1739873.23404458, -1765449.89325174]),
     'split3_test_score': array([-1699508.93309007, -1725588.93920468, -1698298.67027697,
            -1720025.11298924, -1702172.78527575, -1719942.22784881]),
     'split4_test_score': array([-1679003.29886908, -1694922.41882092, -1677063.08946725,
            -1697303.08884581, -1677818.40582847, -1698610.16792677]),
     'mean_test_score': array([-1719942.04784099, -1740082.42169684, -1718429.78335046,
            -1739871.88335775, -1721759.27097243, -1740266.1665887 ]),
     'std_test_score': array([56650.08512851, 56394.74510715, 56584.51142508, 56456.93413971,
            55704.06604602, 55255.29534103]),
     'rank_test_score': array([2, 5, 1, 4, 3, 6], dtype=int32)}




```python
grid_cv_model_result = pd.DataFrame(model.cv_results_)
```





  <div id="df-41313928-4028-469b-85ae-4d97730be1b5">
    <div class="colab-df-container">
      <div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>mean_fit_time</th>
      <th>std_fit_time</th>
      <th>mean_score_time</th>
      <th>std_score_time</th>
      <th>param_learning_decay</th>
      <th>param_n_components</th>
      <th>params</th>
      <th>split0_test_score</th>
      <th>split1_test_score</th>
      <th>split2_test_score</th>
      <th>split3_test_score</th>
      <th>split4_test_score</th>
      <th>mean_test_score</th>
      <th>std_test_score</th>
      <th>rank_test_score</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>59.521121</td>
      <td>1.169099</td>
      <td>1.357571</td>
      <td>0.333215</td>
      <td>0.5</td>
      <td>10</td>
      <td>{'learning_decay': 0.5, 'n_components': 10}</td>
      <td>-1.661864e+06</td>
      <td>-1.821173e+06</td>
      <td>-1.738161e+06</td>
      <td>-1.699509e+06</td>
      <td>-1.679003e+06</td>
      <td>-1.719942e+06</td>
      <td>56650.085129</td>
      <td>2</td>
    </tr>
    <tr>
      <th>1</th>
      <td>70.682179</td>
      <td>3.439387</td>
      <td>1.419921</td>
      <td>0.051294</td>
      <td>0.5</td>
      <td>15</td>
      <td>{'learning_decay': 0.5, 'n_components': 15}</td>
      <td>-1.681440e+06</td>
      <td>-1.839334e+06</td>
      <td>-1.759127e+06</td>
      <td>-1.725589e+06</td>
      <td>-1.694922e+06</td>
      <td>-1.740082e+06</td>
      <td>56394.745107</td>
      <td>5</td>
    </tr>
    <tr>
      <th>2</th>
      <td>57.174286</td>
      <td>1.115272</td>
      <td>1.166481</td>
      <td>0.052065</td>
      <td>0.7</td>
      <td>10</td>
      <td>{'learning_decay': 0.7, 'n_components': 10}</td>
      <td>-1.661794e+06</td>
      <td>-1.820550e+06</td>
      <td>-1.734443e+06</td>
      <td>-1.698299e+06</td>
      <td>-1.677063e+06</td>
      <td>-1.718430e+06</td>
      <td>56584.511425</td>
      <td>1</td>
    </tr>
    <tr>
      <th>3</th>
      <td>68.534995</td>
      <td>0.889509</td>
      <td>1.456286</td>
      <td>0.058003</td>
      <td>0.7</td>
      <td>15</td>
      <td>{'learning_decay': 0.7, 'n_components': 15}</td>
      <td>-1.680468e+06</td>
      <td>-1.838078e+06</td>
      <td>-1.763485e+06</td>
      <td>-1.720025e+06</td>
      <td>-1.697303e+06</td>
      <td>-1.739872e+06</td>
      <td>56456.934140</td>
      <td>4</td>
    </tr>
    <tr>
      <th>4</th>
      <td>58.911231</td>
      <td>1.225354</td>
      <td>1.197858</td>
      <td>0.037726</td>
      <td>0.9</td>
      <td>10</td>
      <td>{'learning_decay': 0.9, 'n_components': 10}</td>
      <td>-1.667521e+06</td>
      <td>-1.821411e+06</td>
      <td>-1.739873e+06</td>
      <td>-1.702173e+06</td>
      <td>-1.677818e+06</td>
      <td>-1.721759e+06</td>
      <td>55704.066046</td>
      <td>3</td>
    </tr>
    <tr>
      <th>5</th>
      <td>69.473486</td>
      <td>1.405512</td>
      <td>1.461510</td>
      <td>0.082703</td>
      <td>0.9</td>
      <td>15</td>
      <td>{'learning_decay': 0.9, 'n_components': 15}</td>
      <td>-1.681832e+06</td>
      <td>-1.835496e+06</td>
      <td>-1.765450e+06</td>
      <td>-1.719942e+06</td>
      <td>-1.698610e+06</td>
      <td>-1.740266e+06</td>
      <td>55255.295341</td>
      <td>6</td>
    </tr>
  </tbody>
</table>
</div>
      <button class="colab-df-convert" onclick="convertToInteractive('df-41313928-4028-469b-85ae-4d97730be1b5')"
              title="Convert this dataframe to an interactive table."
              style="display:none;">

  <svg xmlns="http://www.w3.org/2000/svg" height="24px"viewBox="0 0 24 24"
       width="24px">
    <path d="M0 0h24v24H0V0z" fill="none"/>
    <path d="M18.56 5.44l.94 2.06.94-2.06 2.06-.94-2.06-.94-.94-2.06-.94 2.06-2.06.94zm-11 1L8.5 8.5l.94-2.06 2.06-.94-2.06-.94L8.5 2.5l-.94 2.06-2.06.94zm10 10l.94 2.06.94-2.06 2.06-.94-2.06-.94-.94-2.06-.94 2.06-2.06.94z"/><path d="M17.41 7.96l-1.37-1.37c-.4-.4-.92-.59-1.43-.59-.52 0-1.04.2-1.43.59L10.3 9.45l-7.72 7.72c-.78.78-.78 2.05 0 2.83L4 21.41c.39.39.9.59 1.41.59.51 0 1.02-.2 1.41-.59l7.78-7.78 2.81-2.81c.8-.78.8-2.07 0-2.86zM5.41 20L4 18.59l7.72-7.72 1.47 1.35L5.41 20z"/>
  </svg>
      </button>

  <style>
    .colab-df-container {
      display:flex;
      flex-wrap:wrap;
      gap: 12px;
    }

    .colab-df-convert {
      background-color: #E8F0FE;
      border: none;
      border-radius: 50%;
      cursor: pointer;
      display: none;
      fill: #1967D2;
      height: 32px;
      padding: 0 0 0 0;
      width: 32px;
    }

    .colab-df-convert:hover {
      background-color: #E2EBFA;
      box-shadow: 0px 1px 2px rgba(60, 64, 67, 0.3), 0px 1px 3px 1px rgba(60, 64, 67, 0.15);
      fill: #174EA6;
    }

    [theme=dark] .colab-df-convert {
      background-color: #3B4455;
      fill: #D2E3FC;
    }

    [theme=dark] .colab-df-convert:hover {
      background-color: #434B5C;
      box-shadow: 0px 1px 3px 1px rgba(0, 0, 0, 0.15);
      filter: drop-shadow(0px 1px 2px rgba(0, 0, 0, 0.3));
      fill: #FFFFFF;
    }
  </style>

      <script>
        const buttonEl =
          document.querySelector('#df-41313928-4028-469b-85ae-4d97730be1b5 button.colab-df-convert');
        buttonEl.style.display =
          google.colab.kernel.accessAllowed ? 'block' : 'none';

        async function convertToInteractive(key) {
          const element = document.querySelector('#df-41313928-4028-469b-85ae-4d97730be1b5');
          const dataTable =
            await google.colab.kernel.invokeFunction('convertToInteractive',
                                                     [key], {});
          if (!dataTable) return;

          const docLinkHtml = 'Like what you see? Visit the ' +
            '<a target="_blank" href=https://colab.research.google.com/notebooks/data_table.ipynb>data table notebook</a>'
            + ' to learn more about interactive tables.';
          element.innerHTML = '';
          dataTable['output_type'] = 'display_data';
          await google.colab.output.renderOutput(dataTable, element);
          const docLink = document.createElement('div');
          docLink.innerHTML = docLinkHtml;
          element.appendChild(docLink);
        }
      </script>
    </div>
  </div>





```python
grid_cv_model_result.columns
grid_cv_model_result["mean_test_score"]
```




    0   -1.719942e+06
    1   -1.740082e+06
    2   -1.718430e+06
    3   -1.739872e+06
    4   -1.721759e+06
    5   -1.740266e+06
    Name: mean_test_score, dtype: float64




```python
# Get Log Likelyhoods from Grid Search Output
n_topics = [10, 15, 20, 25, 30]
log_likelyhoods_5 = [round(gscore.mean_validation_score) for gscore in model.cv_results_["params"] if gscore.params['learning_decay']==0.5]
log_likelyhoods_7 = [round(gscore.mean_validation_score) for gscore in model.cv_results_["params"] if gscore.params['learning_decay']==0.7]
log_likelyhoods_9 = [round(gscore.mean_validation_score) for gscore in model.cv_results_["params"] if gscore.params['learning_decay']==0.9]

# Show graph
plt.figure(figsize=(12, 8))
plt.plot(n_topics, log_likelyhoods_5, label='0.5')
plt.plot(n_topics, log_likelyhoods_7, label='0.7')
plt.plot(n_topics, log_likelyhoods_9, label='0.9')
plt.title("Choosing Optimal LDA Model")
plt.xlabel("Num Topics")
plt.ylabel("Log Likelyhood Scores")
plt.legend(title='Learning decay', loc='best')
plt.show()
```


    ---------------------------------------------------------------------------

    AttributeError                            Traceback (most recent call last)

    <ipython-input-32-84537683e6f8> in <module>
          1 # Get Log Likelyhoods from Grid Search Output
          2 n_topics = [10, 15, 20, 25, 30]
    ----> 3 log_likelyhoods_5 = [round(gscore.mean_validation_score) for gscore in model.cv_results_["params"] if gscore.params['learning_decay']==0.5]
          4 log_likelyhoods_7 = [round(gscore.mean_validation_score) for gscore in model.cv_results_["params"] if gscore.params['learning_decay']==0.7]
          5 log_likelyhoods_9 = [round(gscore.mean_validation_score) for gscore in model.cv_results_["params"] if gscore.params['learning_decay']==0.9]


    <ipython-input-32-84537683e6f8> in <listcomp>(.0)
          1 # Get Log Likelyhoods from Grid Search Output
          2 n_topics = [10, 15, 20, 25, 30]
    ----> 3 log_likelyhoods_5 = [round(gscore.mean_validation_score) for gscore in model.cv_results_["params"] if gscore.params['learning_decay']==0.5]
          4 log_likelyhoods_7 = [round(gscore.mean_validation_score) for gscore in model.cv_results_["params"] if gscore.params['learning_decay']==0.7]
          5 log_likelyhoods_9 = [round(gscore.mean_validation_score) for gscore in model.cv_results_["params"] if gscore.params['learning_decay']==0.9]


    AttributeError: 'dict' object has no attribute 'params'


## How to see the dominant topic in each document?

To classify a document as belonging to a particular topic, a logical approach is to see which topic has the highest contribution to that document and assign it.
In the table below, I’ve greened out all major topics in a document and assigned the most dominant topic in its own column.




```python
# Create Document - Topic Matrix
lda_output = best_lda_model.transform(data_vectorized)

# column names
topicnames = ["Topic" + str(i) for i in range(best_lda_model.n_components)]

# index names
docnames = ["Doc" + str(i) for i in range(len(data))]

# Make the pandas dataframe
df_document_topic = pd.DataFrame(np.round(lda_output, 2), columns=topicnames, index=docnames)

# Get dominant topic for each document
dominant_topic = np.argmax(df_document_topic.values, axis=1)
df_document_topic['dominant_topic'] = dominant_topic

# Styling
def color_green(val):
    color = 'green' if val > .1 else 'black'
    return 'color: {col}'.format(col=color)

def make_bold(val):
    weight = 700 if val > .1 else 400
    return 'font-weight: {weight}'.format(weight=weight)

# Apply Style
df_document_topics = df_document_topic.head(15).style.applymap(color_green).applymap(make_bold)
df_document_topics
```

## Review topics distribution across documents


```python
df_topic_distribution = df_document_topic['dominant_topic'].value_counts().reset_index(name="Num Documents")
df_topic_distribution.columns = ['Topic Num', 'Num Documents']
df_topic_distribution
```

## How to visualize the LDA model with pyLDAvis?


```python
pyLDAvis.enable_notebook()
panel = pyLDAvis.sklearn.prepare(best_lda_model, data_vectorized, vectorizer, mds='tsne')
panel
```

## How to see the Topic’s keywords?

The weights of each keyword in each topic is contained in lda_model.components_ as a 2d array. The names of the keywords itself can be obtained from vectorizer object using get_feature_names().

Let’s use this info to construct a weight matrix for all keywords in each topic.


```python
# Topic-Keyword Matrix
df_topic_keywords = pd.DataFrame(best_lda_model.components_)

# Assign Column and Index
df_topic_keywords.columns = vectorizer.get_feature_names()
df_topic_keywords.index = topicnames

# View
df_topic_keywords.head()
```

## Get the top 15 keywords each topic


```python
# Show top n keywords for each topic
def show_topics(vectorizer=vectorizer, lda_model=lda_model, n_words=20):
    keywords = np.array(vectorizer.get_feature_names())
    topic_keywords = []
    for topic_weights in lda_model.components_:
        top_keyword_locs = (-topic_weights).argsort()[:n_words]
        topic_keywords.append(keywords.take(top_keyword_locs))
    return topic_keywords

topic_keywords = show_topics(vectorizer=vectorizer, lda_model=best_lda_model, n_words=15)        

# Topic - Keywords Dataframe
df_topic_keywords = pd.DataFrame(topic_keywords)
df_topic_keywords.columns = ['Word '+str(i) for i in range(df_topic_keywords.shape[1])]
df_topic_keywords.index = ['Topic '+str(i) for i in range(df_topic_keywords.shape[0])]
df_topic_keywords
```

## How to predict the topics for a new piece of text?

Assuming that you have already built the topic model, you need to take the text through the same routine of transformations and before predicting the topic.

For our case, the order of transformations is:

`sent_to_words() –> lemmatization() –> vectorizer.transform() –> best_lda_model.transform()`

You need to apply these transformations in the same order. So to simplify it, let’s combine these steps into a predict_topic() function.


```python
def predict_topic(text, nlp=nlp):
    global sent_to_words
    global lemmatization

    # Step 1: Clean with simple_preprocess
    mytext_2 = list(sent_to_words(text))

    # Step 2: Lemmatize
    mytext_3 = lemmatization(mytext_2, allowed_postags=['NOUN', 'ADJ', 'VERB', 'ADV'])

    # Step 3: Vectorize transform
    mytext_4 = vectorizer.transform(mytext_3)

    # Step 4: LDA Transform
    topic_probability_scores = best_lda_model.transform(mytext_4)
    topic = df_topic_keywords.iloc[np.argmax(topic_probability_scores), :].values.tolist()
    return topic, topic_probability_scores

# Predict the topic
mytext = ["Some text about christianity and bible"]
topic, prob_scores = predict_topic(text = mytext)
print(topic)
```


```python
print(topic)
```

## How to cluster documents that share similar topics and plot?

You can use k-means clustering on the document-topic probabilioty matrix, which is nothing but lda_output object. Since out best model has 15 clusters, I’ve set n_clusters=15 in KMeans().

Alternately, you could avoid k-means and instead, assign the cluster as the topic column number with the highest probability score.

We now have the cluster number. But we also need the X and Y columns to draw the plot.

For the X and Y, you can use SVD on the lda_output object with n_components as 2. SVD ensures that these two columns captures the maximum possible amount of information from lda_output in the first 2 components.


```python
# Construct the k-means clusters
from sklearn.cluster import KMeans
clusters = KMeans(n_clusters=15, random_state=100).fit_predict(lda_output)

# Build the Singular Value Decomposition(SVD) model
svd_model = TruncatedSVD(n_components=2)  # 2 components
lda_output_svd = svd_model.fit_transform(lda_output)

# X and Y axes of the plot using SVD decomposition
x = lda_output_svd[:, 0]
y = lda_output_svd[:, 1]

# Weights for the 15 columns of lda_output, for each component
print("Component's weights: \n", np.round(svd_model.components_, 2))

# Percentage of total information in 'lda_output' explained by the two components
print("Perc of Variance Explained: \n", np.round(svd_model.explained_variance_ratio_, 2))
```


```python
# Plot
plt.figure(figsize=(12, 12))
plt.scatter(x, y, c=clusters)
plt.xlabel('Component 2')
plt.xlabel('Component 1')
plt.title("Segregation of Topic Clusters", )
```

## How to get similar documents for any given piece of text?


```python
from sklearn.metrics.pairwise import euclidean_distances

nlp = spacy.load('en', disable=['parser', 'ner'])

def similar_documents(text, doc_topic_probs, documents = data, nlp=nlp, top_n=5, verbose=False):
    topic, x  = predict_topic(text)
    dists = euclidean_distances(x.reshape(1, -1), doc_topic_probs)[0]
    doc_ids = np.argsort(dists)[:top_n]
    if verbose:        
        print("Topic KeyWords: ", topic)
        print("Topic Prob Scores of text: ", np.round(x, 1))
        print("Most Similar Doc's Probs:  ", np.round(doc_topic_probs[doc_ids], 1))
    return doc_ids, np.take(documents, doc_ids)

```


```python
# Get similar documents
mytext = ["Some text about christianity and bible"]
doc_ids, docs = similar_documents(text=mytext, doc_topic_probs=lda_output, documents = data, top_n=1, verbose=True)
print('\n', docs[0][:500])
```
