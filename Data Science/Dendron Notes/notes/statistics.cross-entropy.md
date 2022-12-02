---
id: ef1zsa9sznmvyig56da7c8q
title: Cross Entropy
desc: ''
updated: 1669462559547
created: 1669460393573
---


### What Is Cross-Entropy?

Cross-entropy is a measure of the difference between two probability distributions for a given random variable or set of events.


In information theory, we like to describe the “surprise” of an event. An event is more surprising the less likely it is, meaning it contains more information.

* Low Probability Event (surprising): More information.
* Higher Probability Event (unsurprising): Less information.

Information h(x) can be calculated for an event x, given the probability of the event P(x) as follows:

$$
h(x) = -\log(P(x))
$$

Entropy is the number of bits required to transmit a randomly selected event from a probability distribution. A skewed distribution has a low entropy, whereas a distribution where events have equal probability has a larger entropy.

A skewed probability distribution has less “surprise” and in turn a low entropy because likely events dominate. Balanced distribution are more surprising and turn have higher entropy because events are equally likely.

* Skewed Probability Distribution (unsurprising): Low entropy.
* Balanced Probability Distribution (surprising): High entropy.

Entropy H(x) can be calculated for a random variable with a set of x in X discrete states discrete states and their probability P(x) as follows:

$$
H(X) = – \sum  P(x) * \log (P(x))
$$

Cross-entropy builds upon the idea of entropy from information theory and calculates the number of bits required to represent or transmit an average event from one distribution compared to another distribution.