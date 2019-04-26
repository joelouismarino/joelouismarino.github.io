---
title: 'Statistical Whitening'
date: 2017-08-01
permalink: /posts/2017/08/statistical_whitening/
tags:
  - whitening
---

Normalization is a fundamental component of machine learning. Take any introductory machine learning course, and you'll learn about the importance of normalizing the inputs to your model. The justification goes something like this: the important patterns in the data often correspond to the <i>relative</i> relationships between the different input dimensions. Therefore, you can make the task of learning and recognizing these patterns easier by removing the constant offset and standardizing the scales.

There have been a number of recent advances in deep learning related to normalization. As just a sampling, the often cited [AlexNet](https://papers.nips.cc/paper/4824-imagenet-classification-with-deep-convolutional-neural-networks.pdf)convolutional neural network used local response normalization, [batch normalization](https://arxiv.org/pdf/1502.03167.pdf) is a powerful technique for normalizing the activations within a network, [layer normalization](https://arxiv.org/pdf/1607.06450.pdf) is another form of normalization suited for recurrent networks, [weight normalization](https://arxiv.org/pdf/1602.07868.pdf) is a way to normalize the weights within a network, and [normalizing flows](https://arxiv.org/pdf/1505.05770.pdf) is a method for building flexible probability distributions, often using normalization operations.

Another interesting aspect about normalization is that it is one area where machine learning and neuroscience seem to agree. Normalization appears to be ubiquitous in the brain, often implemented in the form of [lateral inhibition](https://en.wikipedia.org/wiki/Lateral_inhibition). This is the process by which activity in one neuron inhibits activity in nearby neurons and vice versa, effectively shifting their overall activations to highlight their relative differences. For example, [horizontal cells](https://en.wikipedia.org/wiki/Retina_horizontal_cell) in the retina inhibit neighboring photoreceptor neurons, acting to sharpen the input and allowing the eye to adjust to different lighting levels. Likewise, lateral inhibition appears to be a key component of other sensory processing pathways as well as processing in the [cerebral cortex](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC3236361/), where inhibitory interneurons constitute roughly 20% of the neural population.

I find this commonality of normalization between machine learning and neuroscience encouraging, as it offers the opportunity to find connections between these two fields. Studying normalization in nervous systems may provide suggestions for ways to improve machine learning models, and studying normalization in machine learning models may provide a better theoretical understanding of normalization in nervous systems. I recently decided to learn more about normalization, in particular, **statistical whitening**. In the rest of this blog post, I'll share what I learned along with code for implementing some examples.

### what is normalization?

In the realm of statistics, normalization refers to taking a distribution and transforming it into a standard *normal* (i.e. Gaussian) distribution. For instance, imagine we have samples from the distribution of some variable $X$ defined along one real-valued dimension. This could be height measurements, house prices, sensor readings, etc. To get a normalized version of this distribution, $Z$, we subtract off the mean, $\mu$, and divide by the standard deviation, $\sigma$.

$$ Z = \frac{X - \mu}{\sigma} $$

Note that we're *broadcasting* $\mu$ and $\sigma$ to match the number of samples in $X$, and the division is performed element-wise. In the example below, we draw 500 samples from a Gaussian-distributed variable centered at $15$ with a standard deviation of $5$, then normalize. The resulting normalized distribution has a mean of $0$ and a standard deviation of $1$. From the histograms, we see that the normalized variable's distribution is simply a shifted and scaled replica of the original variable's distribution.

<script src="https://gist.github.com/joelouismarino/b7f6645792899f2b08c1b7639578e486.js"></script>
