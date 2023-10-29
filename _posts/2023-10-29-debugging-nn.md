---
title: Debugging neural networks
date: 2023-10-29
categories: [Machine Learning, Tips]
tags: [debugging]
math: true
---

This post compiles some tips to build strong baseline models and how to avoid bugs and propose methods to find them when designing a new machine learning system that uses a neural network. They are mostly taken from Karparthy's blog and some other sources or Stack-overflow posts.

In the design process, we usually can expect to face more or less the same failure modes:

- implementation bug, be it in the training loop or evaluation method or the model itself,
- an architecture not adapted to the prediction task,
- bad training hyperparameters,
- unadapted weights initialization,
- issues with the datasets such as not enough data, noisy data, class imbalance, biased validation dataset, different training and test distributions.

While not really formalized, we can apply a systematic process to minimize the chance of occurrence of those, or at least minimize the time spent detecting these failure modes and quickly find the relevant solutions. First by probing the model to look for different issues in order to build a simple, bug-free model and training loop, and then improve the base implementation to reach the targeted performance.

## Building a robust baseline

- Fix random seeds to make sure running the same code twice results in the same result, it makes debugging easier because it excludes some of the randomness from the picture. TODO: snippet setting seeds

- Starts with a simple architecture and avoids fancy activations, initialization schemes and optimization algorithms (that have most likely been developed for very specific use cases). Stick with He normal initialization, ReLu and Adam.

- Make sure the labels and inputs are aligned in the batch sampled from the data loader during the training process. To do so simply check that the tuples (input, label) on a small batch from the data loader are what you would expect.

- Make sure inputs are properly normalized: they should take values in $[0, 1]$, have zero mean and unit variance for continuous inputs to avoid saturating activation functions (leading to vanishing gradients). Features should have the same scale to avoid bias in the learning process: if feature 1 has an order of magnitude 1.0 but feature 2 has order of magnitude 1000.0, then gradient components corresponding to feature 2 will be much larger.

- Start with a small training and validation dataset to allow for faster iterations. Only start training on the full dataset when you are confident the model training is stable and behave as expected.

- Avoid using methods that are not necessary for a basic training (for instance LR scheduling, quantization, data augmentations, etc…) in order to reduce the size of the search space of potential bugs.

- Check that the initial loss makes sense: for instance in a multi-class task, the initial loss should be close to $\log(\frac{1}{\text{num classes}})$ after the softmax if the last layer is initialized correctly.

- Include some prior knowledge in the initialization scheme: in a regression task, if you know the mean of the target is around some value x, you should initialize the bias of the last layer to x. If it is a classification task and you know you have a 100:1 class imbalance, set the bias of the logits such that the initial output probability is 0.1. It will help the training converge faster.

- Fit the model on a set of samples with constant values (such as tensors of zeros). This data-independent model should perform worse than the one fitted on the real data and if it does, it’ll indicate that the model is able to extract useful information for the learning task.

- Overfit on a batch of a few samples: with a large enough capacity the model should converge to the global minimum loss (e.g zero). If it does it indicates that the model is able to extract useful information. If the loss increases, your loss might be ill-defined (flipped sign), learning rate too high, numerical instabilities, inputs and labels are not aligned.

- Fit a simple, low capacity model to a small dataset with real data, then increase the model’s capacity (by increasing the number of layers for example). The loss should decrease, if not then there is probably a bug somewhere.

- Explore and visualize the inputs you pass to the model to make sure they’re as expected: it will help to detect bugs on transforms and data augmentations.

- In particular, log every intermediate results shapes and invoke the forward pass on a small batch of random samples to make sure shapes are correct. Because of broadcasting rules, you can have silent failure modes when evaluating operations on tensors with the wrong shapes. Some causes of incorrect shapes can be: wrong dimension in reduce operations (sum, mean, …) or softmax, wrong dimensions in transpose/permute operations, forgot to unsqueeze/squeeze.

- Log the predictions of a small validation batch during training to understand their dynamics. They should become more and more stable (that is not that different from one epoch to another) if the model is learning correctly and in a stable fashion. It also helps understand if the learning rate order of magnitude is good: a small learning rate will make the predictions dynamics change slowly without an apparent lower bound in view while a large learning rate will make the predictions oscillate during the training.

- Log gradients to understand the dependencies in the model. For instance let’s say you introduced a bug where information is mixed batch-wise (mixing information from different samples by using a `reshape` operation instead of a `transpose`). By defining a simple loss like the sum of outputs for a fixed sample and logging gradients of the first operation, you will see non-zero gradients elsewhere than the input 0 in the batch, indicating that there is some dependencies between inputs. (note that you need to remove layers that do some batch-wise normalization with trainable weights as it introduces dependencies between samples but it is not a bug):

```python
# random input tensor for which we want to calculate the gradients during the backpropagation
x = torch.rand([64, 128], requires_grad=True)

logits = model(x)

# define a simple loss function that depends on only one sample
loss = logits[0].sum()

# backpropagate and calculate gradients
loss.backward()

# check the gradients
assert((x.grad[1:] == 0.0).all() and (x[0] != 0.0).any())
```

- Infinites values can show up because of exploding gradients or when using operations that can over/underflow (such as log, exp functions). Logging gradient norms can diagnose the former and logging intermediate results can detect the latter.

- Plot loss curves as it allows to get a quick glance of the learning dynamics: does the model properly converge? Do you see large spikes? Are you overfitting at some point? Is the validation loss close to the training loss?

- You can decompose the error as: $\text{total error} = \text{irreducible error} + \text{training error (bias)} + \text{validation error (variance)} + \text{validation overfitting}$. Model capacity/complexity addresses the bias, regularizing the variance. But sometimes (not necessarily true for deep learning models), increasing the regularization can degrade the training error. When experimenting, it is a good practice to log the contributions of each error mode across experiments to better determine where to act.

![error decomposition](/assets/img/posts/debugging-nn/error-decomposition.png)

## Improving the baseline

It can be simplified to a two steps process: increase the model capacity to decrease the loss until the model overfits, then introduce regularization to trade some of the training loss in favor of validation loss.

- The easiest regularization that doesn’t involve changing the training process nor the model architecture is to simply get more data to build a larger training dataset. It effectively allows to increase the performance of a well designed model without risk.

- Reduce input dimensionality to either remove strongly correlated features or features that are not that informative or add too much detail/noise. It will reduce the model size allowing to train on more data with the same budget.

- Add dropout layers.

- Add and play around with the weight decay strength.

- Add early stopping to stop the training when the model starts to overfit the training dataset (that requires a good validation dataset).

- Introduce learning rate schedulers.
