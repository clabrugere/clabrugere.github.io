---
title: Debugging neural networks
date: 2023-10-29
categories: [Machine Learning, Tips]
tags: [debugging]
math: true
---

This post compiles some tips to build strong baseline models and how to avoid bugs and propose methods to find them when designing a new machine learning system that uses a neural network. They are mostly inspired by the gold mine that is Karparthy's blog, as well as some other sources or Stack-overflow posts.

In the design process, we usually can expect to face more or less the same failure modes:

- implementation bug, be it in the training loop or evaluation method or the model itself (such as shape mismatches),
- an architecture not adapted to the prediction task (using a MLP instead of convolutional network for image modalities),
- bad training hyperparameters like a learning rate too high,
- unadapted weights initialization,
- issues with the datasets such as not enough data, noisy data, class imbalance, biased validation dataset, different training and test distributions.

While not really formalized, we can apply a systematic process to minimize the chance of occurrence of those, or at least minimize the time spent detecting these failure modes and quickly find the relevant solutions. First by probing the model to look for different issues in order to build a simple, bug-free model and training loop, and then improve the base implementation to reach the targeted performance.

## Building a robust baseline

### Data

- When building the data loader object, loop through it to make sure batches of samples have the expected shape, for both input tensors and labels. Moreover, check the signature of the loss function you use to ensure the label's shape is compatible with it. For instance a loss function can expect labels to be of shape `(batch size, 1)` but your data loader samples labels with shape `(batch size,)`.

- Make sure the labels and inputs are aligned in the batch sampled from the data loader during the training process. To do so simply check that the tuples (input, label) on a small batch from the data loader are what you would expect. This mis-alignment can happen if you shuffle inputs and labels independently when building a batch for example.

|               Original dataset               |               Mis-aligned batches                |
| :------------------------------------------: | :----------------------------------------------: |
| (x0, y0)<br>(x1, y1)<br>(x2, y2)<br>(x3, y3) | { (x0, y1), (x1, y2) }<br>{ (x2, y0), (x3, y3) } |

- Make sure inputs are properly normalized: they should take values in $[0, 1]$, have zero mean and unit variance for continuous inputs to avoid saturating activation functions (leading to vanishing gradients). Features should have the same scale to avoid bias in the learning process: if feature 1 has an order of magnitude 1.0 but feature 2 has order of magnitude 1000.0, then gradient components corresponding to feature 2 will be much larger and hence the gradient descent will be biased towards one direction.

- Explore and visualize the inputs you pass to the model to make sure they’re as expected: it will help to detect bugs on transforms and data augmentations.

### Architecture

- Starts with a simple architecture and avoids fancy activations, initialization schemes and optimization algorithms (that have most likely been developed for very specific use cases). Stick with He normal initialization, ReLu and Adam.

- Don’t put regularization initially to make it easier to overfit on a small set of data points.

- For deep networks or if you see vanishing gradients, you can add ‘skip-connections’ parallel to the main trainable block of layers, to build residual neural networks. It allows to flow gradients from the loss down to the input and addresses vanishing gradients.

- Log every intermediate results shapes and invoke the forward pass on a small batch of random samples to make sure shapes are correct. Because of broadcasting rules, you can have silent failure modes when evaluating operations on tensors with the wrong shapes. Some causes of incorrect shapes can be: wrong dimension in reduce operations (sum, mean, …) or softmax, wrong dimensions in transpose/permute operations, forgot to unsqueeze/squeeze.

### Training

- Fix random seeds to make sure running the same code twice results in the same result, it makes debugging easier because it excludes some of the randomness from the picture.

```python
# for pytorch
def set_seed(seed: int):
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

# for tensorflow
def set_seed(seed: int):
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)
    tf.config.experimental.enable_op_determinism()
```

- Start with a small training and validation dataset to allow for faster iterations. Only start training on the full dataset when you are confident the model training is stable and behave as expected.

- Avoid using methods that are not necessary for a basic training (for instance LR scheduling, quantization, data augmentations, etc…) in order to reduce the size of the search space of potential bugs.

- Check that the initial loss makes sense: for instance if the cross-entropy loss is used, the initial loss should be close to $\log(\frac{1}{\text{num classes}})$ if the last layer is initialized correctly ( "default" values can be derived for other losses):

$$
\begin{split}
\text{CE}(y, \hat{y}) & = - \sum_{c \in \mathcal{C}} y_c \log (\hat{y}_c) \\
& = - \sum_{c \in \mathcal{C}} y_c \log (\frac{1}{\text{num classes}}) \\
& = - \log(\frac{1}{\text{num classes}})
\end{split}
$$

- Include some prior knowledge in the initialization scheme: in a regression task, if you know the mean of the target is around some value $\mu$, you should initialize the bias of the last layer to $\mu$. If it is a classification task and you know you have a 100:1 class imbalance, set the bias of the logits such that the initial output probability is 0.1. It will help the training converge faster.

- Fit the model on a set of samples with constant values (such as tensors of zeros). This data-independent model should perform worse than the one fitted on the real data and if it does, it’ll indicate that the model is able to extract useful information for the learning task.

- Fit a simple, low capacity model to a small dataset with real data, then increase the model’s capacity (by increasing the number of layers for example). The loss should decrease, if not then there is probably a bug somewhere.

- Overfit on a batch of a few samples: with a large enough capacity the model should converge to the global minimum loss (e.g zero). If it does it indicates that the model is able to extract useful information. If the loss increases, your loss might be ill-defined (flipped sign), learning rate too high, numerical instabilities, inputs and labels are not aligned.

- Log the predictions of a small validation batch during training to understand their dynamics. They should become more and more stable (that is not that different from one epoch to another) if the model is learning correctly and in a stable fashion. It also helps understand if the learning rate order of magnitude is good: a small learning rate will make the predictions dynamics change slowly without an apparent lower bound in view while a large learning rate will make the predictions oscillate during the training.

- Log gradients to understand the dependencies between model's weights. For instance let’s say you introduced a bug where information is mixed batch-wise (mixing information from different samples by using a `reshape` operation instead of a `transpose`). By defining a simple loss like the sum of outputs for a fixed sample and logging gradients of the first operation, you will see non-zero gradients elsewhere than the input 0 in the batch, indicating that there is some dependencies between inputs. (note that you need to remove layers that do some batch-wise normalization with trainable weights as it introduces dependencies between samples but it is not a bug):

For pytorch:

```python
# random input tensor for which we want to calculate the gradients during the backpropagation
x = torch.rand((64, 128), requires_grad=True)

logits = model(x)

# define a simple loss function that depends on only one sample
loss = logits[0].sum()

# backpropagate and calculate gradients
loss.backward()

# they sould be zero for every index != 0 and not zero otherwise
assert((x.grad[1:] == 0.0).all() and (x.grad[0] != 0.0).any())
```

For tensorflow:

```python
x = tf.Variable(tf.random.uniform((16, 64)))
model = tf.keras.Sequential([
    tf.keras.layers.Dense(32, activation="relu"),
    tf.keras.layers.Dense(1)
])

# compute gradients of the input w.r.t loss
with tf.GradientTape() as tape:
    logits = model(x)
    loss = tf.reduce_sum(logits[0])

grad = tape.gradient(loss, x)

# they sould be zero for every index != 0 and not zero otherwise
assert(tf.reduce_all(grad[1:] == 0.0) & tf.reduce_any(grad[0] != 0.0))
```

- Logging gradient norms at different places can help diagnose infinite values caused by overflows due to exploding gradients. Even without overflow, plotting gradient norms of the different layers will help detect and fix unstable training before it actually diverges if you see gradient norm spikes. In that case weight decay can help as it will add a constraint on the weight's magnitude.

- Probing weight norms is also a good way to detect unstable training: a seemingly weight norm that keeps increasing can be an early sign of divergent training. Just like the point above, adding or increasing weight decay will probably help.

```python
# for pytorch
# compute weights and their gradients norm during a forward-backward pass
names, param_norms, grad_norms = [], [], []
for name, param in in model.named_parameters():
    if param.requires_grad and "bias" not in name:
        names.append(name)
        param_norms.append(param.detach().cpu().norm())
        grad_norms.append(param.grad.detach().cpu().norm())

# plot
fig, axes = plt.subplots(nrows=2, sharex=True, figsize=(20, 20))
axes[0].bar(x=names, height=param_norms)
axes[0].set_ylabel("weight norm")

axes[1].bar(x=names, height=grad_norms)
axes[1].set_ylabel("gradient norm")
axes[1].tick_params(axis="x", labelrotation=90)

```

- Logging intermediate values will help detect numerical instabilities. Those can happen when using functions that can easily under/overflow, such as log or exponential functions.

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

- Profile the training loop in order to ensure you properly use your hardware: you can use the command `watch -n1 nvidia-smi` to continuously monitor GPU usage. If GPU utilization is low, it is most likely because the data loading process is the bottleneck: it takes too long, is not parallelized or there is no prefetching while the GPU is busy.
