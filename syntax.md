
## Variational Inference

Approxmimating intractable integrals to get posterior or evidence.

### Variational Bayes (evidence)

Alternative to Gibbs sampling
$$ Q(Y, \Theta) \approx Q(Y)Q(\Theta)$$

### Parametrisation (posterior)

Parameterise $q = q_{\rho}(Y)$

Fully factored (mean field): $q(Y) = \prod q(Y_i)$

## Sampling

### Monte Carlo

$$ \int F(x)p(x) \approx \frac{1}{T} \sum F(x^{(t)})$$ 

Where $x^{(t)} \sim p$ but p can be hard to sample from.

### Markov Chain Monte Carlo

Choose transition $T(x \to x^{'})$ to ensure $p_{\infty} = p^{*}$

### Gibbs Sampling

For multivariate distribution: sample in turn $p(x_i|x \backslash i)$

## Stochastic Encoding

Choose according to distribution instead of single value: parameters important.

## Generative Adversial Network

Try to pass as a real image. Discrimitativ network: real/fake from generative network.

## Variational Auto-Encoder

Force latent vector to follow standard gaussian distribution.

## Information Theory
### Mutual Information
How similar is the joing distribution to the product of marginals

$$ I(Z, Y; \theta) = \int p(z, y|\theta) log \frac{p(z, y|\theta)}{p(z|\theta)p(y|\theta)}$$
$$ I(X, Y) = KL(p(x, y) || p(x)p(y))$$

$$ I(X, Y) = H(X) - H(X|Y)$$ $H(X|Y)$: what Y does not say about X. 

If independent $p(z, y) = p(z)p(y)$ => $I(Z, Y) = 0$

If all information shared $I(X, Y) = H(X) = H(Y)$

### Entropy
$$ H(X) = - \int p(x) log(p(x)) = \langle log(p(x) \rangle_{p(x)}$$ uncertainty about random variable.

## Dynamic Bayesian Networks
Like RNN but stochastic states, generalization of hidden Markov models and Kalman filters.

## Deep Belief Networks
Multiple layers of hidden variables

## (Restricted) Boltzmann Machine
Same as Sigmoid Belief Network but connections on both sides

## Hopfield Network
Fully connected recurrent network