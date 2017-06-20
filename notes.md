## Information Bottleneck 

Low dimension Representation of input. 

## Deep Information Bottleneck
Stochastic encoding Z.

Markov Chain Y <-> X <-> Z. $P(X, Y, Z) = P(X)P(Y|X)P(Z|X)$. Z does not depend on labels.

Intermediate layer: $p(z | x, \theta)$ maximally informative with contraint on how much information it caries from input.

Parametrise information bottleneck using neural network.

## Variational Inference for Mutual Information

Challenging to calculate mutual information exept X, Y, Z all discrete (counting) or all gaussians.

Variational inference to get lower bound.

## Sampling
Use reparametrisation and Monte Carlo sampling to estimate gradient.

$$ p(y|x) \approx \frac{1}{S} \sum_s q(y|z^s)$$ 

Where $z^s \sim p(z|x)$ and S is set to 1 or 12.

This allows us to use gradient descent and therefore use a neural net to parametrise distribution.

## Encoder
Stochastic parametric encoder: multivariate gaussian

$$p(z|x) = N(z|f_e^{\mu}(x), f_e^{\Sigma}(x))$$ Where f is a multilayer perceptron (feed forward neural network).

Dimension of $\mu$ and $\Sigma$ to tune how much compression we want. We use a diagonal gaussian (another big approximation so that we don't have to calculate inverse)

Having some variance helps as the network will need to learn from a wider range of cases and being less dependant on specific inputs.

## Approximations
Decoder: $$ p(y|z) \approx q(y|z) = S(y|Wz+b)$$ logistic regression or neural net...

$$ p(z) = \int p(z|x)p(x) \approx r(z) \sim N(0, 1) $$ this is the regulariser. This isn't the prior, this is a big approximation. Could you richer marginal approximations.

## Reparametrization trick

$$ p(z|x)dz = p(\epsilon) d\epsilon$$ where $z = f(x, \epsilon)$ deterministic. Stochasticity comes from $\epsilon$ only and that is an input (no need to probagate through it).

## Special cases

$\beta \to 0$: no regularisation or compression => deterministic: $f^\Sigma_e(x) \to 0$

