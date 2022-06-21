# DPART


The **dpart** framework is a generalized and flexible approach to building an effective DP generative model.
The overall training flow relies on an autoregressive generative model [(Fig.1)](__link_here) and could be broken down to two main steps.
First, identifying/specifying a visit order or a prediction matrix that describes how the overall joint distribution is broken down to a series of lower-dimensional conditionals.
In other words, if the dataset $`D`$ is a collection of $`k`$-dimensional datapoints $`x`$ then:

```math
$P(\boldsymbol{x}) = \prod_{i=1}^{k} P(x_{i}|x_1,x_2,...,x_{i-1}) = \prod_{i=1}^{k} P(x_{i}|\boldsymbol{x}_{<i})$
```

Second, given the series of conditionals, they are sequentially estimated by fitting a predictive model (a sampler method).
In order to generate synthetic data, the fitted sampler methods are used to generate one column at a time given previously already generated columns.
