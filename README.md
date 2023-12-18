# omml-project

# 1. Introduction

Usually we perform stochastic learning on high-dimensional input and usually only few features from that input is non-zero. It is often the case that infrequent features are  highly informative and discriminative. The proposed idea is to give small learning rate to frequent features and high rate to rare ones.

## 1.1 The Adaptive Gradient Algorithm

Chapter introduces notations used in the work.

![Untitled](Project%20504f710ab17b443da55cc9c10ec371a8/Untitled.png)

The goal of the study is to  “devise algorithms which are guaranteed to suffer asymptotically sub-linear regret, namely, Rφ(T) = o(T). “

In the study 2 algorithms us used to minimize regret:

- Nesterov’s primal-dual subgradient method
- Regularized dual averaging, and the follow-the-regularized-leader (FTRL) family of algorithms

## 1.2 Outline of Results

## 1.3 Improvements and Motivating Example

An example of the case, where features are sparse. Analysis shows that AdaGRAD can be exponentially smaller in the dimension d than the non-adaptive regret bound. 

ADAGRADs regret:

$$
O(max{logd,d^{1−α/2}}√T).
$$

And online gradient decent regret:

$$
O(√dT).
$$

### 1.3.1 DIAGONAL ADAPTATION

## 1.4 Related Work

# 2. Adaptive Proximal Functions

Regret mostly depends on dual norms of f′t(xt), and the dual norms in turn depend on the choice of ψ. So, to decrease regret we need to change ψ dynamically, during the run

![Untitled](Project%20504f710ab17b443da55cc9c10ec371a8/Untitled%201.png)

# 3. Diagonal Matrix Proximal Functions

As discussed in the introduction, Algorithm 1 should have lower regret than non-adaptive algorithms on sparse data, though this depends on the geometry of the underlying optimization space
X.

E.g if our learning problem is a logistic regression with 0/1-valued features, then the gradient terms in the bound 

$$
\sum_{i=1}^d||g_1:_T,_i||_2
$$

should all be much smaller than √T. 

If some features appear much more frequently than others, then the infimal representation of γT and the infimal equality in Corollary 1 show that we have significantly lower regret by using higher learning rates for infrequent features and lower learning rates on commonly appearing features. Further, if the optimal predictor is relatively dense, as is often the case in predictions problems with sparse inputs, then 

$$
||x^∗||^∞
$$

is the best p-norm we can have in the regret

# 4. Full Matrix Proximal Functions

In this section we derive and analyze new updates when we estimate a full matrix for the divergence ψt instead of a diagonal one. In this generalized case, we use the root of the matrix of outer products of the gradients that we have observed to update our parameters

As in the diagonal case, we build on intuition garnered from an optimization problem, and in particular, we seek a matrix S which is the solution to the following minimization problem:

![Untitled](Project%20504f710ab17b443da55cc9c10ec371a8/Untitled%202.png)

The solution is obtained by defining 

$$
G_t = \sum^{t}_{τ=1}
g_τg_τ^T
$$

and setting S to be a normalized version of the root of G_T , that is, 

$$
S = cG^{1/2}_T /tr(G^{1/2}_T).
$$

![Untitled](Project%20504f710ab17b443da55cc9c10ec371a8/Untitled%203.png)

# 5. Adam

Adam( Adaptive Moment Estimation)

Adam is a child of Adagrad. Adam takes 3 parameters:

- α - learning rate
- β1 - heavy-ball style momentum
- β2 - controller the decay

Taking β1 = 0, β2 = 1 and αn = α gives Adagrad.

The main idea is to find learning rate for each parameter of the network

In expectation, the squared norm of the objective gradient averaged over the trajectory has an upper-bound which is explicit in the constants of the problem, parameters of the optimizer, the dimension d, and the total number of iterations N.
This bound can be made arbitrarily small, and with the right hyper-parameters, Adam can converge with the rate of convergence O(d ln(N)/√N)

## 5.1 **Key-points**

In expectation, the squared norm of the objective gradient averaged over the trajectory has an upper-bound which is explicit in the constants of the problem, parameters of the optimizer, the dimension d, and the total number of iterations N.
This bound can be made arbitrarily small, and with the right hyper-parameters, Adam can converge with the rate of convergence O(d ln(N)/√N)

While Adagrad is asymptotically optimal, it also lead to a slower decrease of the term proportional to F (x0) − F∗, as 1/√N instead of 1/N for Adam

Adam will not converge with its default parameters. It is however possible to choose α and β2 to achieve an critical point for arbitrarily small and, for a known time horizon, they can be chosen to obtain the exact same bound as Adagrad.

## 5.2 **Experiments. Toy example**

We take (Q_i)i∈[6], Bernoulli variables with P [Q_i = 1] = p_i

![Untitled](Project%20504f710ab17b443da55cc9c10ec371a8/Untitled%204.png)

10^6 iterations with batch size 1 when varying either

α, 1 − β1 or 1 − β2 through a range of 13 values uniformly spaced in log-scale between 10^−6 and 1. When varying

- α, we take β1 = 0 and β2 = 1 − 10−6.
- β1, we take α = 10−5 and β2 = 1 − 10−6 (i.e. β2 is so that we are in the Adagrad-like regime).
- β2, we take β1 = 0 and α = 10−6.

## 5.2.1 Experiments. Toy example. Results

![Untitled](Project%20504f710ab17b443da55cc9c10ec371a8/Untitled%205.png)

![Untitled](Project%20504f710ab17b443da55cc9c10ec371a8/Untitled%206.png)

![Untitled](Project%20504f710ab17b443da55cc9c10ec371a8/Untitled%207.png)

## 5.3 **Experiments. CIFAR-10 CNN**

- α - 10^−6 and 10^−2 with 9 values,
- 1 − β1 the range is from 10^−5 to 0.3 with 9 values,
- 1 − β2, from 10^−6 to 10^−1 with 9 values
- batch size 128
- CNN model for object classification

## 5.3.1 Experiments. **CIFAR-10 CNN.** Results

![Untitled](Project%20504f710ab17b443da55cc9c10ec371a8/Untitled%208.png)

![Untitled](Project%20504f710ab17b443da55cc9c10ec371a8/Untitled%209.png)

![Untitled](Project%20504f710ab17b443da55cc9c10ec371a8/Untitled%2010.png)

# 6. AdaGRAD

![Untitled](Project%20504f710ab17b443da55cc9c10ec371a8/Untitled%2011.png)

![Untitled](Project%20504f710ab17b443da55cc9c10ec371a8/Untitled%2012.png)

![Untitled](Project%20504f710ab17b443da55cc9c10ec371a8/Untitled%2013.png)

![Untitled](Project%20504f710ab17b443da55cc9c10ec371a8/Untitled%2014.png)

![Untitled](Project%20504f710ab17b443da55cc9c10ec371a8/Untitled%2015.png)

![Untitled](Project%20504f710ab17b443da55cc9c10ec371a8/Untitled%2016.png)

![Untitled](Project%20504f710ab17b443da55cc9c10ec371a8/Untitled%2017.png)

![Untitled](Project%20504f710ab17b443da55cc9c10ec371a8/Untitled%2018.png)
