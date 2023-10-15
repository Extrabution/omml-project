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
