# ELTRA: An Embedding Method based on Learning-to-Rank to Preserve Asymmetric Information in Directed Graphs

This repository provides (1) a reference implementation of ELTRA, (2) the data, and (3) the proof for CRW scores properties.

## Installation and usage
ELTRA is a novel similarity-based double-vector **E**mbedding method based on listwise Learning-**T**o-**R**ank (LTR) that preserves **A**symmetric information in directed graphs. It is straightforward embedding method implemented by a simple deep neural network consisting of only a projection layer and an output layer.

This repository will be updated soon. 

Thank you

## CRW Scores Properties
In this Appendix, we prove that the CRW scores are _asymmetric_, _bounded_, _monotonic_, _unique_, and _always existent_.

**(1) Asymmetry: for every node-pair $`(u,v)`$ where $`u\!\neq \!v`$, $`S(u,v)\! \neq \!S(v,u)`$.**

***Proof* :** According to Equation (2), if $`u\!\neq\!v`$, $`S_k(u,v)`$ is computed by considering $`O_u`$ and $`I_u`$, while $`S_k(v,u)`$ is computed by considering $`O_v`$ and $`I_v`$; since $`O_u \!\neq\! O_v`$ and $`I_u \!\neq\! I_v`$, then $`S(u,v) \!\neq\! S(v,u)`$

**(2) Bounding: for all $`k`$,  $`0 \!\le \!S_k(u,v) \!\le\! 1`$.**

***Proof* :** As mentioned in Section 3.1.2, if $`u\! \neq \!v`$, then $`S_0(u,v)\!=\!0`$, otherwise $`S_0(u,v)\!=\!1`$; therefore $`0 \!\le \!S_0(u,v)\! \le \!1`$. It means the property holds for $`k\!=\!0`$. Now, we assume that the property holds for $`k`$ such that $`0\! \le\! S_k(u,v)\! \le\! 1`$ for *any* node-pairs $`(u,v)`$ and we prove that the property also holds for $`k\!+\!1`$ as follows. If $`u\! \neq \!v`$, $`S_{k+1}(u,v)`$ is computed by Equation (2) and according to our assumption, we know that $`S_k(u,v) \!\ge\! 0`$, thus

```math
\text{(1) } S_{k+1}(u,v) \! =  \! \frac{C}{2}\! \cdot\! \big(\frac{\sum_{i \in O_u} S_{k}(i,v)}{|O_u|}\! +\! \frac{\sum_{j \in I_u}S_{k}(j,v)}{|I_u|}  \big) \\
```
```math
\ge  \! \frac{C}{2}\! \cdot\! \big(\frac{\sum_{i \in O_u} (0)}{|O_u|}\! +\! \frac{\sum_{j \in I_u}(0)}{|I_u|}  \big)
\ge \! 0
```
also, according to the assumption, we have $`S_k(u,v) \!\le \!1`$, thus

```math
\text{(2) } S_{k+1}(u,v) \! =  \! \frac{C}{2}\! \cdot\! \big(\frac{\sum_{i \in O_u} S_{k}(i,v)}{|O_u|}\! +\! \frac{\sum_{j \in I_u}S_{k}(j,v)}{|I_u|}  \big)
```
```math
\le  \! \frac{C}{2}\! \cdot\! \big(\frac{\sum_{i \in O_u} (1)}{|O_u|}\! +\! \frac{\sum_{j \in I_u} (1)}{|I_u|}  \big)
```
```math
\le \! \frac{C}{2}\! \cdot\! \big(1 +1  \big) \le \! C
```
since $`0 \!< \!C\! <\! 1`$, then $`S_{k+1}(u,v) \!< \!1`$. (1) and (2) denote that $`0\! \le\! S_{k+1}(u,v)\! <\! 1`$. Also, we know that if $`u\!=\!v`$, $`S_{k+1}(u,v)\!=\!1`$. Therefore, $`0\! \le\! S_{k+1}(u,v)\! \le \! 1`$.

**(3) Monotonicity: for every node-pair $`(u,v)`$, the sequence \{$`S_0(u,v), S_1(u,v), ..., S_k(u,v)`$\} is non-decreasing as $`k`$ increases.**

***Proof*:** If $`u\!=\!v`$, $`S_0(u,v) \!=\! S_1(u,v) \!=\! \cdots \!=\! 1`$; thus, the property holds. If $`u \!\neq\! v`$, according to Equation (2), $`S_0(u,v)\!=\!0`$ and by the bounding property, $`0 \!\le S_1(u,v)\! \le 1`$; therefore, $`S_0(u,v)\! \le \!S_1(u,v)`$, which means the property holds for $`k\!=\!0`$. We assume that the property holds for all $`k`$ where $`S_{k-1}(u,v) \! \le \!S_k(u,v) `$ for *any* node-pairs $`(u,v)`$, which means $`S_k(u,v)\!-\! S_{k-1}(u,v) \!\ge\! 0`$; we show the property holds for $`k\!+\!1`$ as follows:
```math
S_{k+1}(u,v) \!-\! S_{k}(u,v)\! =  \! \frac{C}{2}\! \cdot\! \big(\frac{\sum_{i \in O_u} S_{k}(i,v)}{|O_u|}\! +\! \frac{\sum_{j \in I_u}S_{k}(j,v)}{|I_u|}  \big) \!-\! \frac{C}{2}\! \cdot\! \big(\frac{\sum_{i \in O_u} S_{k-1}(i,v)}{|O_u|}\! +\! \frac{\sum_{j \in I_u}S_{k-1}(j,v)}{|I_u|}  \big)
```
```math
= \! \frac{C}{2}\! \cdot\! \big(\frac{\sum_{i \in O_u} S_{k}(i,v)-S_{k-1}(i,v)}{|O_u|}\! +\! \frac{\sum_{j \in I_u}S_{k}(j,v)-S_{k-1}(j,v)}{|I_u|}  \big)
```
```math
\ge \frac{C}{2}\! \cdot\! \big(\frac{\sum_{i \in O_u} (0)}{|O_u|}\! +\! \frac{\sum_{j \in I_u}(0)}{|I_u|}  \big) \!=\!0
```

according to the assumptions, $`S_k(u,v)\! -\! S_{k-1}(u,v) \!\ge\! 0`$ and we already know that $`C\! > \!0`$ therefore, $`S_{k+1}(u,v) \!- \!S_k(u,v) \!\ge\! 0`$, which means $`S_{k+1}(u,v) \!\ge\! S_k(u,v)`$.

**(4) Existence: the fixed point $`S(*,*)`$ of the CRW equation always exists.**

***Proof*:** By the bounding and monotonicity properties, for any node-pairs $`(u,v)`$, $`S_k(u,v)`$ is bounded and non-decreasing as $`k`$ increases. A sequence $`S_k(u,v)`$ converges to $`\lim S(u,v) \!\in\! [0,1]`$, according to the Completeness Axiom of calculus. $`\displaystyle\lim_{k\to\infty} S_{k+1}(u,v) \!=\! \displaystyle\lim_{k\to\infty} S_k(u,v)\! =\! S(u,v)`$ and the limit of a sum is identical to the sum of the limits, therefore

```math
S(u,v) = \displaystyle \lim_{k\to\infty} S_{k+1} =   \! \frac{C}{2}\! \cdot\! \big(\frac{\displaystyle \lim_{k\to\infty} \sum_{i \in O_u} S_{k}(i,v)}{|O_u|}\! +\! \frac{\displaystyle \lim_{k\to\infty} \sum_{j \in I_u}S_{k}(j,v)}{|I_u|}  \big)
```
```math
=   \! \frac{C}{2}\! \cdot\! \big(\frac{\sum_{i \in O_u} \displaystyle \lim_{k\to\infty}  S_{k}(i,v)}{|O_u|}\! +\! \frac{ \sum_{j \in I_u} \displaystyle \lim_{k\to\infty} S_{k}(j,v)}{|I_u|}  \big)
```
```math
=   \! \frac{C}{2}\! \cdot\! \big(\frac{\sum_{i \in O_u} S(i,v)}{|O_u|}\! +\! \frac{ \sum_{j \in I_u} S(j,v)}{|I_u|}  \big) = S(u,v)
```

**(5) Uniqueness: the solution for the fixed-point $`S(*,*)`$ is always unique.**

***Proof*:** Suppose that $`S(*,*)`$ and $`S^\prime(*,*)`$ are two solutions for the CRW equation. Also, for *all* node-pairs $`(u,v)`$, let $`\delta(u,v) \!=\! S(u,v) - S^\prime(u,v)`$ be the difference between these two solutions. Let $`M\!=\!\max\limits_{(u,v)} |\delta(u,v)|`$ be the maximum absolute value of all differences observed for some nod-pairs $`(u,v)`$ (i.e., $`|\delta(u,v)| \!= \!M)`$. We need to prove that $`M\!=\!0`$. If $`u\!=\!v`$, $`M\!=\!0`$ since $`S(u,v)\! =\! S^\prime(u,v) \!=\! 1`$. If $`u \!\neq \!v`$, $`S(u,v)`$ and $`S^\prime(u,v)`$ are computed by CRW, and we have

```math
\delta(u,v) = S(u,v) - S^\prime(u,v)
```
```math
= \; \! \frac{C}{2}\! \cdot\! \big(\frac{\sum_{i \in O_u} S(i,v)-S^\prime(i,v)}{|O_u|}\! +\! \frac{\sum_{j \in I_u}S(j,v)-S^\prime(j,v)}{|I_u|}  \big)
```
```math
= \; \! \frac{C}{2}\! \cdot\! \big(\frac{\sum_{i \in O_u} \delta(i,v)\!}{|O_u|} +\! \frac{\sum_{j \in I_u}\delta(j,v)}{|I_u|} \big)
```
thus,

```math
M = |\delta(u,v)| = \Bigg|  \frac{C}{2}\! \cdot\! \big(\frac{\sum_{i \in O_u} \delta(i,v)}{|O_u|}\! +\! \frac{\sum_{j \in I_u}\delta(j,v)}{|I_u|} \big) \Bigg|
```
```math
\le  \frac{C}{2}\! \cdot\! \big(\frac{\sum_{i \in O_u} |\delta(i,v)|}{|O_u|}\! +\! \frac{\sum_{j \in I_u}|\delta(j,v)|}{|I_u|} \big)
```
```math
 \le  \frac{C}{2}\! \cdot\! \big(\frac{\sum_{i \in O_u} M}{|O_u|}\! +\! \frac{\sum_{j \in I_u}M}{|I_u|} \big)
```
```math
 = \frac{C}{2} \! \cdot\! \big(M\!+\!M) \!=\! C \!\cdot \!M
```
Since $`0\! <\! C \!< \!1`$, surely $`M\!=\!0`$.


## Citation:
> Masoud Reyhani Hamedani, Jin-Su Ryu, and Sang-Wook Kim. 2023. An Embedding Method based on Learning-to-Rank to Preserve Asymmetric Information in Directed Graphs. In Proceedings of the 32th ACM International Conference on Information and Knowledge Management, October 2023, Pages xx–xx. https://doi.org/10.1145/3583780.3614862


