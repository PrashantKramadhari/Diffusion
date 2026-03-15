# <center> DPM-Solver

<div align=center>

[KAIST CS492(C): Diffusion and Flow Models (Fall 2025)](https://mhsung.github.io/kaist-cs492c-fall-2025/)

Programming Assignment 2: DPM-Solver

Instructor: [Minhyuk Sung](https://mhsung.github.io) (mhsung [at] kaist.ac.kr)

TA: [Juil Koo](https://63days.github.io/) (63days [at] kaist.ac.kr)

<img src="https://drive.google.com/uc?id=1f4TzEneFq5WVnWFGmUUBiwe-QhDCkyVR"></img>
</div>

## Abstract
DPM-Solver is a novel ordinary differential equation (ODE) solver specifically designed to solve PF-ODEs in just a few steps by exploiting their special structure. The key insight is that the PF-ODE of diffusion models can be decomposed into a linear part and a nonlinear part, a property known as the semi-linear structure. With this decomposition, one can compute the exact solution of the linear part while approximating only the nonlinear part using a Taylor expansion. Increasing the order of the Taylor expansion leads to more accurate solutions. Notably, the authors show that the first-order DPM-Solver is mathematically equivalent to DDIM.


<center>

**Detailed submission guidelines and grading criteria are provided at the end of the document.**

</center>

## Quick Summary of DPM-Solver

<center>
You can skip this summary and go directly to the implementation part below if you are already familiar with these models.
</center>

In this assignment, we explore a high-order solver of the ODEs for diffusion models. The more accurate the solver is, the faster the sampling can be done with fewer steps. Let's say the forward pass of diffusion models is represented as follows:

$$
\begin{align*}
q(x_t | x_0) = \mathcal{N}(x_t | \alpha(t) x_0, \sigma^2(t) I).
\end{align*}
$$

The forward pass can be expressed its corresponding SDE form:

$$
dx_t = f(t) x_t dt + g(t) dw_t, \\
\text{where } f(t) = \frac{d \log \alpha_t}{dt} \text{ and } g^2(t) = \frac{d\sigma_t^2}{dt} - 2 \frac{d \log \alpha_t}{dt}\sigma_t^2.
$$

According to [Song et al. [ICLR'21]](https://arxiv.org/abs/2011.13456), we can also compute the ODE form of the _reverse_ process:

$$
\begin{align*}
dx_t = [f(t) x_t + \frac{g^2(t)}{2 \sigma_t} \epsilon_\theta(x_t, t) ] dt,
\end{align*}
$$

which is called Probability Flow ODE (PF-ODE). Note that this ODE equation is a first-order linear non-homogeneous equation, for which exact solution can be computed:

$$
\begin{align*}
x_t = e^{\int_s^t f(\tau) d\tau} x_s + \int_s^t ( e^{\int_\tau^t f(r) dr} \frac{g^2(\tau)}{2 \sigma_\tau} \epsilon_\theta(x_\tau, \tau) ) d\tau,
\end{align*}
$$

where the first term on the right side is a linear term, and the second term, which involves an integral over noise predictions, is a non-linear term.

To simplify the ODE solution equaiton, we introduce a new notation, $\lambda_t := \log (\alpha_t / \sigma_t)$, and use "change-of-variable" for $\lambda$. Then, we have:

$$
\begin{align*}
x_t = \frac{\alpha_t}{\alpha_s} x_s - \alpha_t \int\_{\lambda_s}^{\lambda_t} e^{-\lambda} \hat{\epsilon}\_\theta (\hat{x}_\lambda, \lambda) d\lambda,
\end{align*}
$$

where $\hat{x}\_\lambda := x\_{t(\lambda)}$ and $\hat{\epsilon}\_\theta (\hat{x}_\lambda, \lambda):= \epsilon\_\theta (x\_{t(\lambda)}, t(\lambda))$.


Now the simplified solution reveals the integral is represented as the _exponentially weighted integral_ of $\hat{\epsilon}_\theta$, which is closely related to the _exponential integrators_ commonly discussed in the literature of ODE solvers.

We can apply a Taylor expansion into the solution:

$$
\begin{align*}
\hat{\epsilon}\_\theta (\hat{x}\_\lambda, \lambda) = \sum_{n=0}^{k-1} \frac{(\lambda - \lambda\_{s})^n}{n!} \hat{\epsilon}_\theta^{(n)} (\hat{x}\_{\lambda\_s}, \lambda_s) + \mathcal{O}((\lambda - \lambda_s)^k),
\end{align*}
$$

where $\hat{\epsilon}\_\theta^{(n)}$ is the $n$-th order derivative of $\hat{\epsilon}_\theta$.


As a result, we can obtain an approximation for $x_s \rightarrow x_t$ with the $k$-order approximation. For instance, as $k=1$, which is called DPM-Solver-1, the solution is:

$$
\begin{align*}
x_t = \frac{\alpha_t}{\alpha_s} x_s - \sigma_t (e^{h}-1) \epsilon_\theta (x_s, s), \text{ where } h = \lambda_t - \lambda_s.
\end{align*}
$$

## What to Submit

<details>
<summary><b>Submission Item List</b></summary>
</br>

- [ ] Jupyter notebook file

- [ ] Loss curve screenshot
- [ ] Chamfer Distance (screenshot) for DPM-Solver-1
- [ ] Sampling visualization for DPM-Solver-1

- [ ] Chamfer Distance (screenshot) for DPM-Solver-2
- [ ] Sampling visualization for DPM-Solver-2

</details>


Submit a single ZIP file named `{NAME}_{STUDENT_ID}.zip` containing the two followings:

1. Jupyter notebook file containing your code implementation.
2. A single PDF document named `{NAME}_{STUDENT_ID}.pdf` that includes:
    - Your name and student ID
    - All results listed in the submission item list above (screenshots, metrics, visualizations, etc.)


## Grading

**You will receive a zero score if:**
- **you do not submit,**
- **your code is not reproducible, or**
- **you modify any code outside of the section enclosed with `TODO` or use different hyperparameters that are supposed to be fixed as given.**

**Plagiarism in any form will also result in a zero score and will be reported to the university.**

**Your score will incur a 10% deduction for each missing item.**

Otherwise, you will receive up to 20 points from this assignment that count toward your final grade.

- Task 1 (DPM-Solver-1)
  - 10 points: Achieve CD lower than **40**.
  - 5 points: Achieve CD between **40** and **80**.
  - 0 point: otherwise.
- Task 2 (DPM-Solver-2)
  - 10 points: Achieve CD lower than **40**.
  - 5 points: Achieve CD between **40** and **80**.
  - 0 point: otherwise.


## Further Readings

If you are interested in this topic, we encourage you to check out the further materials below.

- [DPM-Solver: A Fast ODE Solver for Diffusion Probabilistic Model Sampling in Around 10 Steps](https://arxiv.org/abs/2206.00927)
- [DPM-Solver++: Fast Solver for Guided Sampling of Diffusion Probabilistic Models](https://arxiv.org/abs/2211.01095)
- [Score-Based Generative Modeling through Stochastic Differential Equations](https://arxiv.org/abs/2011.13456)
- [Elucidating the Design Space of Diffusion-Based Generative Models](https://arxiv.org/abs/2206.00364)


For $k \leq 2$, Taylor expansion needs additional intermediate points between $t$ and $s$. Below is an algorithm when $k=2$, called DPM-Solver-2.

<img src="https://drive.google.com/uc?id=1uqAbHbVC-UoUoGlnf_hbElay0uXswt84"></img>
