# Autonomous Direct Preference Optimization (DPO) & Self-Play Alignment Engine

## 1. Overview
The **Alignment Subsystem** implements closed-form **Direct Preference Optimization (DPO)** coupled with **Autonomous Reasoning Self-Play (RLAIF)**. It allows the core to align model policy towards sound reasoning and away from logical contradictions without needing an external reward model or human intervention.

---

## 2. Mathematical Formulation

### Implicit Reward Function
Under the Bradley-Terry preference model, DPO reparameterizes the reward function directly in terms of the policy $\pi_\theta$ and frozen reference policy $\pi_{\text{ref}}$:
$$r_\theta(x, y) = \beta \log \frac{\pi_\theta(y \mid x)}{\pi_{\text{ref}}(y \mid x)}$$

### DPO Loss Function
The objective minimizes:
$$\mathcal{L}_{\text{DPO}}(\theta; \pi_{\text{ref}}) = -\mathbb{E}_{(x, y_w, y_l)} \left[ \log \sigma \left( \beta \log \frac{\pi_\theta(y_w \mid x)}{\pi_{\text{ref}}(y_w \mid x)} - \beta \log \frac{\pi_\theta(y_l \mid x)}{\pi_{\text{ref}}(y_l \mid x)} \right) \right]$$

With conservative label smoothing $\epsilon$:
$$\mathcal{L}_{\text{DPO}} = -(1 - \epsilon) \log \sigma(u) - \epsilon \log \sigma(-u)$$
where $u = r_\theta(x, y_w) - r_\theta(x, y_l)$.

### Analytical Autograd Gradients
To avoid numerical underflow or $\log(0)$ divergence, the gradient is computed analytically in the backward pass:
$$\nabla_{\pi_\theta(y_w)} \mathcal{L}_{\text{DPO}} = -\frac{\beta}{B} \sigma(-u)$$
$$\nabla_{\pi_\theta(y_l)} \mathcal{L}_{\text{DPO}} = +\frac{\beta}{B} \sigma(-u)$$

This guarantees monotonic probability increase for verified reasoning paths ($y_w$) and probability suppression for contradictory paths ($y_l$), scaled dynamically by the model's error $\sigma(-u)$.

---

## 3. Autonomous Reasoning Self-Play Cycle

```
                       +-------------------------------+
                       |  Prompt / Problem Statement   |
                       +---------------+---------------+
                                       |
                                       v
                       +-------------------------------+
                       | ReasoningSearchEngine (MCTS)  |
                       |    - PUCT Exploration         |
                       |    - Beam Thought Expansion   |
                       +---------------+---------------+
                                       |
                     +-----------------+-----------------+
                     |                                   |
                     v                                   v
             [ Candidate Trace 1 ]               [ Candidate Trace 2 ]
                     |                                   |
                     +-----------------+-----------------+
                                       |
                                       v
                       +-------------------------------+
                       |     SelfVerifier Engine       |
                       |  - Contradiction Detection    |
                       |  - N-Gram Loop Penalty        |
                       |  - Deductive Bonus            |
                       +---------------+---------------+
                                       |
                                       v
                +---------------------------------------------+
                | Identify:                                   |
                |   y_w = Valid / Conclusive Trace (Score > T)|
                |   y_l = Contradictory / Suboptimal Trace    |
                +----------------------+----------------------+
                                       |
                                       v
                       +-------------------------------+
                       |     DPOTrainer Engine         |
                       |  - Policy Forward Pass        |
                       |  - Reference Forward Pass     |
                       |  - Fused LogSumExp CE         |
                       |  - Dynamic Loss Scaling (AMP) |
                       |  - AdamW Optimizer Step       |
                       +-------------------------------+
```

---

## 4. Verification and CLI Usage

### Run Test Suite
```cmd
.\build_ninja\bin\aic_tests.exe
```

### Execute Self-Play DPO Alignment Session
```cmd
.\build_ninja\bin\aic.exe align 10 --amp
```
