# Autonomous Reasoning Engine (Tree-of-Thoughts & MCTS)

## 1. Overview
In standard language model inference, tokens are generated greedily or via nucleus sampling token-by-token along a single linear trajectory:
$$P(y_1, y_2, \dots, y_T \mid x) = \prod_{t=1}^T P(y_t \mid x, y_{<t})$$
This has catastrophic failure modes for multi-step reasoning, mathematical proof, logic puzzles, and planning: an early error cannot be corrected and propagates forward.

The **Autonomous Reasoning Engine** (`include/reasoning/`, `src/reasoning/`) introduces a tree-based cognitive architecture combining:
- **Tree-of-Thoughts (ToT)**: Decomposes reasoning into explicit intermediate semantic units (`ThoughtStep` / `ThoughtNode`).
- **Predictor Upper Confidence bounds for Trees (PUCT)**: Guided tree search balancing exploitation of high-value thoughts with exploration of alternative trajectories.
- **Closed-Loop Self-Verification (`SelfVerifier`)**: Detects logical contradictions, penalizes cyclic repetitions, rewards valid deductions, and prunes invalid search branches in real time.

---

## 2. Mathematical Foundation

### A. Predictor Upper Confidence Bound for Trees (PUCT)
When traversing the thought tree from parent node $s$ to child action $a$:
$$\text{PUCT}(s, a) = Q(s, a) + c_{\text{puct}} \cdot P(s, a) \cdot \frac{\sqrt{N(s)}}{1 + N(s, a)}$$
Where:
- $Q(s, a) = \frac{W(s, a)}{N(s, a)}$: Mean accumulated action value (heuristic confidence score).
- $P(s, a)$: Prior probability assigned to the thought candidate by the `SelfVerifier`.
- $N(s)$: Total visit count of parent state $s$.
- $N(s, a)$: Visit count of edge $(s, a)$.
- $c_{\text{puct}} = 1.414$: Exploration constant.

### B. Self-Verification & Invariant Checking
Each proposed thought candidate step $S = (t_1, \dots, t_L)$ is verified across four dimensions:
1. **Log Probability & Perplexity Confidence**:
   $$\overline{\log P} = \frac{1}{|S|} \sum_{i=1}^{|S|} \log P(t_i \mid \text{context})$$
   Mapped to $[0, 1]$ via calibrated sigmoid: $\sigma(2.0 \cdot (\overline{\log P} + 1.5))$.
2. **Repetition & Loop Penalty**:
   Detects degenerate $N$-gram loops ($k \ge 3$ consecutive repeated tokens) and applies penalty $\Delta_{\text{rep}} = 0.3$.
3. **Deductive Transition Bonus**:
   Detects logical transition markers (`"therefore"`, `"because"`, `"since"`, `"implies"`, `"step"`) and awards bonus $\Delta_{\text{deduct}} \le 0.15$.
4. **Contradiction Invariant Pruning**:
   Detects formal contradiction phrases (e.g. `"1 = 0"`, `"false is true"`). When triggered:
   - Sets $\text{score} = 0.05$.
   - Sets `is_valid = false`.
   - The engine prunes the branch immediately, preventing search budget from being wasted on invalid trajectories.

---

## 3. MCTS Search Workflow
```
[Root: Prompt Tokens]
        |
    [Selection via PUCT]
        v
  [Expansion: K Candidate Thoughts]
        |
        +---> Candidate 1 (Contradiction -> PRUNED)
        |
        +---> Candidate 2 (Valid Deduction: Score 0.85)
        |           |
        |       [Rollout / Verification]
        |           v
        |       [Backpropagation: Q, N updated up to Root]
        |
        +---> Candidate 3 (Score 0.60)
```

---

## 4. CLI Usage
Execute multi-step Tree-of-Thoughts reasoning directly:
```cmd
.\build_ninja\bin\aic.exe reason "Suppose x = 10, y = 20. Find x + y step by step:" example_ckpts\stable_step_21.aic
```
Outputs:
- Full search statistics (Nodes Explored, Pruned Branches, Search Latency).
- Step-by-step reasoning trajectory with confidence scores.
- Final extracted conclusion.
