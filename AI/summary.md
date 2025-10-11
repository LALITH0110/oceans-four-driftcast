# Using Reinforcement Learning to Simulate and Control Plastic Drift in Oceans

## Overview

This document summarizes our conversation on how Reinforcement Learning (RL) can be applied to both **simulate** and **control** plastic drift in the oceans. The goal is twofold:

1. **Improve ocean drift simulations (A)** — by using RL to learn corrections to imperfect physics-based models.
2. **Optimize cleanup strategies (B)** — by using RL to plan where and when to deploy cleanup actions (like boats or booms).

---

## 1. The Problem: Plastic Drift Dynamics

Plastic at the ocean surface is moved by:
- Ocean currents
- Winds
- Surface waves (Stokes drift)
- Random turbulence (diffusion)

This can be described by an advection–diffusion equation or, equivalently, a stochastic particle model:

$$dX_t = u(X_t, t)\,dt + \alpha\,w(X_t, t)\,dt + s(X_t, t)\,dt + \sqrt{2\kappa(X_t)}\,dW_t$$

where $X_t$ is particle position and $u, w, s, \kappa$ are the environmental drivers.

In practice, real plastic motion deviates from this due to unresolved small-scale physics (eddies, beaching, vertical mixing, etc.).

---

## 2. RL for Improving the Drift Model (Task A)

### Idea
The physical model gives a “first guess.” RL learns a **correction** term that captures missing physics.  
This yields a hybrid model: physics + learned residual.

### MDP Formulation

| Element | Meaning |
|----------|----------|
| State ($s_t$) | Local environment: particle position, velocity field, vorticity, distance to coast, etc. |
| Action ($a_t$) | Small correction to the drift velocity (2D vector) |
| Transition | Physics + correction: $x_{t+1} = f_{physics}(x_t) + a_t$ |
| Reward | Negative distance between simulated and observed trajectories or fields |
| Policy | Neural net $\pi_\theta(a \vert s)$ mapping local features to corrections |

### Goal
Find $\pi_\theta$ that maximizes the similarity between simulated and observed drift data.

### Algorithmic View
- The simulator is **differentiable** (e.g., in JAX).
- Use **gradient-based RL** or direct policy optimization (PPO, DDPG, or gradient descent on rollout loss).

This is a form of **Physics-Informed RL (PIRL)**.

---

## 3. RL for Optimizing Cleanup or Intervention (Task B)

### Idea
Use RL to find where and when to deploy cleanup assets (boats, barriers, drones) for maximal plastic removal at minimal cost.

### MDP Formulation

| Element | Meaning |
|----------|----------|
| State ($s_t$) | Plastic concentration map, boat/boom locations, environmental conditions |
| Action ($a_t$) | Move boats, deploy/retract booms, or choose next waypoint |
| Transition | Determined by drift simulator (updated plastic distribution after advection and cleanup) |
| Reward | Plastic removed − operational cost − constraint penalties |
| Policy | Strategy $\pi_\theta(a \vert s)$ deciding cleanup actions |

### Algorithms
- **Policy Gradient / Actor–Critic** methods (PPO, A2C) for continuous actions.
- **Multi-Agent RL** if multiple boats or drones coordinate.
- Equivalent to solving an **optimal control** problem over an advection–diffusion field.

---

## 4. How Tasks A and B Interact

1. **Train Task A** to correct and calibrate the physical simulator.
2. **Use that improved simulator** in Task B to optimize real-world cleanup policies.

This forms a **two-layer RL system**:
- Inner layer: model correction.
- Outer layer: control and decision-making.

---

## 5. Conceptual Summary

| Aspect | Task A (Improve Drift) | Task B (Optimize Cleanup) |
|--------|------------------------|----------------------------|
| Purpose | Learn missing physics | Optimize cleanup control |
| Type | Model-based RL | Policy-based control RL |
| Action | Velocity correction | Cleanup movement/deployment |
| Reward | Fit to observations | Plastic removed − cost |
| Environment | Differentiable physics simulator | Stochastic simulator |
| Algorithms | Gradient-based (PIRL, DDPG) | PPO / Actor–Critic / Multi-Agent RL |

---

## 6. Key Insights

- RL doesn’t replace physics — it **sits on top of it**, learning corrections and control strategies.
- Combining RL with a differentiable ocean simulator bridges **data-driven learning** and **physical modeling**.
- The framework can be extended to study **source attribution**, **fragmentation dynamics**, or **autonomous cleanup planning**.

---

## 7. Next Steps (Practical Plan)

1. **Build a toy simulator** (2D grid, synthetic currents).
2. **Train drift-correction RL** to fix known biases.
3. **Train cleanup RL** using that improved simulator.
4. Evaluate on observed drifter data or real current fields.

---

**Author:** ChatGPT (GPT-5) and User  
**Date:** October 2025
