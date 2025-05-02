# Deep Decision Making and Reinforcement Learning: Final Project Submission

## Team Members  
<p align="center">
  <strong>Jayesh Chaudhari</strong> &nbsp;&nbsp;&nbsp;&nbsp;
  <strong>Satyam Kumar</strong> &nbsp;&nbsp;&nbsp;&nbsp;
  <strong>Varad Vijay Suryavanshi</strong> &nbsp;&nbsp;&nbsp;&nbsp;
  <strong>Rivujit Das</strong>
</p>

<p align="center">
  jsc9903@nyu.edu &nbsp;&nbsp;&nbsp;&nbsp;
  sk12075@nyu.edu &nbsp;&nbsp;&nbsp;&nbsp;
  vs3273@nyu.edu &nbsp;&nbsp;&nbsp;&nbsp;
  rd3681@nyu.edu
</p>



## Title  
**Improving World Model Robustness in DreamerV3 Using Mixed Policy Sampling**

## Contents
- [1. Introduction](#1-introduction)
- [2. Motivation for the Change](#2-motivation-for-the-change)
- [3. Proposed Modification: Mixed Policy Sampling](#3-proposed-modification-mixed-policy-sampling)
- [4. Technical Implementation](#4-technical-implementation)
- [5. Effects on Training Dynamics](#5-effects-on-training-dynamics)
- [6. Assumptions and Constraints](#6-assumptions-and-constraints)
- [7. Conclusion](#7-conclusion)


## 1. Introduction

DreamerV3 is a state-of-the-art model-based reinforcement learning (MBRL) algorithm that enables agents to plan and learn from imagined experiences. It leverages a world model to simulate future trajectories, thus significantly improving sample efficiency. At its core, DreamerV3 uses a policy network (an `MLPHead`) that outputs a probability distribution over actions, and typically selects actions with higher probabilities during training and interaction.

However, this behavior leads the agent to primarily visit trajectories deemed optimal or high-reward according to the current state of the policy. As a result, suboptimal, rare, or "bad" states might never be explored. This lack of diversity in the training data can cause the world model to generalize poorly, especially in complex environments with deceptive rewards or partial observability.

<p align="center">
  <img src="images/arrow1.png" width="500" style="margin-right: 20px;" />
  <img src="images/arrow2.png" width="500" />
</p>


## 2. Motivation for the Change

The world model in DreamerV3 is trained using observations collected during environment interaction. If the policy continually samples only high-probability actions, the resulting state transitions tend to be narrow and repetitive, focused around a narrow band of optimal trajectories. Consequently:

- The world model fails to accurately capture the dynamics of rare or suboptimal regions of the state space.
- The policy becomes brittle, unable to recover when it encounters unforeseen states during inference.
- The imagined rollouts used for policy improvement remain anchored to overly idealized conditions.

To address these limitations, we introduce a controlled exploration strategy by augmenting the action sampling process during real-world interaction.


## 3. Proposed Modification: Mixed Policy Sampling

We modify the agent's policy method such that during environment interaction, the batch of actions is composed of both policy-driven and randomly sampled actions. The implementation details are as follows:

- **Batch Size**: We configure the system to use a batch size of 20 parallel environment instances.
- **Policy Sampling**: For the first 16 environments (instances 0 to 15), actions are sampled from the learned policy distribution, preserving the original behavior of DreamerV3.
- **Random Sampling**: For the remaining 4 environments (instances 16 to 19), actions are uniformly sampled from the full discrete action space (`0` to `17`, inclusive, for Atari).

This ensures that in each training step, 80% of actions reflect learned behavior, while 20% inject purely exploratory behavior.

<p align="center">
  <img src="images/EA.png" width="500" style="margin-right: 20px;" />
  <img src="images/PA.png" width="500" />
</p>



## 4. Technical Implementation

The modification is implemented inside the `policy` method of the custom `Agent` class:

- After computing the `act` dictionary from the policy network, we generate a host-side PRNG key to sample 4 random actions.
- These random actions are inserted into the second half of the `act['action']` tensor using JAX's `.at[].set()` API.
- This operation is performed outside of the JAX tracing context to avoid disallowed host-to-device data transfers.

Furthermore, to support this change:

- The `config.yaml` file is updated to set `batch_size: 20`.
- The number of parallel environments in the runner is configured to 20.

## 5. Effects on Training Dynamics

This mixed sampling strategy influences the DreamerV3 learning process in two key ways:

1. **Improved World Model Coverage**: Suboptimal and "bad" states, introduced via random actions, enhance the diversity of the replay buffer. The world model learns to simulate a broader and more accurate range of environment dynamics.

2. **Robust Policy Learning**: Since DreamerV3 uses the final `k` steps from real trajectories to seed imagination rollouts, the inclusion of difficult or novel states forces the policy to learn strategies for recovery and generalization.

## 6. Assumptions and Constraints

- This change applies only during real-world environment interaction. The policy used in imagination remains unchanged to preserve the benefits of gradient-based policy improvement.
- Batch size and environment count must be consistent (20 in this case) to maintain alignment between real trajectories and imagination seeds.
- The fraction of random actions (4/20 = 20%) is tunable. Larger ratios may induce more exploration but may destabilize training.


## 7. Conclusion

We introduce a lightweight yet impactful modification to the DreamerV3 framework to promote better exploration and robustness. By injecting randomly sampled actions into a portion of the interaction batch, we encourage the agent to visit diverse states, making both the world model and policy more capable of handling suboptimal and unexpected situations. This strategy preserves the core structure of DreamerV3 while addressing one of its key limitations in exploration and generalization.

This mixed-policy sampling technique offers a promising direction for enhancing MBRL agents, particularly in environments where optimal trajectories are hard to discover without structured exploration.


