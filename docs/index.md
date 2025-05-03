<style>
header {
  display: none;
}
</style>

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
Online Exploratory World Model

## Contents
- [Introduction](#1-introduction)
- [Proposed Methodology](#2-proposed-methodology)
  - [Method 1 : Online Exploratory World Model](#21-method-1--online-exploratory-world-model)
    - [Actions Scorer](#211-actions-scorer)
  - [Method 2 : Ideal WM](#22-method-2--ideal-wm)
  - [Inducing Exploration in Dreamer-V3](#23-inducing-exploration-in-dreamer-v3)
- [Results](#3-results)
- [Conclusion](#4-conclusion)
- [Future Directions](#5-future-directions)

## 1. Introduction

Reinforcement learning (RL) has demonstrated significant advancements across a diverse range of domains, from strategic board games to sophisticated robotic control tasks. Nevertheless, purely model-free RL approaches typically demand extensive interaction data, face considerable challenges in dealing with environments with sparse or long-horizon rewards, and necessitate substantial hyperparameter tuning and retraining for each new task even within the same domain.

To mitigate these limitations, world models have emerged as an effective paradigm. A world model learns an internal representation of environmental dynamics, enabling agents to anticipate the future states resulting from a sequence of actions. Such models provide the capability for the agent to internally simulate or "imagine" future scenarios, significantly reducing reliance on inefficient trial-and-error interactions with the real environment.

Historically, many world models have operated directly within pixel-space, reconstructing raw images to predict future observations. However, this approach incurs substantial computational costs due to intensive image reconstruction requirements and frequently relies on complex diffusion-based models. Consequently, recent research has increasingly favored latent-space prediction, wherein models operate on compressed, low-dimensional representations of environmental states.

 Recent innovations like DINO-WM utilize pretrained visual embeddings such as those derived from DINOv2 to construct task-agnostic latent dynamics models. These pretrained embeddings eliminate the computational overhead associated with pixel reconstruction entirely, enabling more efficient prediction and planning. Moreover, models such as DINO-WM exhibit significant generalization capabilities across varying task configurations and environments, even in the absence of explicit reward supervision. This advancement underscores the potential of latent-space world models to address fundamental RL challenges, thereby advancing efficiency, generalization, and adaptability in reinforcement learning.



## 2. Proposed Methodology

### 2.1 Method 1 : Online Exploratory World Model

We try to work on these two recent methods and try to solve the issues in these methods. DINO-WM assumes having access to offline datasets with sufficient state-action coverage, which can be challenging to obtain for highly complex environments, and it also not the approach that humans would generally take while performing a task if you are play a game you would just know the basic rules or maybe not even that and start playing by taking random actions and making your understanding of the games dynamics better overtime. So we try to collect data using various exploration strategies. This data is mixed with optimal paths (reward maximizing actions) and suboptimal paths (for exploration) and trained on WM. These explorations strategies will involve maximising reward and reward would be of different types and would vary across environments. 

Our data collection would look like following:
- Start from state S_0
- Say n possible actions, use action_scorer to get optimal action, take the best exploratory action based  on the exploration reward aside from the optimal action
- Build a short suboptimal exploratory path
- Train WM on both optimal path and suboptimal paths

#### 2.1.1 Actions Scorer

The reward strategies can be broadly categorized into extrinsic, intrinsic, hybrid, and hierarchical rewards. In our case the intrinsic reward strategies seem to be relevant so we try to work on them. We implement exploration/curiosity based reward strategies. Examples of these strategies in the pushT environment can be increasing the number of collisions between the pusher robotic arm and the T block, increasing pixel to pixel change in the environment per step.


<p style="text-align: center;">
  <img src="demo_page/images/arrow1.png" width="900"/>
</p>
<p align="center">
  <strong><span style="font-size: 18px;">Figure 1: Blue: optimal path (according to some action_scorer); red: suboptimal paths.</span></strong>
</p>

<p align="center">
  <img src="images/as.png" width="500"/>
</p>

<p align="center">
  <strong><span style="font-size: 18px;">Figure 2: Action scorer. </span></strong>
</p>
In addition to global exploration strategies, we adopt a local exploration approach to effectively train and refine our world model. Specifically, starting from an identified optimal trajectory (represented by the dark blue line), we systematically explore additional nearby states within a defined local window. This local exploration involves investigating multiple alternative paths branching off from the current optimal trajectory.
Within this local exploration window, we calculate and evaluate rewards for all potential state-action pairs explored. Importantly, if any of these alternative paths within the local window yield a higher reward compared to the previously identified optimal path, the superior alternative path (represented by the light blue line) replaces the current optimal trajectory, becoming the new focus for exploration. This dynamic updating ensures continuous refinement and adaptation of the optimal path based on the most rewarding outcomes discovered through local exploration.
These locally explored suboptimal paths also provide diverse and valuable training data, enriching the world model's understanding by covering a broader range of environmental dynamics. This comprehensive exploration methodology enhances the predictive capability of our world model, significantly increasing policy robustness and adaptability to diverse and unforeseen environmental conditions.


<p align="center">
  <img src="images/arrow2.png" width="500" />
</p>

<p align="center">
  <strong><span style="font-size: 18px;">Figure 3: Tree based local search. </span></strong>
</p>

### 2.2 Method 2 : Ideal WM
The current approach in DINO-WM involves training the world model followed by planning, our proposed methodology integrates these stages into an iterative cycle. Initially, we perform an initial phase of world model training using exploration-derived data. Once the world model has acquired foundational dynamics knowledge, we proceed to a planning stage where optimized actions are computed. These optimized actions, derived from planning, are then incorporated back into further training of the world model, enriching its predictive capabilities and aligning its understanding closely with optimal decision-making patterns.

This iterative cycle consisting of alternating training and planning phases is repeated multiple times. Each iteration progressively refines the world model by continually incorporating the latest optimal actions identified during planning. This continuous feedback loop between planning and training ensures that the world model dynamically improves, effectively integrating strategic insights from planning into its predictive structure.


<p align="center">
  <img src="images/EA.png" width="500" />
</p>

<p align="center">
  <strong><span style="font-size: 18px;">Figure 4: Existing Architecture</span></strong>
</p>

<p align="center">
  <img src="images/PA.png" width="500" />
</p>
<p align="center">
  <strong><span style="font-size: 18px;">Figure 5: Proposed Architecture</span></strong>
</p>


### 2.3 Inducing Exploration in Dreamer-V3

To test our hypothesis on Dreamer-V3, we modify the agent's policy method such that during environment interaction, the batch of actions is composed of both policy-driven and randomly sampled actions. The implementation details are as follows:

- **Batch Size**: We configure the system to use a batch size of 32 parallel environment instances.
- **Policy Sampling**: For the first 16 environments (instances 0 to 15), actions are sampled from the learned policy distribution, preserving the original behavior of DreamerV3.
- **Random Sampling**: For 4 environments (instances 16 to 19), actions are uniformly sampled from the full discrete action space (`0` to `17`, inclusive, for Atari).
- **Criteria Based on Middle Portion**: 4 instances: Exploration reward based on pixel to pixel change of middle portion (breakout tile)
- **Criteria Based on Lower Portion**: 4 instances: Exploration reward based on pixel to pixel change of lower part (disk movement change)
- **Criteria Based on Upper Portion**: 4 instances: Exploration reward based on pixel to pixel change of uppermost part (score change)

This ensures that in each training step, 50% of actions reflect learned behavior, while 50% inject purely exploratory behavior.

## 3. Results

<h3>Results PushT (Method 1 DINO WM)</h3>
The Dino-WM results on the PushT environment highlight several limitations. While exploratory actions sampled from the distribution introduced some variability, the optimal action selection—being greedily biased toward the nearest path to the T—caused the pusher to remain near the object without meaningful interaction. Although training loss decreased quickly, it plateaued early, suggesting insufficient convergence. The model struggled to learn effective dynamics due to limited training epochs, simplistic planning, and the absence of expert data, which made capturing realistic physics particularly challenging.
<p align="center">
  <img src="gifs/output_final_0_failure-ezgif.com-video-to-gif-converter.gif" width="300" style="margin-right: 20px;">
  <img src = "images/M1Dino.png">
</p>

<h3>Results Atari (Method 1 DINO WM)</h3>
While random exploration paths sometimes resulted in accidental paddle alignment, the success rate was extremely low due to undirected sampling. The optimal path strategy, using tree-based greedy reward selection, showed consistent short-term success by immediately targeting reachable bricks but failed to maintain the necessary paddle alignment for sustained 
<p align="center">
  <img src="gifs/episode1-ezgif.com-video-to-gif-converter.gif" width="300" style="margin-right: 20px;">
  <img src = "images/M22.png">
</p>


<h3>Results PushT (Method 2 DINO WM)</h3>
PushT environment demonstrate incremental improvement over Method 1, with slightly more effective action behaviors emerging during planning. As in Method 1, the greedy criteria-based planning fails to produce goal-directed behavior consistently, leading the pusher to interact ineffectively with the T-shaped object. Additionally, the absence of expert demonstrations continues to hinder the model’s ability to learn accurate physical interactions, emphasizing the need for better-informed action selection strategies and more diverse training data to improve model performance in complex physical environments.

<p style="text-align: center;">
  <img src="gifs/output_final_2_failure-ezgif.com-video-to-gif-converter.gif" width="400" style="margin-right: 20px;" />
  <img src="images/M2.png" width="400" />
</p>


<h3>Results (Inducing exploration in Dreamer-V3)</h3>
The primary reason our modified DreamerV3 model did not achieve the desired performance is the significantly reduced training duration. While the original DreamerV3 model was trained for 10^10 steps, our model was trained for only 10^5 steps, limiting its opportunity to thoroughly learn optimal policies. Additionally, introducing random trajectories as seed states for the imagination process inadvertently slowed policy convergence, as the model frequently imagined suboptimal or irrelevant scenarios. To address this, we propose masking these random-action instances during the imagination phase, ensuring the policy training focuses exclusively on trajectories derived from its learned distribution, potentially accelerating convergence and improving performance.
<p align="center">
  <img src="gifs/episode1_gray-ezgif.com-video-to-gif-converter.gif" width="300"><br>
</p>


## 4. Conclusion

We introduce a lightweight yet impactful modification to the DreamerV3 framework to promote better exploration and robustness. By injecting randomly sampled actions into a portion of the interaction batch, we encourage the agent to visit diverse states, making both the world model and policy more capable of handling suboptimal and unexpected situations. This strategy preserves the core structure of DreamerV3 while addressing one of its key limitations in exploration and generalization.

This mixed-policy sampling technique offers a promising direction for enhancing MBRL agents, particularly in environments where optimal trajectories are hard to discover without structured exploration.

## 5. Future Directions

- Try a neural network–based action scorer.
- Integrate planning inside world model (WM) training and jointly optimize planning and world model
- Evaluate across diverse environments to assess robustness and failure modes of the method.
