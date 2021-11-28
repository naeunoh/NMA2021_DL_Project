# NMA2021 Deep Learning Project

### Title: Building a controller in model-free and model-based RL in two-step task

### Team Falling Humans : Jaeyoung Jeon, NaEun Oh, Hanbo Xie, Jindu Wang 


Unveiling the fundamental cognitive process of maximizing reward in a naturalistic environment has been a long-going challenge in both cognitive neuroscience and computer science. Recently, there are many attempts to use reinforcement learning to better understand the relevant mechanisms of the human brain.The human brain employs both exploration and exploitation strategies in learning about a new environment, where they generally apply model-free and model-based RL algorithms. The standard view is that both algorithms run in parallel either through integration or competition. However, previous work assumes unidirectional transition from model-free to model-based algorithms but not vice versa, which is more likely to occur in a changing environment. Moreover, the fundamental structure that directly employs algorithms in parallel before sufficient evidence accumulation may not adequately explain human behavior of exploration under uncertain environments.

This research intends to investigate how the transition between model-free and model-based learning contributes to adaptive learning under uncertainty, especially how different models represent exploitation and exploration. We simulate various agents of RL playing a two-step bandit task with shifting probabilities of rewards and state transitions. We define a model that shifts between model-free and model-based RL with an independent controller module based on uncertainty. We expect the shifting model to outperform the independent RLs on the fraction of rewarded trials and total reward. In addition, the shifting model will exhibit patterns of stay probabilities that differ from independent RLs. This research suggests the adaptive transition between learning models provides a better understanding of the decision making process and its underlying architecture.
