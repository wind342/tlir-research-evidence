# FoCM 结论修订稿（待统一修改）

**状态：** 讨论确认的候选结论；暂存于 ICML 摘要所在仓库，供以后集中修改。**未修改 ICML 摘要或 FoCM 论文正文。**

## Conclusion — English

Through systematic causal experiments across seven neural-network systems, we revealed the underlying dynamics of training, learning, inference, and reinforcement. Training produces state-dependent functional responses through realized updates; learning persistently reorganizes distributed functional support; capability is a target-specific decision on a continuous functional readout, with its acquisition, loss, and recovery determined by responses crossing decision boundaries; inference recruits and combines learned support while preserving the learned state; and reinforcement connects the consequences of inference to subsequent training.

We formulated these experimentally established relationships as a unified system of coupled dynamical equations. Starting from these equations and using the original mathematical models and conditions of the respective studies, we derived mathematical relationships across fourteen established theories. The core relationships can also be recovered from elementary neural-network computations. Together, these findings open the neural-network black box to mechanistic investigation and connect previously separate research fields through a unified mathematical framework.

## 中文翻译

通过对七种神经网络系统开展系统性因果实验，我们揭示了训练、学习、推理与强化的内在动力学机制。训练通过实际更新产生状态依赖的功能响应；学习使分布式功能支撑发生持久重组；**能力是对连续功能读出进行目标特定判定的结果，其获得、丧失与恢复取决于功能响应是否跨越判定边界**；推理在保持已学习状态不变的情况下调用并组合已学习的功能支撑；强化则将推理产生的后果与后续训练联系起来。

我们将这些通过实验建立的关系表述为一套统一的耦合动力学方程组。从这些方程出发，结合各项原始研究所采用的数学模型与条件，我们推导出了横跨十四种既有理论的数学关系。这些核心关系还可以从神经网络最基本的计算中推导出来。这些发现共同推动了对神经网络黑箱内部机制的研究，并通过统一的数学框架，将此前相互独立的研究领域联系起来。

## 待统一修改时的说明

- 保留研究的实际顺序：实验发现 → 统一动力学方程 → 14项数学推导。
- 结论明确包含训练、学习、**能力的定义及获得/丧失/恢复的判定机制**、推理、强化。
- 第五章「基本计算推导核心关系」的发现已经写入第二段。
- 这是讨论稿；未经后续明确指示，不直接替换 FoCM 正文。
