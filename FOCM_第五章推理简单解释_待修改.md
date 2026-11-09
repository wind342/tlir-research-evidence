# FoCM 第五章：推理的简单计算解释（待修改草稿）

**状态：** 待讨论／待统一修改；**未写入 FoCM 论文正文，也不修改 ICML 摘要。**

**目的：** 为第五章 *A Simple Theory* 的推理与反馈小节保存一个具体计算示例。由基本神经网络运算出发，而不是把 Attention 当成推理方程的前提。它是示例性的数学解释，跨架构证据仍以 INF-E01、INF-G01、INF-G02 和独立压力测试为准。

## 建议的英文段落

### Inference recruits and combines learned computation

Consider a single nonlinear neuron with two learned input weights:

$$
f_\theta(x)=\operatorname{ReLU}(ax_1+bx_2).
$$

At the fixed learned state $\theta^*=(2,1)$, the two component contributions are $s_1(x)=2x_1$ and $s_2(x)=x_2$, and inference produces

$$
Y_x=\operatorname{ReLU}(s_1(x)+s_2(x)),
\qquad \Delta_{\mathrm{inf}}L=0.
$$

The same learned components contribute differently under different inputs. Their combination can also be non-additive. To measure this, introduce reversible component gates $g_1,g_2\in\{0,1\}$ and evaluate

$$
f_{g_1,g_2}(x)=
\operatorname{ReLU}(2g_1x_1+g_2x_2).
$$

For $x=(-1,1)$, the four gated outputs $(f_{11},f_{10},f_{01},f_{00})$ are $(0,0,1,0)$, giving the exact interaction

$$
I_{12}=f_{11}-f_{10}-f_{01}+f_{00}=-1.
$$

For $x=(1,1)$, the same interaction is zero. Thus query-dependent functional contributions and non-additive component effects arise from elementary nonlinear computation with unchanged learned parameters.

Inference also depends on the versions formed through learning. Replacing the trained first-component weight $a^*=2$ with a pre-learning version $a_{\mathrm{pre}}=1$ changes the output at $x=(1,1)$ from $3$ to $2$; exact restoration recovers the original result. The gates and version replacements are diagnostic interventions, distinct from ordinary inference.

These calculations illustrate the TLIR inference relation: a fixed learned state supplies functional components, the current input determines their contributions, and nonlinear computation combines them into an output. No persistent learning update is required.

### Feedback closes the loop

When the consequences of inference enter subsequent training, they can change the next training action and thereby the learned computation available to later inference:

$$
Y_t\longrightarrow B_{t+1}
\longrightarrow A_{t+1}\longrightarrow Z_{t+2}.
$$

Inference preserves the current learned state; feedback-driven training can change the learned state used by future inference.

## 留待一起修改时处理

- 与 FoCM 第五章前四节统一变量、行文密度与公式编号。
- 检查是否将单神经元的两项输入贡献称为“components”最合适，以及它与实验的组件门控定义是否足够清楚。
- 保留“诊断门控／旧版本替换”与“原生冻结推理”的区别。
- 如需进一步增强最后一节，可以考虑添加一个同样简单的反馈影响后续参数更新的计算实例，而不仅列出闭环箭头。
- **不要把此示例写成 Attention 推导，也不要宣称单神经元例子独立证明了跨系统普遍性。**
