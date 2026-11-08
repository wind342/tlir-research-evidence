# TLIR 推理方程的四项探索性数学外推及外部研究证据

**Status: EXPLORATORY / 探索性；非冻结实验、非正式第15—18项外推**  
**记录日期：** 2026-11-08  
**推导起点：** TLIR 统一方程中已经通过实验建立的训练、学习、能力读出、冻结推理与反馈关系  
**证据类型：** 数学推论／明确条件下的示例／2026-11-08 外部公开文献核对  
**版本边界：** 独立研究笔记；不修改 TLIR 原方程、14 项正式 TAR 外推、冻结协议、结果、独立检查器或已有论文主张

## 0. 摘要与来源

本笔记记录一次以 **TLIR 推理方程 \(\mathcal I\)** 为切入点、联立前序状态转移与能力读出方程所作的四个数学探索：

1. **完整当前行为并不必然决定同一更新后的行为**：行为等价类一般不一定在训练状态转移下封闭。
2. **固定持久学习状态仍可实现贝叶斯后验更新**：上下文驱动的瞬时支撑调用可以精确表示后验预测。
3. **固定学习状态下的推理计算量可以决定是否越过某一输出判定边界**：对满足压缩映射条件的推理过程，可推导充分的步数界。
4. **推理—反馈—训练闭环能够具有可计算的稳定性阈值**：特定二维线性特例中得到反馈增益与衰减之间的临界等式。

这里的 **TLIR 基础方程与先前因果实验** 属于已建立研究；四个带附加数学条件的推论属于 **本次探索**。外部文献属于 **相关理论或实验先例**，并不表示其作者测试或证明了此处的完整 TLIR 推论。探索只整理数学及公开证据，**未新增原生实验、独立复现实验、冻结预测或新的机器学习 benchmark**。

TLIR 原始基础文档：[统一动力学 V5](TLIR_UNIFIED_DYNAMICS_V5.md)；相关冻结证据定位：[EXPERIMENT_INDEX.md](EXPERIMENT_INDEX.md)、[CROSS_SYSTEM_EVIDENCE_CHAIN.md](CROSS_SYSTEM_EVIDENCE_CHAIN.md)、[RL_EVIDENCE_CHAIN.md](RL_EVIDENCE_CHAIN.md)。

## 1. [T] 共同起点：TLIR 原方程

以下保留 TLIR 定义与数学对象的区别。

**训练的实际动作与状态转移：**

$$
Z_t=(L_t,\omega_t),\qquad
A_t=\mathcal U(Z_t,D_t,B_t),\qquad
Z_{t+1}=\mathcal T(Z_t,A_t).
$$

参数化特例中，实际更新为

$$
\delta_t=\Psi(\theta_t,\omega_t,D_t,B_t),\qquad
\theta_{t+1}=\theta_t+\delta_t.
$$

**学习的功能支撑、实际响应、能力读出：**

$$
P_{t,q}=\mathcal P_q(L_t;\mathcal J_q),\quad
\Delta P_{t,q}=P_{t+1,q}-P_{t,q},
$$

$$
R_{t,q}=m_q(L_{t+1})-m_q(L_t),\qquad
C_{t,q}=\chi_q\bigl(m_q(L_t)\bigr).
$$

**冻结推理：**

$$
S_{x,c}^{\mathrm{active}}=\Pi_{x,c}(L^*),\qquad
Y_{x,c}=H_{x,c}\bigl(S_{x,c}^{\mathrm{active}}\bigr),\qquad
\Delta_{\mathrm{inf}}L=0.
$$

用组合符号可写

$$
\mathcal I(L^*,x,c)=H_{x,c}\bigl(\Pi_{x,c}(L^*)\bigr).
$$

这里 \(\Pi\) 是支撑调用／选择／组合意义上的算子，**并不预设线性投影，不依赖 Attention 架构**。

**反馈闭环：**

$$
Y_t=\mathcal I(L_t,x_t,c_t),\quad
B_{t+1}=\mathcal B(B_t,Y_t,E_t),
$$

$$
A_{t+1}=\mathcal U(Z_{t+1},D_{t+1},B_{t+1}),\quad
Z_{t+2}=\mathcal T(Z_{t+1},A_{t+1}).
$$

这组 [T] 是下面四个探索的共同数学起点。下面的假定和专用符号属于各自特例 [A]，不是重新定义 TLIR，也不是从先验概念替代其实验发现的定义。

---

## 2. 探索 E-I01：当前行为等价不保证训练动力学等价

**问题。** 如果两个持久状态在全部输入与上下文上产生同一推理函数，是否能仅靠这个当前函数决定同一训练动作作用后的新函数？

### [T]

定义固定状态的完整行为映射：

$$
F_L:(x,c)\mapsto \mathcal I(L,x,c).
$$

训练状态发生 \(\mathcal T(L,A)\) 的变化（下式为省略优化器记忆的参数状态特例）。

### [A] 数学条件

设两个不同状态 \(L_1,L_2\) 满足 **全输入行为等价** \(F_{L_1}=F_{L_2}\)，且允许对两个参数状态施加同一个预先给定的增量 \(A\)。

具体取一维模型：

$$
\mathcal I(L,x)=L^2x,\qquad
L_1=1,\qquad L_2=-1,\qquad
\mathcal T(L,A)=L+A,\qquad A=0.1.
$$

这是一个**受控同增量干预的数学反例**，不是关于 Adam 或梯度下降在两个状态下会自行形成相同增量的断言。

### [R] 推导

更新前，对任何 \(x\)：

$$
F_{L_1}(x)=F_{L_2}(x)=x.
$$

更新后：

$$
F_{\mathcal T(L_1,A)}(x)=1.21x,\qquad
F_{\mathcal T(L_2,A)}(x)=0.81x.
$$

因此：

$$
\boxed{
F_{L_1}=F_{L_2}
\;\not\Longrightarrow\;
F_{\mathcal T(L_1,A)}=F_{\mathcal T(L_2,A)}.
}
$$

若使用输入 \(x=1\) 上的目标边界 \(m(L)=\mathcal I(L,1)-1.1\)，并令 \(C(L)=\mathbf1[m(L)>0]\)，两者更新前均为错误，而更新后仅 \(L_1\) 的后继状态为正确。

更一般地，在状态等价关系 \(L\sim L'\iff F_L=F_{L'}\) 下，只有当

$$
L\sim L'\Longrightarrow\mathcal T(L,A)\sim\mathcal T(L',A)
\quad\text{对允许的全部动作 }A\text{ 成立}
$$

时，才能在该等价商空间上定义由当前行为与动作唯一决定的封闭转移 \(\overline{\mathcal T}([L],A)\)。

**探索结论：** 当前完整输入输出函数可能不足以作为未来训练响应的封闭动力学状态；问题比“仅凭准确率不足以测量学习”更强。

### 外部证据

- Dinh 等人，*Sharp Minima Can Generalize For Deep Nets*，ICML 2017，证明函数等价的参数表示可以具有不同参数空间几何。这是 **行为等价与内部状态差异** 的数学先例；并未测试上述同增量反例：[PMLR](https://proceedings.mlr.press/v70/dinh17b.html)。
- *Exact learning dynamics of deep linear networks with prior knowledge*，研究不同初始化对学习轨迹的影响，并报告最终函数／损失相同时内部表示和学习轨迹可不同。这是 **初始化与训练历史依赖** 的相关理论与实验，不是同增量反例的独立验证：[论文全文](https://pmc.ncbi.nlm.nih.gov/articles/PMC10955603/)。

**证据结论：** 这里的特定反例由直接代数严格成立；外部文献支持问题的重要性与相邻现象，不是新增 TLIR 实验验证。

---

## 3. 探索 E-I02：固定学习状态下的贝叶斯推断

### [T]

$$
Y_{x,c}=H_{x,c}(\Pi_{x,c}(L^*)),\qquad \Delta_{\mathrm{inf}}L=0.
$$

### [A] 数学条件

设 \(L^*\) 已包含一个有限假说族 \(h=1,\dots,K\)、正先验 \(\pi_h\)、条件似然 \(p_h(z)\) 和预测分布 \(p_h(y\mid x)\)。上下文 \(c_n=(z_1,\ldots,z_n)\) 在真实假说 \(h^*\) 下独立同分布，相关似然为正，KL 散度存在。

取当前推理支撑调用为贝叶斯权重：

$$
\Pi_{c_n}(L^*)_h
=
\frac{\pi_h\prod_{i=1}^n p_h(z_i)}
{\sum_{j=1}^{K}\pi_j\prod_{i=1}^n p_j(z_i)}.
$$

并用 \(H\) 混合假说预测。

### [R] 推导

$$
\boxed{
\mathcal I(L^*,x,c_n)
=
\sum_{h=1}^{K}
\Pi_{c_n}(L^*)_h\,p_h(y\mid x)
}
$$

即固定模型中的贝叶斯后验预测。

对任一竞争假说 \(h\)，由后验赔率恒等式：

$$
\log\frac{\Pr(h^*\mid c_n)}{\Pr(h\mid c_n)}
=
\log\frac{\pi_{h^*}}{\pi_h}
+\sum_{i=1}^{n}\log\frac{p_{h^*}(z_i)}{p_h(z_i)}.
$$

对 \(z_i\sim p_{h^*}\) 取期望：

$$
\boxed{
\mathbb E_{h^*}
\log\frac{\Pr(h^*\mid c_n)}{\Pr(h\mid c_n)}
=
\log\frac{\pi_{h^*}}{\pi_h}
+
nD_{\mathrm{KL}}(p_{h^*}\|p_h).
}
$$

只要 KL 散度大于零，真实假说相对于竞争假说的**期望对数赔率**随上下文长度线性增长，而 \(L^*\) 始终固定。

**探索结论：** 推理中的上下文适应可由持久状态固定的支撑调用实现；贝叶斯后验变化不是对模型持久参数进行训练更新。并非主张所有 ICL 都是贝叶斯算法。

### 外部证据

- Sang Michael Xie、Aditi Raghunathan、Percy Liang、Tengyu Ma，*An Explanation of In-context Learning as Implicit Bayesian Inference*，ICLR 2022。使用潜在概念／HMM 混合设定分析上下文推断，并在合成数据上检验相关 ICL 现象：[arXiv:2111.02080](https://arxiv.org/abs/2111.02080)。
- Madhur Panwar、Kabir Ahuja、Navin Goyal，*In-Context Learning through the Bayesian Prism*，ICLR 2024。在可处理任务族上发现高容量 Transformer 的预测接近贝叶斯基准，也报告偏离贝叶斯预测器的情形：[arXiv:2306.04891](https://arxiv.org/abs/2306.04891)；[ICLR 论文](https://proceedings.iclr.cc/paper_files/paper/2024/file/d81cd83e7f6748af351485d73-Paper-Conference.pdf)。

**证据结论：** 外部理论和实验直接支持“冻结模型也可通过上下文实现贝叶斯式推断”这一类现象；这里的后验赔率公式则在所列概率假设下代数成立。

---

## 4. 探索 E-I03：固定状态下的推理步数与目标边界

### [T]

TLIR 区分冻结推理与持久学习变化，并通过连续读出及判定边界表述能力结果：

$$
\Delta_{\mathrm{inf}}L=0,\qquad C=\chi(m).
$$

### [A] 数学条件

固定 \(L^*\)、输入及上下文。设瞬时推理工作状态（不是持久学习状态）满足迭代

$$
w_{k+1}=F_{L^*,x,c}(w_k).
$$

假设 \(F\) 在所考虑的完备度量空间中是压缩映射，\(0<\rho<1\)，具有不动点 \(w_\infty\)。对指定目标，**探索性地定义推理预算 \(k\) 的读出** \(m_{q,k}=m_q(L^*,w_k)\)，它不是训练时刻 \(t\) 的持久状态读出。设对 \(w\) 的 Lipschitz 常数为 \(M\)，且 \(m_q(L^*,w_\infty)\ge\gamma>0\)。

### [R] 推导

由压缩映射：

$$
\|w_k-w_\infty\|
\le \rho^k\|w_0-w_\infty\|.
$$

由 Lipschitz 条件：

$$
m_q(L^*,w_k)
\ge\gamma-M\rho^k\|w_0-w_\infty\|.
$$

故只要

$$
M\rho^k\|w_0-w_\infty\|<\gamma,
$$

推理预算 \(k\) 的输出即保证处于正边界。特别地，当初始误差上界不小于阈值时，一个充分条件为

$$
\boxed{
k>
\frac{\log\!\left(M\|w_0-w_\infty\|/\gamma\right)}
{\log(1/\rho)}.
}
$$

若初始上界已小于 \(\gamma\)，则 \(k=0\) 即充分。这里是严格的**充分步数条件**，不是一般 LLM 的经验步数定律。

**探索结论：** 在固定学习状态下，有限次瞬时迭代可改变特定推理预算下的能力读出。训练中的持久能力状态与推理预算相关的当前输出，必须分开表述。

### 外部证据

- Charlie Snell、Jaehoon Lee、Kelvin Xu、Aviral Kumar，*Scaling LLM Test-Time Compute Optimally Can be More Effective than Scaling Parameters for Reasoning*，ICLR 2025。实验表明，恰当分配测试时计算可以改善推理结果，部分 FLOPs 匹配情形下小模型加推理计算可超过更大模型：[ICLR proceedings](https://proceedings.iclr.cc/paper_files/paper/2025/hash/1b623663fd9b874366f3ce019fdfdd44-Abstract-Conference.html)。
- Wenkai Yang、Shuming Ma、Yankai Lin、Furu Wei，*Towards Thinking-Optimal Scaling of Test-Time Compute for LLM Reasoning*，NeurIPS 2025。实验发现过长推理在部分领域反而降低表现，并非更多计算总会更好：[NeurIPS proceedings](https://papers.neurips.cc/paper_files/paper/2025/hash/3e22bea3b170f4c2aebb9c48d98ae64d-Abstract-Conference.html)。

**证据结论：** 已有实验支持测试时计算会改变表现、且效果可能非单调；**它们没有直接检验此处的压缩映射假设和步数上界**。

---

## 5. 探索 E-I04：推理—反馈—训练闭环的稳定性条件

### [T]

$$
Y_t=\mathcal I(L_t,x_t,c_t),\qquad
B_{t+1}=\mathcal B(B_t,Y_t,E_t),
$$

$$
A_{t+1}=\mathcal U(Z_{t+1},D_{t+1},B_{t+1}),\qquad
Z_{t+2}=\mathcal T(Z_{t+1},A_{t+1}).
$$

### [A] 数学条件与对齐方式

为研究一个同步反馈周期，在保持 TLIR 原关系的前提下**按训练动作发生的周期重新编号**。固定外部数据与环境条件，写成

$$
Z_{t+1}=\mathcal T\bigl(Z_t,\mathcal U(Z_t,D,B_t)\bigr),
$$

$$
B_{t+1}=\mathcal B\bigl(B_t,\mathcal I(L_t,x,c),E\bigr).
$$

研究联合状态 \((Z_t,B_t)\) 在可微固定点附近的小扰动，记 \(\mathcal I_Z\) 为经 \(Z\mapsto L\) 对持久状态求导的链式导数。

### [R] 一般局部线性化

$$
\boxed{
J_{\mathrm{loop}}=
\begin{pmatrix}
\mathcal T_Z+\mathcal T_A\mathcal U_Z
&
\mathcal T_A\mathcal U_B\\
\mathcal B_Y\mathcal I_Z
&
\mathcal B_B
\end{pmatrix}.
}
$$

此处各偏导在相同固定点、固定外部条件下求值。若谱半径 \(\rho(J_{\mathrm{loop}})<1\)，固定点局部渐近稳定；若存在特征值模大于 1，则该线性化存在增长方向。

### [A] 二维线性特例

忽略优化器记忆，令

$$
Y_t=L_t,\qquad
L_{t+1}=(1-\alpha)L_t+\beta B_t,
$$

$$
B_{t+1}=\gamma B_t+\kappa Y_t,
$$

并取 \(0<\alpha<1\)、\(0<\gamma<1\)、\(\beta,\kappa\ge0\)。

于是：

$$
J=
\begin{pmatrix}
1-\alpha&\beta\\
\kappa&\gamma
\end{pmatrix},
\qquad
\det(I-J)=\alpha(1-\gamma)-\beta\kappa.
$$

在这些符号与范围条件下，\(J\) 是非负矩阵，其主特征值在

$$
\boxed{\beta\kappa=\alpha(1-\gamma)}
$$

处等于 1；\(\beta\kappa<\alpha(1-\gamma)\) 时谱半径小于 1，反向严格不等式时主特征值大于 1。

### [R] 与目标功能响应、能力边界联立

对局部状态更新中由反馈增量 \(\Delta B\) 导致的那一部分作用，若目标边界 \(m_q\) 可微，则有一阶特化：

$$
\Delta m_q
\simeq
\nabla_Zm_q^\top\,
\mathcal T_A\mathcal U_B\,
\Delta B.
$$

这个标量可以为正、为零或为负；所以反馈提高某个目标时，不保证所有其他目标同步获得正响应。累计响应是否越过其他目标的能力边界，仍依赖实际轨迹及读出。

**探索结论：** 在指定反馈模型下，可以从 TLIR 的闭环结构导出可计算的稳定性阈值与跨目标响应方向。这个数值临界式只属于所列线性特例，**不是现有 GRU 实验报告的实测阈值**。

### 外部证据

- Vincent Liu、Han Wang、Ruo Yu Tao、Khurram Javed、Adam White、Martha White，*Measuring and Mitigating Interference in Reinforcement Learning*，CoLLAs 2023，报告干扰测量与控制性能不稳定之间的联系：[PMLR](https://proceedings.mlr.press/v232/liu23a.html)。
- Brahma S. Pavse 等人，*Learning to Stabilize Online Reinforcement Learning in Unbounded State Spaces*，ICML 2024，研究在线 RL 的策略稳定性，并采用稳定性方法：[PMLR](https://proceedings.mlr.press/v235/pavse24a.html)。
- Ali Saheb Pasand 等人，*Stable Deep Reinforcement Learning via Isotropic Gaussian Representations*，ICML 2026，分析非平稳学习目标下的表示稳定性与训练不稳定：[PMLR](https://proceedings.mlr.press/v306/pasand26a.html)。

**证据结论：** 外部研究支持 RL／反馈系统中干扰和稳定性问题的实际存在，但**没有直接验证本节的 TLIR 局部 Jacobian 或特例阈值等式**。TLIR 自身的 RL-E05/E06 提供反馈分配、功能支撑变化、其他能力损失与恢复的独立实验锚点，不能重标为本节新实验。

---

## 6. 数学推论、外部证据及现有 TLIR 结果的对应表

| 探索 ID | 严格数学内容 | 外部证据的准确作用 | 记录性质 |
|---|---|---|---|
| E-I01 | 完整行为等价不蕴含同增量更新后的等价：明确代数反例；给出商空间转移的相容条件 | 函数等价表示与初始化依赖的相关理论／实验 | 新反例分析，未新增原生干预 |
| E-I02 | 指定假说模型下的贝叶斯预测及期望对数后验赔率恒等式 | 贝叶斯式 ICL 的已有理论与实验 | 原方程的概率模型特化 |
| E-I03 | 压缩推理迭代 + Lipschitz 读出下的能力边界充分步数 | 测试时计算的收益与非单调经验现象 | 条件性数学上界；非通用推理规模律 |
| E-I04 | 指定同步反馈闭环的 Jacobian、二维线性特例稳定阈值 | RL 干扰、策略／表示稳定性已有研究 | 条件性控制理论特化；非 GRU 实测阈值 |

**共同发现（探索性）：** 相同的 TLIR 状态—训练动作—持久功能组织—冻结推理—能力读出—反馈结构，在不同原有数学条件下，分别连接到行为可辨识性、贝叶斯后验、推理计算收敛和闭环稳定性等研究领域。这说明统一方程有**超出当前14项正式外推的数学特化空间**；不等于这些外部论文已经直接检验 TLIR 整体，也不把文献类比计为新原生实验。

## 7. 记录边界与出处

1. **已建立的 TLIR 来源**： [TLIR_UNIFIED_DYNAMICS_V5.md](TLIR_UNIFIED_DYNAMICS_V5.md)；[实验导航](EXPERIMENT_INDEX.md)；[INF-E01 原生冻结推理](experiments/gfg_nanogpt_training_learning_inference_projection_v1/)；[INF-G01 跨系统无 Attention 冻结推理](experiments/gfg_cross_system_frozen_inference_projection_v1/)；[RL-E05/E06 证据链](RL_EVIDENCE_CHAIN.md)。
2. **新增内容**：四项数学探索、对照文献表和证据分级，仅为本文件的研究笔记，不声称新冻结实验证据。
3. **未改动**：既有14项正式数学外推、所有冻结的实验协议、历史数据、结果与独立检查器。
4. **不要求用户继续增加实验**：本文件的用途是保存已有推导及公开文献对应关系；不是给已完成研究设立新的实验义务。

**简要结论：** TLIR 的推理方程 \(\mathcal I\) 与此前已建立的训练、功能响应、能力读出及反馈方程联立后，在明确数学条件下，可以推导出四组可核查的关系，并与多个独立研究领域的已发表理论和实验形成可审查的对应。这是一次**探索性数学外推与文献核对记录**，而非正式论文的新实验证明。
