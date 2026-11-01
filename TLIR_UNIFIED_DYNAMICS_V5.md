# TLIR 统一动力学方程组（V5）

**Training–Learning–Inference–Reinforcement (TLIR)**  
**定位：实验约束的统一动力学方程组。**

本文将 TL-E01–E08、INF-E01、TL-P01 和 RL-E01–E06 的实验关系组织为一个动力学系统：同一次真实训练更新驱动持久状态演化，产生目标特定的功能响应、分布式支撑变化和能力边界变化；冻结推理调用已有支撑，行为后果可以反馈进入下一次训练。**统一的是核心动力学关系，具体网络、优化器、读出与反馈实现可以不同。**

本文件为独立的方程组版本，不修改论文正文或冻结实验记录。

## 1. 变量与状态

在训练时刻 \(t\)：

\[
Z_t=(L_t,\omega_t).
\]

- \(L_t\)：决定模型功能的持久学习状态；在所研究的参数化网络中主要由 \(\theta_t\) 及相关持久变量实现。
- \(\omega_t\)：训练机制的内部状态，例如优化器记忆。
- \(Z_t\)：参数／学习状态与训练机制的联合接收状态。
- \(A_t\)：已实际形成并执行的训练动作，包括完成联合状态转移所需的实际更新信息。
- \(D_t\)：该次训练的输入信息；\(B_t\)：来自先前行为后果、可影响训练的反馈信息。
- \(q\)：具体目标；\(m_q(L)\)：目标特定的功能读出（在分类／边界实验中为有符号 margin）。
- \(P_{t,q}\)：固定干预协议下测得的目标功能支撑，可为向量。
- \(C_{t,q}\)：目标能力的可观察读出。

## 2. 统一耦合动力学主方程

\[
\boxed{
\begin{aligned}
\text{Training:}\quad
&A_t=\mathcal U(Z_t,D_t,B_t),\\
&Z_{t+1}=\mathcal T(Z_t,A_t),\\[4pt]
\text{Response:}\quad
&R_{t,q}=m_q(L_{t+1})-m_q(L_t),\\[4pt]
\text{Learning:}\quad
&P_{t,q}=\mathcal P_q(L_t;\mathcal J_q),\\
&\Delta P_{t,q}=P_{t+1,q}-P_{t,q},\\[4pt]
\text{Capability:}\quad
&C_{t+1,q}
 =\chi_q\!\left(m_q(L_t)+R_{t,q}\right),\\[4pt]
\text{Inference:}\quad
&S_{x,c}^{\mathrm{active}}=\Pi_{x,c}(L^*),\\
&Y_{x,c}=H_{x,c}(S_{x,c}^{\mathrm{active}}),\\
&L_{\mathrm{after}}^*=L_{\mathrm{before}}^*,\\[4pt]
\text{Feedback:}\quad
&B_{t+1}=\mathcal B(B_t,Y_t,E_t),\\
&A_{t+1}=\mathcal U(Z_{t+1},D_{t+1},B_{t+1}).
\end{aligned}
}
\]

其中 \(\mathcal J_q\) 是固定的组件干预和支撑测量协议；\(\chi_q\) 是任务的能力判定规则。对于有符号分类 margin，\(\chi_q(m)=\mathbf1[m>0]\)。\(x,c\) 分别表示推理输入与上下文；\(L^*\) 是冻结的持久学习状态；\(E_t\) 是行为的环境后果。反馈信息 \(B_t\) 可汇总此前完整交互历史。

**耦合关系：** \(R_{t,q}\)、\(\Delta P_{t,q}\) 与 \(C_{t+1,q}\) 均由同一次 \(Z_t\to Z_{t+1}\) 确定。它们是同一次真实演化的不同功能观测，而非三条彼此独立的状态轨迹。

## 3. 实验建立的动力学关系

### 3.1 实际动作—接收状态耦合（TL-E01–E03）

对于参数化网络，设一次实际参数更新为 \(\delta\)，目标响应定义为

\[
R_q(\theta,\delta)
=m_q(\theta+\delta)-m_q(\theta).
\]

相互交换实际更新的实验建立了以下状态依赖关系：

\[
\boxed{
\exists\,\theta_A,\theta_B,\delta,q:\quad
R_q(\theta_A,\delta)\ne R_q(\theta_B,\delta).
}
\]

**含义：** 相同参数增量的功能效应取决于接收状态。优化器记忆还参与实际更新的形成：

\[
\delta_t=\Psi(\theta_t,\omega_t,D_t,B_t).
\]

TL-E01记录能力形成、下降与恢复，发现损失、准确率、训练步数等单一观测量不足以完整表征这些变化。TL-E02的有效演化暂停实验进一步确立训练状态演化对能力形成的作用。

### 3.2 有限幅度非线性响应（TL-E04–E05）

令 \(\delta_t=\theta_{t+1}-\theta_t\) 为一次实际形成的更新，在固定接收状态上改变其施加幅度：

\[
\boxed{
r_{t,q}(\alpha)
=m_q(\theta_t+\alpha\delta_t)-m_q(\theta_t).
}
\]

实验记录了存在非线性响应：

\[
\boxed{
\exists\,t,q,\alpha:\quad
r_{t,q}(\alpha)\ne \alpha r_{t,q}(1).
}
\]

观察到的形态包括近线性、饱和、加速、回转与符号反转。这里 \(\alpha\) 是**对一次真实更新的实验幅度干预**，不是训练时间。

TL-E05识别了三个主要响应条件因子族：

\[
\boxed{
\mathcal F_{t,q}=(F1_{t,q},F3_{t,q},F5_t),
}
\]

分别为目标边界状态（F1）、目标特定更新几何（F3）及参数—优化器接收状态（F5）。这些是实验识别的主要条件因子族。

### 3.3 分布式功能支撑及非加性交互（TL-E06）

以固定学习状态 \(L\) 和组件干预集合 \(C\) 定义读出 \(f_q(L;C)\)。对组件 \(i,j\) 测量：

\[
\boxed{
I_{ij,q}
=f_{\mathrm{full}}-f_{-i}-f_{-j}+f_{-i,-j}.
}
\]

实验观察到非零二阶交互：

\[
\boxed{\exists\,i,j,q:\quad I_{ij,q}\ne 0.}
\]

组件必要性、备份、替代、协同及失效余量共同构成目标的干预式功能支撑测量 \(\mathcal P_q(L;\mathcal J_q)\)。

### 3.4 持久学习重组（TL-E07）

同一次真实更新引起：

\[
\boxed{
\Delta P_{t,q}
=\mathcal P_q(L_{t+1};\mathcal J_q)
-\mathcal P_q(L_t;\mathcal J_q).
}
\]

在相同可比测量协议下，实验记录 **1,636 次有效支撑分配转移**及 **228 次主要支撑组件交接**。

**Learning 是真实训练导致的持久功能组织变化。** \(\Delta P_{t,q}\) 和 \(R_{t,q}\) 由共同的状态变化产生，不规定其中一个必然先成为另一个的原因。

### 3.5 功能变化与能力边界（TL-E07–E08）

对于具有有符号 margin 的目标：

\[
\boxed{
\begin{aligned}
C_{t,q}&=\mathbf1[m_q(L_t)>0],\\
m_q(L_{t+1})&=m_q(L_t)+R_{t,q},\\
C_{t+1,q}&=\mathbf1[m_q(L_t)+R_{t,q}>0].
\end{aligned}
}
\]

因此，真实训练的目标响应与能力获得、维持、丧失、恢复之间具有明确的边界关系。实验记录到：

\[
\boxed{
\exists\,t,q:\quad
\Delta P_{t,q}\ne0
\;\land\;
C_{t+1,q}=C_{t,q}.
}
\]

**内部功能支撑可以发生持久重组，而外部正确性暂时保持不变。** 能力改变对应目标特定读出边界的跨越。

### 3.6 冻结推理与已学习版本依赖（INF-E01）

在冻结的持久学习状态 \(L^*\) 下：

\[
\boxed{
\begin{aligned}
S_{x,c}^{\mathrm{active}}&=\Pi_{x,c}(L^*),\\
Y_{x,c}&=H_{x,c}(S_{x,c}^{\mathrm{active}}),\\
L_{\mathrm{after}}^*&=L_{\mathrm{before}}^*.
\end{aligned}
}
\]

推理按查询和上下文调用、组合已形成的功能支撑；瞬时计算可以复杂，但持久学习状态固定。

将组件 \(j\) 替换为训练前的精确版本：

\[
\boxed{
D_{j,q}
=f(q;L^*)-f(q;L^{(j\leftarrow\mathrm{pre})}).
}
\]

INF-E01通过组件门控、查询条件效应、非加性交互、精确版本回滚与恢复检验了这些关系。

### 3.7 反馈闭环与再组织（RL-E01–E06）

行为的后果重新进入训练：

\[
\boxed{
\begin{aligned}
Y_t&=\mathcal I(L_t,x_t),\\
B_{t+1}&=\mathcal B(B_t,Y_t,E_t),\\
A_{t+1}&=\mathcal U(Z_{t+1},D_{t+1},B_{t+1}).
\end{aligned}
}
\]

反馈关系可以通过不同的奖励、信用分配或其他机制实现。**动力学核心是后果改变后续真实训练动作，而动作改变下一阶段的持久功能组织。**

RL-E01通过匹配因果分叉建立了后果绑定／反馈关系对实际参数更新和后续行为的作用；RL-E02–E04进一步检验了长期、历史依赖和随机条件中的信用关系。

RL-E06从同一个已掌握四技能的共享 GRU 状态出发，比较均衡、轻度集中、高度集中和完全集中的正确反馈分配。在已测试的剂量序列 \(d_1<d_2<d_3<d_4\) 中：

\[
\boxed{
\begin{aligned}
P_{\mathrm{target}}^{\mathrm{share}}(d_1)
&<P_{\mathrm{target}}^{\mathrm{share}}(d_2)
<P_{\mathrm{target}}^{\mathrm{share}}(d_3)
<P_{\mathrm{target}}^{\mathrm{share}}(d_4),\\
\overline m_{\mathrm{other}}(d_1)
&>\overline m_{\mathrm{other}}(d_2)
>\overline m_{\mathrm{other}}(d_3)
>\overline m_{\mathrm{other}}(d_4).
\end{aligned}
}
\]

实验结果：强化技能支撑份额由 **25.54%** 提升至 **35.20%**，其他技能平均 margin 由 **9.474** 降至 **3.962**。从完全集中反馈第 800 步的相同状态改为均衡反馈，其他技能准确率由 **70.31%** 恢复至 **99.48%**，被强化技能保持 **100%**。

**反馈分配与持续训练可以重新引导持久功能支撑，并改变已有能力的保持与恢复。**

## 4. 定量预测的具体实现：TL-P01

TL-E05的主要响应条件引出对**已经形成的实际更新**所导致的目标边界的预测。

令正确类别为 \(y\)，竞争类别为 \(c\)，隐表示为 \(h\)，并定义：

\[
b_c=w_y-w_c,\quad
\Delta b_c=\Delta w_y-\Delta w_c,
\quad h'=Dh[\delta_t],\quad
h''=D^2h[\delta_t,\delta_t].
\]

则预测的更新后类别边界为：

\[
\boxed{
\begin{aligned}
\widehat g_{y,c}={}&
b_c^\top h+\Delta b_c^\top h
+b_c^\top h'\\
&+\Delta b_c^\top h'
+\tfrac12 b_c^\top h''.
\end{aligned}
}
\]

预测正确性由下式判断：

\[
\widehat C_{t+1,q}
=\mathbf1\!\left[\min_{c\ne y}\widehat g_{y,c}>0\right].
\]

nanoGPT确认集结果：**4,652／5,088 = 91.43%** 预测准确率，**91.49%** 四类能力转移宏平均召回率。

TL-P01给出了上述统一关系的可计算预测实例。不同系统可以使用适合其状态与算子特性的预测实现。

## 5. 数学结构与实验关系对应

| 动力学关系 | 对应实验 | 数学表达 |
|---|---|---|
| 状态与真实更新形成 | TL-E01–E03 | \(Z_t,A_t\mapsto Z_{t+1}\) |
| 接收状态依赖 | TL-E03 | \(R_q(\theta_A,\delta)\ne R_q(\theta_B,\delta)\) 的实测实例 |
| 有限幅度非线性 | TL-E04 | \(r_{t,q}(\alpha)\ne\alpha r_{t,q}(1)\) 的实测实例 |
| 响应主要条件因子 | TL-E05 | F1／F3／F5 |
| 分布式非加性支撑 | TL-E06 | \(I_{ij,q}\ne0\) 的实测实例 |
| 持久支撑重组 | TL-E07 | \(\Delta P_{t,q}=\mathcal P_q(L_{t+1})-\mathcal P_q(L_t)\) |
| 能力读出边界 | TL-E08 | \(C_{t+1,q}=\chi_q(m_{t,q}+R_{t,q})\) |
| 冻结推理 | INF-E01 | \(Y=H(\Pi(L^*)),\,\Delta_{\mathrm{inf}}L=0\) |
| 目标边界预测 | TL-P01 | \(\widehat g_{y,c}\) |
| 反馈改变后续训练 | RL-E01–E04 | \(B_{t+1}\mapsto A_{t+1}\) |
| 反馈驱动支撑再组织 | RL-E05–E06 | 反馈剂量、margin、支撑份额及恢复干预 |

## 6. 统一表达

\[
\boxed{
\begin{gathered}
\text{接收状态 + 真实训练动作}\\
\Downarrow\\
\text{同一次持久状态演化}\\
\Downarrow\\
\left\{
\begin{array}{l}
\text{目标特定功能响应}\\
\text{分布式功能支撑重组}\\
\text{目标能力边界变化}
\end{array}
\right.\\
\Downarrow\\
\text{固定学习状态下调用与组合已形成的支撑}\\
\Downarrow\\
\text{行为后果反馈至后续真实训练动作}
\end{gathered}
}
\]

**TLIR将Training、Learning、Inference和Reinforcement组织为同一人工神经系统中的耦合动力学关系：真实训练动作的功能效应依赖接收状态并可表现出非线性；训练持续重组分布式功能支撑；能力变化由目标特定边界读出；推理在固定学习状态下调用已有支撑；反馈将行为后果带回后续训练，进一步改变功能组织与能力。核心关系统一，具体实现自由。**

## 7. 论文与证据导航

- [当前论文逐节实验对应图](TLIR_PAPER_EVIDENCE_MAP.md)
- [原始与扩展实验索引](EXPERIMENT_INDEX.md)
- [TL-E03／E04：真实更新交换与幅度路径](experiments/gfg_nanogpt_stepwise_support_transition_v1/)
- [TL-E05：响应因子分析](experiments/gfg_nanogpt_response_factor_analysis_v1/)
- [TL-E06：分布式功能支撑](experiments/gfg_nanogpt_support_redundancy_v1/)
- [TL-E07：支撑重新分配](experiments/gfg_nanogpt_support_reallocation_audit_v1/)
- [TL-E08：能力边界](experiments/gfg_nanogpt_identity_aligned_margin_crossing_v1/)
- [INF-E01：冻结推理](experiments/gfg_nanogpt_training_learning_inference_projection_v1/)
- [TL-P01：实际更新目标边界预测](experiments/gfg_nanogpt_actual_update_boundary_v1/RESULTS.md)
- [反馈实验与证据链](RL_EVIDENCE_CHAIN.md)
- [21组统一解释与14项外推](TLIR_SECTION5_21_PAIRS_14_EXTRAPOLATIONS.md)
