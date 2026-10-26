# TLIR theory-first conditional extrapolations: 14 derivations and falsification targets

**Research-record companion to the Science manuscript, prepared from the author's GFG/TLI/TLIR handoff.**

The following material reproduces the scientific portion of the original research handoff's Section 36, comprising **14 separate derivation/comparison records**, the repeated higher-level invariants, and the original procedural independence caveat. The manuscript discusses three representative cases; this companion retains all fourteen for scientific inspection.

**Critical interpretation:** The listed consequences are conditional and differ in evidence strength. The session-local derivation-before-search procedure is **not a cryptographically blinded or independently timestamped preregistration**: pretrained model knowledge may overlap with cited historical results. Literature agreements are not claims that the cited papers directly tested the entire TLIR theory. Some cases are mathematical specializations or conceptual correspondences and must not be promoted to universal laws. External references and individual equations should be independently checked prior to formal new-paper use. No new native experiments were performed to create this file.

The original TLIR evidence remains separately organized at [the paper-to-evidence map](TLIR_PAPER_EVIDENCE_MAP.md). The related [21 historical positive/negative stress tests](TLIR_HISTORICAL_STRESS_TEST_21.md) are distinct from the theory-first derivations below.

---

## 三十六、TLIR 理论外推—外部独立确认（theory-first external confirmation）

这条证据线必须与“21 组历史正反统一压力测试”分开。后者是：TLIR 核心关系先由独立真实系统实验形成，再去面对既有历史正反结果；本节更进一步，要求**先从已经建立的 TLIR 关系自然推出一个新的定律/定理级命题，冻结推导，再检索外部文献，看独立研究是否已经观测、证明、部分支持或反驳该命题**。

固定流程：

\[
\boxed{
\text{TLIR core relations}
\rightarrow
\text{theory-first deduction}
\rightarrow
\text{freeze assumptions / equations / predicted boundary}
\rightarrow
\text{external literature search}
\rightarrow
\text{proved / observed / partial / unresolved / contradicted}
}
\]

这类证据不是普通 post-hoc compatibility。关键条件是：外部结果没有参与本次局部推导的形成，而推导完成后才被检索出来。它可以记为 **theory-first external confirmation**。如果外部工作给出严格定理，则是 theorem-level external confirmation；若给出真实系统观测，则是 empirical external confirmation。它仍不应被偷换成“外部论文直接证明 TLIR 整体”，但它是强独立外部证据，因为 TLIR 用已有核心关系重新导出了由其他研究路线独立得到的结果。

### 36.1 Edge of Stability：局部稳定边界是演化中 receiver-dependent dynamics 的投影

先从 TLIR 写真实训练映射：

\[
S_{t+1}=F(S_t,A_t),\qquad A_t=\pi(S_t),
\]

因此完整一步动力学为

\[
S_{t+1}=\mathcal F(S_t)=F(S_t,\pi(S_t)).
\]

真正的局部稳定性对象应是完整状态映射的 Jacobian：

\[
\boxed{\rho(D\mathcal F(S_t))\approx 1}
\]

而不是先验地等同于 raw Hessian 阈值。对 vanilla gradient descent 的 parameter-only、local-quadratic 特例：

\[
\theta_{t+1}=\theta_t-\eta\nabla L(\theta_t),\qquad
J_t=I-\eta H_t,
\]

沿 Hessian eigenmode 得到

\[
|1-\eta\lambda_i|<1,
\]

于是局部 marginal boundary 为

\[
\boxed{\lambda_{\max}=2/\eta}.
\]

TLIR 的外推不是否认该式，而是把它定位为：**冻结当前 receiver、采用局部 quadratic projection 时的边界**。真实有限训练动作同时改变 receiver、curvature、eigendirection、optimizer state 与 functional organization，因此瞬时满足 \(\lambda_{\max}>2/\eta\) 不推出完整多步系统必然发散。自然预测是：超过局部边界后的 finite response 可以反过来改变 receiver/curvature，使系统重新进入或贴近 marginal region；对 Adam / momentum 等带状态优化器，真正 edge 应依赖联合 parameter–optimizer receiving state，而不应普遍等于 raw Hessian 的 \(2/\eta\)。

外部确认：

- Cohen et al., *Gradient Descent on Neural Networks Typically Occurs at the Edge of Stability* (2021, arXiv:2103.00065)：真实神经网络 full-batch GD 中 \(\lambda_{\max}\) 长期停留在略高于 \(2/\eta\) 附近，短时 loss 非单调但长期仍下降。对应“瞬时局部不稳定 ≠ 长期真实动力学发散”。
- Lewkowycz et al., *The large learning rate phase of deep learning: the catapult mechanism* (2020, arXiv:2003.02218)：超临界大 learning rate 下，网络可经历 loss 爆涨、kernel/curvature 改变后重新进入可收敛状态。对应 finite response 改变后续 receiver。
- Damian, Nichani & Lee, *Self-Stabilization: The Implicit Bias of Gradient Descent at the Edge of Stability* (ICLR 2023; arXiv:2209.15594)：给出 self-stabilization 理论，说明不稳定方向上的运动通过高阶项降低 curvature，并可近似为受 \(S(\theta)\le 2/\eta\) 约束的 projected dynamics。它为“边界附近存在动态回推机制”提供理论级外部确认。
- Cohen et al., *Adaptive Gradient Methods at the Edge of Stability* (NeurIPS 2022; arXiv:2207.14484)：Adam 相关稳定对象是 preconditioned Hessian；在 \(\beta_1=0.9\) 时经验阈值约为 \(38/\eta\)，并可通过适配 preconditioner 进入 raw-curvature 很高区域。直接支持“optimizer state / action geometry 改变真正稳定边界；\(2/\eta\) 不是普适神经动力学常数”。

因此本例的 TLIR 定位是：

\[
\boxed{
\text{Edge of Stability}
=
\text{evolving receiver-dependent finite dynamics 的局部冻结稳定投影}
}
\]

证据级别：**theory-first external empirical + theoretical confirmation**。

### 36.2 Post-capability learning / implicit bias：能力形成后 Learning 继续，并由 residual training-action weighting 组织 boundary geometry

从 TLIR 已建立关系出发：

\[
\boxed{\text{Learning}\neq\text{Capability formation}}
\]

因此，即使 classification capability 已经形成，只要 actual training action 仍非零，persistent Learning 仍可继续。对线性可分二分类，定义 identity-aligned margin

\[
m_i=y_i w^\top x_i.
\]

当所有 \(m_i>0\) 时，训练集 correctness 已经达到 100%，但 logistic / cross-entropy 的 gradient 仍不为零。大 margin 下：

\[
-\nabla L(w)\propto \sum_i y_i x_i e^{-m_i}.
\]

因此 capability 形成以后，training action 会指数集中到 margin 最小、最靠近 readout boundary 的样本。令

\[
w=r u,\qquad \|u\|=1,\qquad \gamma_i(u)=y_i u^\top x_i,
\]

则大 \(r\) 时

\[
L(w)\approx \sum_i e^{-r\gamma_i(u)}
\]

由最小 normalized margin 主导，因此继续降低 loss 的方向问题自然趋向：

\[
\boxed{
\max_{\|u\|=1}\min_i y_i u^\top x_i
}
\]

即 max-margin organization。这里 max-margin 不是 TLIR 额外加入的规则，而是三条既有关系的特例：

\[
\boxed{
\text{Learning}\neq\text{Capability}
+
\text{actual-action dependence}
+
\text{readout boundary}
}
\]

对 exponential-tail loss，若方向已接近最大间隔方向 \(u^\star\)，径向动作可近似满足

\[
\frac{dr}{dt}\propto e^{-\gamma^\star r},
\]

从而自然得到

\[
\boxed{r(t)\sim \frac{1}{\gamma^\star}\log t},
\]

即 capability 早已稳定，而参数尺度 / margin 仍缓慢继续组织。

外部严格定理确认：

- Soudry et al., *The Implicit Bias of Gradient Descent on Separable Data* (JMLR 2018, 19:70)：证明对线性可分数据的 unregularized logistic regression，gradient descent 的 predictor direction 收敛到 hard-margin SVM / max-margin direction；论文还明确指出这解释了为什么 training error 已为零后继续优化仍有意义。其渐近结构包含主导 \(\log t\) 增长。
- Nacson et al., *Convergence of Gradient Descent on Separable Data* (AISTATS 2019)：证明广泛 super-polynomial tail loss 可趋向 \(L_2\) max-margin，而更重尾 loss 不一定如此。它命中 TLIR 的边界预测：**改变 residual action weighting / loss tail，会改变长期 implicit organization**。
- Ji et al., *Gradient descent follows the regularization path for general losses* (COLT 2020)：进一步证明 exponentially-tailed losses 必然趋向 maximum-margin direction，而 polynomially-tailed losses 可趋向 poor-margin direction。再次确认 endpoint 依赖实际 loss/action law，而不是“classification 已正确”这一 observable capability 本身。
- Gunasekar et al., *Characterizing Implicit Bias in Terms of Optimization Geometry* (ICML 2018)：Mirror Descent、Natural Gradient、不同 norm 的 Steepest Descent 可产生不同 implicit bias。对应 TLIR 的 actual-action / optimizer-geometry dependence：objective 相同不意味着 persistent Learning endpoint 相同。
- Lyu & Li, *Gradient Descent Maximizes the Margin of Homogeneous Neural Networks* (ICLR 2020)：把上述结构扩展到 ReLU / LeakyReLU 等 homogeneous neural networks；在 logistic / cross-entropy training 下，normalized margin 持续增加并趋向相关 margin-maximization constrained problem 的 KKT point。它说明“capability 形成后 boundary geometry 仍继续组织”并非线性模型专属。

因此可记录一个 TLIR 上位原则：

\[
\boxed{
\begin{aligned}
&\textbf{Post-capability formation principle:}\\
&C_q\ \text{达到稳定正确边界后，只要 actual training action 仍非零，Learning 可继续；}\\
&\text{其长期方向由 residual feedback/action weighting 对 boundary geometry 的选择性作用决定。}
\end{aligned}
}
\]

在 separable classification + exponential-tail loss 的局部 regime 下，它特殊化为：

\[
\boxed{
\text{post-capability learning}\rightarrow\text{max-margin organization}
}
\]

证据级别：**theory-first external theorem-level confirmation**，且包含正结果（max-margin）与边界结果（loss tail / optimizer geometry 改变后 endpoint 改变）。

### 36.3 Recursive feedback contraction：递归反馈中的信息收缩与真实数据锚定

这一例专门对应 TLIR 的 feedback closure，而不是 Training/Learning 侧的又一个局部现象。由已建立闭环

\[
\boxed{
Inference_t
\rightarrow
Data/Feedback_{t+1}
\rightarrow
Training_{t+1}
\rightarrow
Learning_{t+1}
\rightarrow
Inference_{t+1}
}
\]

先不读取 model-collapse 文献结论，只考虑原始数据分布中某个低概率但真实存在的 mode / tail。记它在第 \(t\) 代有效训练分布中的质量为 \(q_t\)。若当前 learned system 经 inference 生成下一代训练数据时，对这一 mode 的平均保留率为

\[
0<\rho<1,
\]

并且后续训练数据完全由上一代模型生成数据替换，则最简单递推为

\[
\boxed{q_{t+1}=\rho q_t}.
\]

因此

\[
\boxed{q_t=q_0\rho^t\rightarrow 0}.
\]

这给出一个 TLIR 原生外推：

\[
\boxed{
\textbf{Recursive feedback contraction law:}
\quad
\text{在闭环中被系统性欠保留的信息，若持续由闭环自身替换生成，将按乘性递推收缩。}
}
\]

它不依赖“AI-generated data 天生有害”这一标签；真正决定方向的是 inference 输出进入下一轮 training 时，对原始 formation space 中各类信息的保留/丢失结构。

若每一代仍持续注入比例 \(\alpha\) 的真实外部数据，其中该 mode 的真实质量为 \(q_{\mathrm{real}}\)，其余 \(1-\alpha\) 来自模型生成数据，则

\[
\boxed{
q_{t+1}=(1-\alpha)\rho q_t+\alpha q_{\mathrm{real}}.
}
\]

只要 \(\alpha>0\)，该线性闭环具有非零固定点

\[
\boxed{
q^\star=
\frac{\alpha q_{\mathrm{real}}}
{1-(1-\alpha)\rho}>0.
}
\]

因此同一 TLIR 闭环自然推出一组正反结果：

\[
\boxed{
\alpha=0
\Rightarrow
\text{纯递归 synthetic feedback 可使 tail / minority information 持续收缩}
}
\]

而

\[
\boxed{
\alpha>0
\Rightarrow
\text{持续真实数据锚定可形成非零稳定 information floor}
}
\]

这里的关键不是某个特定 generative model，而是：

\[
\boxed{
\text{Inference output 是否以有偏、收缩的方式重新成为 Training input。}
}
\]

外部独立确认：

- Shumailov et al., *AI models collapse when trained on recursively generated data* (Nature, 2024)：递归使用前代模型生成数据训练后代模型时，原始分布 tails / minority modes 会优先丢失，随后支持进一步收缩并出现 model collapse；论文同时给出离散与 Gaussian 情形的理论分析。对应 \(q_{t+1}=\rho q_t\) 型闭环收缩。
- Gerstgrasser et al., *Is Model Collapse Inevitable? Breaking the Curse of Recursion by Accumulating Real and Synthetic Data* (ICML 2024)：若每一代以 synthetic data 替换真实数据，误差可随递归恶化；若持续保留/累积真实数据，则 collapse 可被避免，并可得到与递归代数无关的受控误差上界。对应 \(\alpha>0\) 后存在稳定外部锚定。
- Fu et al., *Towards Theoretical Understandings of Self-Consuming Generative Models* (ICML 2024)：从理论上分析 self-consuming generative loops，说明当每轮混合中的真实数据比例/数量足够时，synthetic distribution 与真实分布之间的偏离可以被控制。再次对应外部信息注入改变闭环固定点，而不是“self-generated”这一标签单独决定结局。

因此本例的 TLIR 定位是：

\[
\boxed{
\text{model collapse / anti-collapse}
=
\text{Inference}\rightarrow\text{Feedback/Data}\rightarrow\text{Training 闭环中的信息保留率与外部锚定共同决定的动力学结果}
}
\]

证据级别：**theory-first external empirical + theoretical confirmation**，且同时命中正结果（recursive replacement 导致 collapse）与边界/反结果（真实数据持续锚定可避免 collapse）。

### 36.4 Spectral Bias / Frequency Principle：频率不是本体，receiver–action response spectrum 才是上位对象

本轮按 theory-first 流程先从 TLIR 已建立的 current receiver + actual action → functional response 关系出发，不先读取目标文献结论。对平方损失 / gradient descent 的局部冻结 receiver regime，有

\[
\Delta f_t
\approx
J_t\Delta\theta_t
=
-\eta J_tJ_t^\top r_t
=
-\eta K_t r_t.
\]

若

\[
K_t\phi_k=\lambda_k\phi_k,
\]

则残差模态满足

\[
r_{k,t+1}\approx(1-\eta\lambda_k)r_{k,t},
\qquad
r_k(t)\approx e^{-\eta\lambda_k t}r_k(0).
\]

因此冻结的 TLIR 预测不是“神经网络天生先学低频”，而是：

\[
\boxed{
\text{functional mode 的形成时标由当前 receiver 对当前 training action 的 response gain 决定。}
}
\]

在局部 kernel regime 中可写成

\[
\boxed{
\tau_k^{-1}\propto\lambda_k.
}
\]

低频优先只是在某些 receiver / data / architecture 下，低频方向恰好具有更大的 \(\lambda_k\) 的特殊情形。于是 theory-first 冻结边界是：改变 representation、data geometry、loss geometry、initialization 或 architecture，只要改变 response spectrum，就应改变 frequency ordering；若高频 mode 的 effective gain 被提高，spectral bias 应减弱、消失，甚至反转。长期 feature learning 又会令 \(K_t\) 本身演化，因此固定 kernel 的频率排序只应是局部投影，不是完整训练本体。

外部独立确认：

- Cao et al., *Towards Understanding the Spectral Bias of Deep Learning* (IJCAI 2021, DOI 10.24963/ijcai.2021/304)：严格证明训练可沿 NTK eigenfunctions 分解，每个方向的收敛率由对应 eigenvalue 决定；在球面分布特例下低阶 spherical harmonics 更易学习。它直接命中 \(\lambda_k\rightarrow\tau_k\) 的冻结预测。
- Basri et al., *The Convergence Rate of Neural Networks for Learned Functions of Different Frequencies* (NeurIPS 2019)：从 kernel eigenspectrum 推导不同频率函数的学习时间，并用实验验证；说明 frequency bias 是谱结构的表现，而不是单独的“频率标签定律”。
- Tancik et al., *Fourier Features Let Networks Learn High Frequency Functions in Low Dimensional Domains* (NeurIPS 2020)：通过 Fourier feature mapping 改变 effective NTK 的 bandwidth，使标准 MLP 原本极慢的高频收敛显著改善。对应“改变 receiver / representation → 改变 spectrum → 改变 formation ordering”。
- Yu, Yang & Townsend, *Tuning Frequency Bias in Neural Network Training with Nonuniform Data* (ICLR 2023; arXiv:2205.14300)：对 nonuniform data 给出频率偏置理论，并通过 Sobolev loss 放大、削弱、抵消甚至反转 intrinsic frequency bias。它直接命中 theory-first 冻结的“ordering 可因 action/loss geometry 改变甚至反转”。

因此本例的 TLIR 定位是：

\[
\boxed{
\text{Spectral Bias}
=
\text{receiver/action response spectrum 在 frequency basis 下的局部 manifestation}
}
\]

更上位地说：

\[
\boxed{
\text{frequency 不是基本动力学变量；response eigenmode 才是。}
}
\]

证据级别：**theory-first external theorem-level core confirmation + empirical boundary confirmation**。

真正会打击该外推的是：在严格 matched receiver / action 的条件下，人为显著重排 response eigenspectrum 后，modal learning order 仍稳定只由频率标签决定；或者跨大量体系证明 \(\lambda_k\) 对 formation timescale 没有系统关系。

### 36.5 Benign Overfitting：插值不是决定量，action-selected solution × population functional geometry 才决定是否 benign

本轮先从最干净的过参数线性回归特例推导。令

\[
y=X\beta^\star+\varepsilon,\qquad p>n.
\]

平方损失、从 \(\beta_0=0\) 开始 gradient descent 时，每一步

\[
\Delta\beta_t\in\operatorname{row}(X),
\]

若最终达到插值 \(X\hat\beta=y\)，则 action law 选中的 row-space 解为

\[
\hat\beta=X^\top(XX^\top)^{-1}y.
\]

定义

\[
P_X=X^\top(XX^\top)^{-1}X,
\]

则

\[
\hat\beta-\beta^\star
=
-(I-P_X)\beta^\star
+
X^\top(XX^\top)^{-1}\varepsilon.
\]

对真实输入协方差 \(\Sigma\)，population excess risk 为

\[
R=(\hat\beta-\beta^\star)^\top\Sigma(\hat\beta-\beta^\star).
\]

条件于 \(X\) 并对零均值独立噪声取期望后，交叉项为零，得到两类来源：

\[
\boxed{
R
=
\underbrace{\beta^{\star\top}(I-P_X)\Sigma(I-P_X)\beta^\star}_{\text{未形成 / 未覆盖的 signal}}
+
\underbrace{\sigma^2\operatorname{tr}\!\left[\Sigma X^\top(XX^\top)^{-2}X\right]}_{\text{拟合噪声方向对 population 的影响}}
}
\]

因此 theory-first 冻结命题是：

\[
\boxed{
\text{Interpolation 本身不是 generalization 的决定量。}
}
\]

真正决定 benign / harmful 的，是 actual training dynamics 在多个 interpolating solutions 中选择了哪个 learned state，以及为了吸收训练噪声而形成的方向，在 population/readout geometry 中是否具有低 functional impact。

由此冻结出四个边界预测：

1. 同样 0 training error，可以有完全不同的 test error；
2. 不同 optimizer / initialization / norm geometry 即使都达到插值，也可选中不同 solution，导致不同 generalization；
3. benign overfitting 应出现在“吸收噪声的可用方向很多，但这些方向单个对 population prediction 影响很小”的谱结构中；
4. 若噪声被迫进入 population-weight 很大的方向，interpolation 应变得 harmful。

外部独立确认：

- Bartlett, Long, Lugosi & Tsigler, *Benign Overfitting in Linear Regression* (PNAS 2020, DOI 10.1073/pnas.1907378117)：给出 minimum-norm interpolator 近最优预测的条件，以 data covariance 的 effective rank 表述；论文明确指出，在该 setting 中必须有大量“对 prediction 不重要”的参数方向，而且数量要显著超过样本量。直接命中“低 population-impact noise-support directions”这一冻结预测。
- Hastie, Montanari, Rosset & Tibshirani, *Surprises in High-Dimensional Ridgeless Least Squares Interpolation* (Annals of Statistics 2022, DOI 10.1214/21-AOS2133)：定量分析 ridgeless interpolation 的 high-dimensional risk 与 double-descent / overparameterized regime，进一步说明 zero training error 本身不能决定 test risk。
- Gunasekar et al., *Characterizing Implicit Bias in Terms of Optimization Geometry* (ICML 2018)：在 underdetermined linear models 与 separable classification 中研究不同 Mirror Descent / Natural Gradient / Steepest Descent geometry 如何选择不同 global minima。它直接对应 TLIR 的 actual-action / optimizer-geometry dependence：同一个 objective 与同样的 zero-error boundary，不保证形成同一个 learned state。

因此本例的 TLIR 定位是：

\[
\boxed{
\text{Benign overfitting}
=
\text{training-boundary interpolation}
+
\text{low-population-impact noise-support formation}
}
\]

更一般地：

\[
\boxed{
\text{generalization of an interpolator}
\neq
\text{interpolation status};
\qquad
\text{它取决于 action-selected learned state 与 population functional geometry。}
}
\]

证据级别：**theory-first external theorem-level confirmation**。

真正的强反例是：跨 matched data covariance、matched interpolating boundary 与 matched population geometry，仅“是否插值”这一二值标签就足以决定 generalization，而 optimization-selected solution / noise-support geometry 不再提供任何额外信息。

### 36.6 Scaling Laws / Chinchilla：幂律是 formation spectrum 与 active bottleneck regime 的宏观投影

本轮先从 response-mode dynamics 推导幂律来源。若未形成的功能误差可分解为局部 response modes，单个模态近似满足

\[
e_k(t)\sim e^{-\lambda_k t}.
\]

若小 response-rate 区域的 mode density 具有 scale-free 尾部

\[
\rho(\lambda)\sim c\lambda^{\alpha-1},\qquad \lambda\to0^+,
\]

则总 residual 的连续近似为

\[
L(t)-L_\infty
\sim
\int_0^\infty\rho(\lambda)e^{-2\lambda t}\,d\lambda
\propto t^{-\alpha}.
\]

因此冻结的第一条 TLIR 命题是：

\[
\boxed{
\text{Power law}
=
\text{broad / scale-free formation-rate spectrum 的宏观投影。}
}
\]

exponent \(\alpha\) 取决于 receiver / action / data 共同诱导的慢模态谱，不应被当成跨体系不变的宇宙常数。

再把 model size \(N\) 与 data exposure \(D\) 看成两个不同 formation bottleneck。在一个稳定 regime 中若

\[
L(N,D)-L_\infty\approx AN^{-a}+BD^{-b},
\]

而 dense training compute 近似

\[
C\propto ND,
\]

则在固定 \(C\) 下最小化可得到

\[
\boxed{
N_\star\propto C^{\frac{b}{a+b}},
\qquad
D_\star\propto C^{\frac{a}{a+b}}
}
\]

以及

\[
\boxed{
L_\star(C)-L_\infty
\propto
C^{-\frac{ab}{a+b}}.
}
\]

由此 theory-first 冻结预测是：model/data/compute scaling 的 power law 来自 formation spectrum 与 bottleneck competition；architecture、optimizer、data distribution、training schedule 或 feature-learning regime 改变 response spectrum / bottleneck exponent 后，最优 \(N:D\) allocation 与 observed exponent 都应改变。Kaplan 与 Chinchilla 可以在不同 experimental regime 中同时正确，而不要求存在唯一 universal token/parameter ratio。

外部独立确认：

- Kaplan et al., *Scaling Laws for Neural Language Models* (2020, arXiv:2001.08361)：实证发现 language-model cross-entropy loss 对 model size、dataset size、compute 呈跨多个数量级的 power law，并据此给出 fixed-compute allocation。
- Hoffmann et al., *Training Compute-Optimal Large Language Models* (2022, arXiv:2203.15556)：扫描超过 400 个模型后得到与 Kaplan regime 不同的 compute-optimal allocation；model size 与 training tokens 应近似等比例扩展。它直接表明“scaling law 稳定”不意味着“某一 allocation ratio 是完整动力学常数”。
- Bahri et al., *Explaining Neural Scaling Laws* (PNAS 2024)：提出 variance-limited 与 resolution-limited 等不同 scaling regimes，并在 large-width / kernel 情形把 scaling exponent 与 kernel spectrum、data manifold resolution 联系起来。对应 formation-spectrum 与 active-bottleneck 两部分冻结预测。
- Bordelon, Atanasov & Pehlevan, *A Dynamical Model of Neural Scaling Laws* (ICML 2024)：用 gradient-descent random-feature 动力学同时研究 training time、dataset size、model size scaling，并推出非对称 compute-optimal scaling；说明不同 bottleneck 可以具有不同 exponent。
- Bordelon, Atanasov & Pehlevan, *How Feature Learning Can Improve Neural Scaling Laws* (2024, arXiv:2409.17858)：在超越 kernel limit 的可解模型中证明 feature learning 对 hard tasks 可显著改善 training/compute scaling exponent，并改变 compute-optimal strategy。它直接命中“receiver/kernel 演化会改变 exponent”的冻结边界。

因此本例的 TLIR 定位是：

\[
\boxed{
\text{Neural scaling law}
=
\text{formation spectrum}
+
\text{active resource bottleneck regime 的宏观投影}
}
\]

证据级别：**theory-first external theorem/theory + large-scale empirical confirmation**。

需要严格保留的边界：这里的 \(\rho(\lambda)\rightarrow t^{-\alpha}\) 是 TLIR 的上位可解特例，不应偷换成“上述外部论文逐项证明了这一 exact integral derivation”。外部确认的是谱产生幂律、资源瓶颈分 regime、feature learning 可改变 exponent / compute-optimal strategy 这些结构性命题。

真正会打击这一外推的是：跨 architecture / data / optimizer / feature-learning regime 大幅改变 response spectrum 与 bottleneck structure 后，scaling exponent 与 compute-optimal allocation 仍保持同一固定常数，而且这些 TLIR 条件变化不提供任何解释增量。

### 36.7 Dynamical Isometry：良好初始化是 receiver action-response transport 良条件化的特殊 regime

从 TLIR 的 receiver-conditioned response 出发，某层参数动作 \(\Delta\theta_\ell\) 对输出的一阶作用可写成

\[
\delta f
\approx
\sum_\ell J_{L\leftarrow\ell}B_\ell\Delta\theta_\ell.
\]

反向 training action 同样通过 \(J_{L\leftarrow\ell}^\top\) 传播。因此若跨层 Jacobian 的 singular values 大量趋近 0 或爆大，当前 receiver 会把不同层/方向的 action-response gain 压成极端各向异性；若 singular values 集中在 1 附近，则 forward response 与 backward action 可跨深度保持量级。

所以冻结的 TLIR 上位命题是：

\[
\boxed{
\text{Dynamical isometry}
=
\text{当前 receiver 的跨层 action-response transport 被良好条件化的特殊 regime。}
}
\]

由此冻结四个预测：

1. \(s_i(J)\approx1\) 时深层训练应更容易，因为作用不会沿深度系统性消失或爆炸；
2. exact isometry 不应是完整 Learning 的普遍必要条件，residual routing、normalization、gating、optimizer preconditioning 等只要让有效 action-response operator 保持可用，也可训练；
3. initialization 的影响应在 early receiver 尚未重组时最强，训练后的 Jacobian spectrum 可以继续演化，因此 initial criticality 不是永久动力学定律；
4. activation / architecture / depth 改变后，保持良好 conditioning 的初始化条件也应改变，不存在与 receiver 无关的 universal variance constant。

外部独立确认：

- Saxe, McClelland & Ganguli, *Exact Solutions to the Nonlinear Dynamics of Learning in Deep Linear Neural Networks* (ICLR 2014; arXiv:1312.6120)：给出 deep linear learning 的精确动力学，并发现特殊 orthogonal initialization 可使很深网络保持近似 depth-independent 的学习时间，同时保持梯度传播。
- Pennington, Schoenholz & Ganguli, *Resurrecting the Sigmoid in Deep Learning Through Dynamical Isometry* (NeurIPS 2017; arXiv:1711.04735)：把学习速度与 input-output Jacobian 的完整 singular-value distribution 联系起来；singular values 聚集于 1 的 dynamical-isometry regime 可使深网络学习快多个数量级，并且条件依赖 nonlinearity / initialization。
- Xiao et al., *Dynamical Isometry and a Mean Field Theory of CNNs: How to Train 10,000-Layer Vanilla Convolutional Neural Networks* (ICML 2018)：用理论导出的 orthogonal convolution initialization 实际训练 10,000 层以上 vanilla CNN，说明有效跨层 response transport 本身具有直接工程后果。
- Tarnowski et al., *Dynamical Isometry is Achieved in Residual Networks in a Universal Way for any Activation Function* (AISTATS 2019)：在 residual architecture 中得到更普适的 Jacobian spectral condition，说明 DI 条件会随 architecture/routing 改变，而不是固定 activation 常数。
- Bachlechner et al., *ReZero is All You Need: Fast Convergence at Large Depth* (UAI 2021; arXiv:2003.04887)：用 zero-initialized residual gate 直接建立 initial dynamical isometry，可训练数千层 fully connected networks 与 120-layer Transformer。对应“改变 routing / receiver organization 也能重建良好 action-response transport”。

因此本例的 TLIR 定位是：

\[
\boxed{
\text{critical initialization / dynamical isometry}
=
\text{receiver-conditioned transport geometry 的局部良条件化状态}
}
\]

证据级别：**theory-first external theoretical + empirical confirmation**。

真正的反例应证明：在跨层 Jacobian / effective response operator 已严重 ill-conditioned 且不存在 residual/gating/preconditioning 等补偿路径时，深层 action 与 functional response 仍可在任意深度稳定等效运输，且训练速度与该 conditioning 长期系统性无关。

### 36.8 Information Bottleneck：compression 可以是真实局部机制，但不是 Learning / generalization 的普遍必要本体

这一次故意选择一个历史上有强正反争论的理论，避免 theory-first 路线只挑容易命中的对象。

从 TLIR 核心定义出发，Learning 要求的是 persistent functional reorganization；它没有任何核心关系要求 hidden representation 的

\[
I(X;T_\ell)
\]

在 Learning 后半程必须单调下降。因此搜索前冻结的判断是：

\[
\boxed{
\text{representation compression 不是 generalization / Learning 的普遍必要条件。}
}
\]

但如果 actual action、noise、regularization、saturation 或 architecture 选择性消除 task-irrelevant directions，那么 compression 可以成为某些 regime 中真实存在的 local manifestation。由此冻结预测：

1. 一些网络/setting 会出现 fitting 后的 compression；
2. 另一些网络即使 generalize 很好也可以没有该 compression；
3. compression 是否出现应依赖 activation、noise、estimator / measurement definition 与 actual training dynamics；
4. 对 deterministic / near-invertible representation，raw mutual information 可能并不是合适的非平凡状态量；即使如此，task-relevant clustering / support reorganization 仍可发生。

外部独立确认与反确认：

- Shwartz-Ziv & Tishby, *Opening the Black Box of Deep Neural Networks via Information* (2017, arXiv:1703.00810)：提出 information-plane 与 fitting→compression 两阶段叙事，是该理论的经典正结果。
- Saxe et al., *On the Information Bottleneck Theory of Deep Learning* (ICLR 2018)：系统检验上述强版本后发现其三项一般性主张均不在 general case 成立；compression 是否出现高度依赖 nonlinearity，例如 tanh 可出现，而 linear / ReLU setting 并不普遍出现。直接命中“compression 不是 Learning 的必要条件，且取决于 receiver/activation regime”。
- Goldfeld et al., *Estimating Information Flow in Deep Neural Networks* (ICML 2019)：指出 deterministic DNN 中 \(I(X;T_\ell)\) 可成为 vacuous quantity；在 noisy representation 下建立可估计框架，并把观察到的 compression 与类内几何 clustering 联系起来，同时给出 compression 与 generalization 未必因果相关的新证据。对应“measurement projection 与真实 functional reorganization 必须区分”。
- Kawaguchi et al., *How Does Information Bottleneck Help Deep Learning?* (ICML 2023; arXiv:2305.18887)：给出 rigorous learning theory 说明控制 information bottleneck 可以控制 generalization error，但明确结论是它**是一条途径，而不是唯一或必要途径**。这几乎直接命中 TLIR 冻结的 bounded-local-mechanism 定位。

因此本例的 TLIR 定位是：

\[
\boxed{
\text{Information Bottleneck compression}
=
\text{某些 training / noise / representation regimes 下的 functional reorganization projection，}
\text{而不是 Learning 的普遍本体。}
}
\]

证据级别：**theory-first external boundary / negative confirmation + bounded positive theorem confirmation**。

这条很重要，因为它说明 theory-first external confirmation 不是只能“把著名理论证明成对”。TLIR 也可以在检索前预测：某个强理论的哪一部分应成立、哪一部分一旦被升级为普遍必要条件就应失败；外部历史随后确实表现为“强 compression hypothesis → 反例 → measurement clarification → bounded useful mechanism”。

真正会打击 TLIR 的不是“某个网络出现 compression”，而是跨大量 architecture / activation / optimizer / noise regime 严格证明：generalization 与 persistent functional learning **必须且只可能**通过单调 \(I(X;T)\) compression 实现，并且改变 support / boundary / action geometry 在控制 MI 后不再提供任何额外因果信息。

### 36.9 Catastrophic Forgetting / Continual Learning：遗忘是 action–support overlap 与 signed response 吃掉 boundary slack 的结果

这一例先从 TLIR 已建立的 receiving state、actual action、functional response、support 与 boundary 关系出发，不先读取 continual-learning 文献结论。设旧能力 \(q\) 已形成，identity-aligned readout margin 为

\[
m_q(S_t)>0.
\]

新任务在当前 receiver \(S_t\) 上形成实际 training action \(\Delta_t\)。对旧能力的一步局部响应：

\[
\Delta m_q
\approx
\nabla_S m_q(S_t)^\top\Delta_t
+
\frac12\Delta_t^\top H_q(S_t)\Delta_t+
\cdots.
\]

记旧能力局部敏感方向为

\[
g_q(S_t)=\nabla_Sm_q(S_t),
\]

则多步近似可写成

\[
m_q(S_T)
\approx
m_q(S_0)
+
\sum_{t<T}g_q(S_t)^\top\Delta_t
+
\text{receiver-evolution / higher-order terms}.
\]

因此搜索前冻结的上位命题是：

\[
\boxed{
\textbf{Forgetting 不是由 update norm 决定，}
\textbf{而由新 action 在旧功能敏感/support 子空间中的累积 signed response，}
\textbf{相对于旧能力剩余 boundary slack 是否足够大决定。}
}
\]

在最简单的 SGD 局部情形，若新任务 update 为

\[
\Delta\theta=-\eta g_{\rm new},
\]

旧任务 loss gradient 为 \(g_{\rm old}\)，则

\[
\Delta L_{\rm old}
\approx
-\eta g_{\rm old}^\top g_{\rm new}.
\]

所以理论先验地分出三种 regime：

\[
g_{\rm old}^\top g_{\rm new}<0
\Rightarrow
\text{interference / forgetting tendency},
\]

\[
g_{\rm old}^\top g_{\rm new}\approx0
\Rightarrow
\text{local preservation},
\]

\[
g_{\rm old}^\top g_{\rm new}>0
\Rightarrow
\text{positive transfer}.
\]

同时必须把两个概念分开：

\[
\boxed{
\text{support / tangent-space overlap 决定“能不能互相影响”；}
\qquad
\text{signed alignment 决定“影响方向”。}
}
\]

于是冻结预测是：

1. 同样 update norm 可产生完全不同的 forgetting；
2. 新任务 action 若与旧能力功能敏感子空间近似正交，旧能力应大幅保留；
3. overlap 大但 signed alignment 为正时，不只“不忘”，还应出现 positive backward transfer；
4. 普通 parameter-L2 anchoring 不是根本对象，真正该保护的是旧功能高敏感方向；Fisher/EWC 只应是局部近似；
5. rehearsal/replay 的本质应是向 actual action 中加入 counter-action，抵消旧 margin/support 的负漂移；
6. observable accuracy 只有在累积 functional damage 真正吃光 boundary slack 后才下降，因此内部 interference 可早于 discrete forgetting 出现。

外部独立确认：

- Kirkpatrick et al., *Overcoming Catastrophic Forgetting in Neural Networks* (PNAS 2017, DOI 10.1073/pnas.1611835114)：EWC 不均匀地保护参数，而是用旧任务 Fisher information 估计参数重要性，对高重要方向施加更强 penalty。它命中“功能敏感方向应被选择性保护，而不是统一 L2”的冻结预测。
- Lopez-Paz & Ranzato, *Gradient Episodic Memory for Continual Learning* (NeurIPS 2017)：把旧任务梯度约束直接用于避免 loss 上升，同时允许 beneficial backward transfer。它命中 signed alignment 不仅可产生 forgetting，也可产生 positive transfer。
- Farajtabar et al., *Orthogonal Gradient Descent for Continual Learning* (AISTATS 2020)：把新任务 gradient 投影到旧任务输出近似不改变的子空间，直接命中“对旧功能敏感方向正交化可减少 forgetting”。
- Doan et al., *A Theoretical Analysis of Catastrophic Forgetting through the NTK Overlap Matrix* (AISTATS 2021)：理论上证明两个任务越在 NTK / functional subspace 上重叠，catastrophic forgetting 潜势越强，并分析 projected-gradient 方法如何减弱它。对应“overlap 决定 interaction capacity”。
- Rolnick et al., *Experience Replay for Continual Learning* (NeurIPS 2019)：replay 显著减轻 forgetting。TLIR 的上位解释是旧数据重新进入 Training，使 total actual action 中加入保护旧功能的 counter-component，而不是把 replay 仅理解成“保存样本”。

因此本例的 TLIR 定位是：

\[
\boxed{
\text{Catastrophic forgetting / positive transfer / EWC / OGD / GEM / replay}
=
\text{同一 action–receiver–support–boundary dynamics 的不同 interaction regimes 与控制策略。}
}
\]

可将其记录为上位原则：

\[
\boxed{
\textbf{Support-Overlap Interference Law:}
\quad
\text{retention / forgetting / transfer 由 functional overlap、signed action response 与 boundary slack 共同决定。}
}
\]

证据级别：**theory-first external theoretical + empirical confirmation**。

这条外推还与内部 RL-E06 形成直接呼应：support / margin 可以在 mastered capability 仍保持 100% 时持续恶化，直到最后跨 readout boundary 才表现为离散能力丢失。外部 continual-learning 理论与内部 causal feedback experiment 因而落回同一个 TLIR 分层。

真正会打击该外推的是：在严格 matched old functional state / boundary slack 下，大幅改变新 action 与旧 support-sensitive subspace 的 overlap / signed alignment，却长期不改变旧功能响应、margin 或 forgetting 概率；或者证明 forgetting 只由 raw update norm 决定，而 functional alignment 不提供额外信息。

### 36.10 Grokking：表面“突然泛化”可以是连续 support formation 穿过 population boundary

这一例先只使用 TLIR 中 Learning、Capability 与 readout boundary 的分离。设训练集 capability 与 population capability 分别由

\[
C_{\rm train}=B_{\rm train}(S_t),
\qquad
C_{\rm pop}=B_{\rm pop}(S_t)
\]

读出。训练集 boundary 可以先被跨过，而真正支撑 population generalization 的慢 functional coordinate \(z_t\) 仍继续形成。若

\[
z_{t+1}-z_t
\approx
-\gamma(z_t-z^\star),
\qquad
\gamma\ll1,
\]

则 train accuracy 可以很早达到 100%，但 test capability 要等到

\[
z_t>z_c
\]

才跨 population readout boundary。因此搜索前冻结：

\[
\boxed{
\text{Grokking}
=
\text{slow persistent support / representation formation}
+
\text{late population-boundary crossing}.
}
\]

由此得到四个预言：

1. train accuracy 饱和后，内部 representation / support 应继续系统演化；
2. grokking delay 应由 actual-action law（optimizer、weight decay、data schedule 等）控制，而不是“达到 100% train accuracy 后固定等待一个常数时间”；
3. 如果直接促进 generalizable support 的形成，delay 应缩短甚至消失；
4. test accuracy 的“突然跳变”不要求内部动力学也突然，连续内部 formation 可以经离散 readout boundary 表现为 sudden capability emergence。

外部独立确认：

- Power et al., *Grokking: Generalization Beyond Overfitting on Small Algorithmic Datasets* (2022, arXiv:2201.02177)：首先系统报告训练误差早已近零、test accuracy 却在很久以后突然上升的现象，并显示 regularization / data regime 对 delay 有强影响。
- Nanda et al., *Progress Measures for Grokking via Mechanistic Interpretability* (2023, arXiv:2301.05217)：对 modular addition Transformer 做机制级逆向分析，把训练连续分为 memorization、circuit formation、cleanup 三个阶段。表面上的 sudden grokking 背后存在可连续跟踪的 Fourier circuit amplification 和 memorization cleanup，直接命中“continuous hidden formation + discrete observable boundary crossing”。
- 后续 grokking 理论反复把 delay 与 regularization strength、optimizer dynamics、slow mode / representation transition 联系起来；这与 TLIR 的冻结预测一致：delay 是 actual-action / receiver dynamics 的时标，不是 capability plateau 本身产生的固定计时器。

因此本例的 TLIR 定位是：

\[
\boxed{
\text{Grokking}
=
\text{Learning 在 training capability 已形成后继续，}
\text{直到 population-support state 穿过新的 readout boundary。}
}
\]

证据级别：**theory-first external mechanistic + theoretical confirmation**。

真正会打击该外推的是：在高时间分辨率、identity-aligned internal measurement 下，grokking 的 test jump 长期反复发生，却在 jump 以前完全不存在任何持续的 support / representation / margin 形成，而且改变 optimizer / regularization / action law 也不改变 delay 或形成路径。

### 36.11 μP / Maximal Update Parametrization：尺度迁移应保持 functional action-response，而不是保持裸参数统计

这一例直接对应交接说明中“大模型首先测试 scale transport”的判断。设一层

\[
h^{\ell+1}=W^\ell h^\ell.
\]

宽度改变时，一次参数动作对下一层 representation 的一阶作用为

\[
\delta h^{\ell+1}
=
\Delta W^\ell h^\ell
+
W^\ell\delta h^\ell
+
\cdots.
\]

如果 width \(n\to\infty\) 时

\[
\Delta W^\ell h^\ell\to0,
\]

则该层的 feature-changing action 退化，进入近似 lazy / frozen-feature regime；若该项随 width 发散，则 action-response 失控。因此搜索前冻结 TLIR 的 scale-transport 条件：

\[
\boxed{
\Delta W^\ell h^\ell=O(1)
\quad
\text{for every feature-learning layer as width changes.}
}
\]

由此提前推出：

1. input / hidden / output tensor role 不同，应需要不同 width scaling；单一 parameter variance / LR scaling 不一定能同时保持所有层的 functional update；
2. 合理 parameterization 应使 feature change 在无限宽极限仍是非退化 \(O(1)\)，而不是自动掉进纯 NTK / lazy limit；
3. 如果不同 width 下保持的是同一个 normalized action-response regime，那么最优 learning rate 等超参数应可以跨 width transport；
4. 这种 transfer 的根本原因应是 layerwise functional action-response 被保持，而不是“参数数值分布看起来相似”。

外部独立确认：

- Yang & Hu, *Tensor Programs IV: Feature Learning in Infinite-Width Neural Networks* (ICML 2021)：证明 standard / NTK parameterization 的无限宽极限不支持非退化 feature learning，而合适的 maximal-update 类 scaling 可使无限宽 network 仍发生 feature learning。直接命中“feature-changing action 必须在 width→∞ 时保持非退化”的冻结条件。
- Yang et al., *Tensor Programs V: Tuning Large Neural Networks via Zero-Shot Hyperparameter Transfer* (2022, arXiv:2203.03466)：在 μP 下，大量最优超参数随 model width 保持稳定，可以从小模型 zero-shot transfer 到大模型；在 Transformer 与 ResNet 上验证，并展示从 40M proxy 向 6.7B GPT-3 规模迁移的工程结果。对应“保持 normalized action-response regime 后超参数可跨尺度 transport”。

因此本例的 TLIR 定位是：

\[
\boxed{
\mu\text{P}
=
\text{跨 width 保持 role-specific functional update amplitude 的一种 scale-normalized regime。}
}
\]

这使 μP 不再只是一个独立的 parameterization trick，而可以被理解为 TLIR scale transport 的一个精确局部实现：不同 width 的系统并不需要参数逐值相似，只需要在对应 role 上保持等效的 action→response 量级与 feature-learning regime。

证据级别：**theory-first external theorem + engineering confirmation**。

真正会打击这一外推的是：找到一种 parameterization，使跨 width 的 layerwise functional update 全部系统性趋零或发散，但仍能保持同一种非退化 feature-learning dynamics 与稳定超参数 transfer；或者证明 μTransfer 与 functional-update normalization 无关，仅由一个不涉及 receiver/action 的独立常数控制。

### 36.12 Sharpness / Flat Minima / SAM：raw parameter sharpness 不是 functional ontology，sharpness-aware action 可以是真实局部控制

从 TLIR 出发，局部 parameter perturbation \(\delta\) 对当前 receiver 的 loss response 可写成

\[
\Delta L
\approx
 g^\top\delta
+
\frac12\delta^\top H\delta.
\]

但如果两个 parameterizations \(\theta\) 与 \(\phi(\theta)\) 实现完全相同的 input-output function，则 raw Hessian eigenvalue、parameter-space distance 和某些 sharpness 指标可以因坐标变换而改变，而 functional capability 完全不变。因此搜索前冻结：

\[
\boxed{
\text{raw parameter-space sharpness 不是完整 functional state，}
\text{也不应成为 reparameterization-invariant 的普遍 generalization law。}
}
\]

但同时，如果 training action 明确优化某个 perturbation family \(\mathcal D\) 下的最坏响应：

\[
L_{\rm robust}(\theta)
=
\max_{\delta\in\mathcal D}L(\theta+\delta),
\]

那么它确实改变 actual-action geometry，并可把系统推向对该 perturbation set 更稳健的 receiver。因此第二条冻结命题是：

\[
\boxed{
\text{flatness / sharpness 可以是有用的 robustness projection，}
\text{但不是 complete neural dynamics 的本体。}
}
\]

由此预言：

1. 应存在 function / generalization 几乎不变、raw sharpness 却大幅变化的重参数化反例；
2. sharpness 与 generalization 的关系应依赖 parameterization、norm 与 perturbation definition；
3. SAM 可以有效，是因为它改变了 actual training action / local robustness objective，而不是因为“最 flat minimum”是普适真理；
4. SAM 不应普遍最优，因为它只优化指定 local perturbation family，而不是 whole-system functional dynamics。

外部独立确认：

- Dinh et al., *Sharp Minima Can Generalize For Deep Nets* (ICML 2017)：利用 ReLU network 的 parameter symmetries 构造 functionally equivalent 但可任意更 sharp 的 minima；parameter geometry 可剧烈变化而 generalization 不变。直接命中 frozen negative prediction。
- Foret et al., *Sharpness-Aware Minimization for Efficiently Improving Generalization* (ICLR 2021)：通过 min-max neighborhood objective 同时压低 loss value 与 neighborhood sharpness，在 CIFAR、ImageNet、finetuning 等多类任务上改善 generalization。它命中“sharpness-aware action 可以是真实有效的局部控制方法”。
- 后续工作继续显示 raw flatness、parameterization 与 generalization 并非一一对应；这与 TLIR 的定位一致：有用的 local geometry 不自动升级成 complete functional cause。

因此本例的 TLIR 定位是：

\[
\boxed{
\text{Sharpness}
=
\text{receiver 的一种局部 parameter-space projection};
\qquad
\text{SAM}
=
\text{改变 actual-action geometry 的 robustness control method。}
}
\]

证据级别：**theory-first external positive + falsification-boundary confirmation**。

真正会打击该外推的是：证明某个 raw parameter-space sharpness scalar 在所有 function-preserving reparameterization 下严格 invariant，并且跨 matched receiver/action/history 后单独充分决定完整 generalization / finite response；或者 SAM 的收益在控制 actual-action difference 后仍完全由“flat minimum 标签”独立决定。

### 36.13 Gradient Noise Scale / Critical Batch Size：critical batch 是 current receiver 下的 functional signal-to-action-noise transition

令单样本梯度为 \(g_i(S_t)\)，batch-size \(B\) 的估计梯度为

\[
\hat g_B
=
 g+
\xi_B,
\qquad
\mathbb E[\xi_B]=0,
\qquad
\operatorname{Cov}(\xi_B)=\frac{\Sigma_g(S_t)}{B}.
\]

training action 真正造成的 functional response 在局部近似下是

\[
\Delta f
\approx
-\eta J_t\hat g_B.
\]

因此 functional-action noise covariance 为

\[
\operatorname{Cov}(\Delta f)
\approx
\frac{\eta^2}{B}
J_t\Sigma_g(S_t)J_t^\top.
\]

所以搜索前冻结的 TLIR 命题是：

\[
\boxed{
\textbf{Critical batch size 不是固定数据常数，}
\textbf{而是 current receiver 下 functional signal 与 action-noise 的相对尺度。}
}
\]

由此得到：

1. 小 \(B\) 区域增大 batch 会近似按 \(1/B\) 降低 gradient/action noise，因此并行收益接近线性；
2. 当 noise 已低于有效 signal / curvature / response scale，再增大 \(B\) 收益应迅速递减，形成 critical batch；
3. \(J_t\)、mean gradient 与 gradient covariance 都随 Learning 演化，因此 critical batch 应随 training state / loss phase 变化；
4. optimizer、preconditioner、LR schedule 改变 action geometry 后，raw gradient-noise scale 不应成为跨 optimizer 普适常数；
5. “large batch generalization gap”不应由 batch-size label 单独决定；若 LR、warmup、training duration、normalization 等改变后有效 actual action 被重新匹配，gap 可以大幅变化。

外部独立确认：

- McCandlish et al., *An Empirical Model of Large-Batch Training* (2018, arXiv:1812.06162)：提出 gradient noise scale，并跨 supervised learning、RL、generative models 显示它可预测 largest useful batch size；同时发现 noise scale 会随 loss / training process 增大或变化。直接命中“critical batch 随 receiver/training state 演化”。
- Shallue et al., *Measuring the Effects of Data Parallelism on Neural Network Training* (JMLR 2019)：系统测量 batch-size scaling 的 time/data-efficiency trade-off，显示不同 task / optimizer 的 useful-batch regime 并不统一。
- Hoffer et al., *Train Longer, Generalize Better: Closing the Generalization Gap in Large Batch Training of Neural Networks* (NeurIPS 2017)：改变 training duration / optimization regime 后可以显著缩小 large-batch generalization gap，说明 batch-size label 不是 complete training action。

因此本例的 TLIR 定位是：

\[
\boxed{
\text{Critical Batch Size}
=
\text{current receiver 下 functional signal-to-action-noise transition 的局部工程投影。}
}
\]

证据级别：**theory-first external empirical + theoretical convergence**。

真正会打击该外推的是：在训练过程中 \(J_t\)、gradient covariance、optimizer state 明显改变时，critical batch 却长期保持同一与 receiver 无关的精确常数；或者在 matched functional action noise 条件下，仅改变 batch-size 标签仍稳定改变 finite functional response / generalization。

### 36.14 Double Descent：interpolation singularity 与 post-threshold support redistribution 的两阶段投影

这一例把历史矩阵中的 Double Descent 从“可统一解释”升级为真正 theory-first 推导。对过参数线性 regression，noise fitting component 为

\[
\hat\beta_{\rm noise}=X^+\varepsilon.
\]

其 population impact 包含

\[
R_{\rm noise}
\propto
\operatorname{tr}(\Sigma X^+X^{+\top}).
\]

若

\[
X=U\operatorname{diag}(s_i)V^\top,
\]

则 noise amplification 中出现

\[
\sum_i\frac{w_i}{s_i^2}.
\]

因此当 model capacity 接近“刚好能够 interpolation”的边界时，若最小 singular directions \(s_i\to0\)，为了把 training residual 压到零，training action 会在 near-null directions 上产生巨大 response amplification，导致 test risk peak。继续增加冗余后，如果 actual-action / implicit bias 选择 min-norm 或大量 low-population-impact directions，noise-support 可重新分散，test risk 再次下降。

所以搜索前冻结：

\[
\boxed{
\text{Double Descent}
=
\text{approximation improvement}
\rightarrow
\text{interpolation-boundary response amplification}
\rightarrow
\text{redundant low-impact support redistribution}.
}
\]

由此预言：

1. peak 应靠近 interpolation / ill-conditioning boundary，而不是固定 parameter count；
2. ridge、early stopping、PCR 等限制 near-null response 的机制应压低甚至抹平峰；
3. data noise、sample size、feature spectrum 改变时，峰的位置和高度应移动；
4. 继续过参数化后是否第二次下降，取决于 actual-action 的 implicit solution selection，不是“参数越多必然变好”；
5. epoch-wise double descent 应同样可能发生：width 不变，但随着 training trajectory 的 effective fitted subspace 穿过 interpolation boundary，也会出现风险峰。

外部独立确认：

- Belkin et al., *Reconciling Modern Machine-Learning Practice and the Classical Bias–Variance Trade-Off* (PNAS 2019)：系统提出 double-descent risk curve，并把 peak 与 interpolation threshold 联系起来。
- Hastie et al., *Surprises in High-Dimensional Ridgeless Least Squares Interpolation* (Annals of Statistics 2022)：在 high-dimensional ridgeless least squares 中严格分析 interpolation 附近的风险爆涨与过参数化后的第二次下降，直接对应 singular-direction noise amplification 与 post-threshold solution geometry。
- Nakkiran et al., *Deep Double Descent: Where Bigger Models and More Data Hurt* (ICLR 2020)：在现代深网中观察 model-wise、sample-wise、epoch-wise double descent，并提出 effective model complexity；尤其 epoch-wise double descent 直接命中“不是 raw parameter count，而是 trajectory/effective interpolation boundary”的冻结预测。
- 多类 ridge / early stopping / spectral regularization 结果显示，限制 near-null / high-variance response 可以明显压低 interpolation peak，进一步支持该机制边界。

因此本例的 TLIR 定位是：

\[
\boxed{
\text{Double Descent}
=
\text{receiver/function space 靠近 interpolation singularity 时的 response amplification}
+
\text{越过阈值后的 action-selected support redistribution。}
}
\]

证据级别：**theory-first external theorem/theory + empirical confirmation**。

真正会打击该外推的是：改变 interpolation threshold、noise spectrum、regularization 与 implicit solution selection 后，double-descent peak 的位置/高度仍完全不动；或者在不存在任何 interpolation ill-conditioning / near-null amplification 的 regime 中仍稳定出现同构 peak，且无法由其他 TLIR boundary / support transition 解释。

### 36.15 Theory-first 外推的归纳层：反复出现的不是案例，而是同一组 TLIR 上位不变量

截至当前，已完成或已纳入交接说明的 theory-first external confirmation 至少覆盖：

\[
\boxed{
\begin{aligned}
&\text{Edge of Stability}\\
&+\text{Post-capability implicit bias / max-margin}\\
&+\text{Recursive feedback contraction}\\
&+\text{Spectral Bias}\\
&+\text{Benign Overfitting}\\
&+\text{Scaling Laws / Chinchilla}\\
&+\text{Dynamical Isometry}\\
&+\text{Information Bottleneck}\\
&+\text{Catastrophic Forgetting / Continual Learning}\\
&+\text{Grokking}\\
&+\mu\text{P / maximal-update parameterization}\\
&+\text{Sharpness / SAM}\\
&+\text{Gradient Noise Scale / Critical Batch Size}\\
&+\text{Double Descent}.
\end{aligned}
}
\]

这些方向跨 optimization stability、implicit bias、generalization、frequency learning、representation、width scaling、batch scaling、continual learning、feedback dynamics、interpolation 与 information theory。它们并非都“证明 TLIR 是对的”的同一种证据；真正值得记录的是：**不同领域的局部理论在 theory-first 推导中反复落回同一小组上位关系。**

#### I1. Receiver dependence：作用的效果不是 action 自身固有属性

反复出现于：Edge of Stability、Spectral Bias、Dynamical Isometry、Critical Batch、Continual Learning、Scaling Laws。

\[
\boxed{
R_t=R(S_t,A_t),
\qquad
R(A_t)\text{ alone generally insufficient.}
}
\]

同一个 update、gradient noise、perturbation、frequency mode 或 feedback，在不同 receiving state 中可以产生不同 finite functional effect。很多“固定常数”之所以失效，本质上是把 receiver evolution 从动力学里删掉了。

#### I2. Actual-action geometry：名义 hyperparameter / optimizer label 不是 training action 本体

反复出现于：implicit bias、μP、SAM、large-batch scaling、continual learning、Chinchilla allocation。

真正进入系统的是形成后的 action，包括 optimizer memory、preconditioning、loss weighting、batch composition、feedback binding、parameterization 等，而不是标签“SGD / Adam / batch=4096 / model=7B”本身。

\[
\boxed{
\text{same objective / same accuracy / same nominal hyperparameter}
\not\Rightarrow
\text{same actual action}.}
\]

#### I3. Functional response / support formation：Learning 的核心对象是持久功能组织，不是单一 observable

反复出现于：Grokking、Neural spectral learning、Benign Overfitting、Continual Learning、Recursive Feedback、Information Bottleneck。

\[
\boxed{
\text{Learning}
=
\text{persistent functional reorganization},
\quad
\text{not merely loss decrease / parameter movement / accuracy change}.}
\]

这也是为什么 capability plateau 后仍可继续 implicit-bias organization、Neural Collapse 候选方向、support concentration、grokking circuit formation 或 forgetting margin erosion。

#### I4. Boundary crossing：observable capability 可以离散，内部 formation 不必离散

反复出现于：post-capability learning、Grokking、Double Descent 的 interpolation boundary、RL-E06、adversarial / robustness trade-off。

\[
\boxed{
\text{continuous internal formation}
+
\text{discrete readout criterion}
\Rightarrow
\text{apparently sudden capability transition}.}
\]

因此 accuracy / benchmark pass-fail 不能被当成 Learning state 本身。

#### I5. Overlap / interaction structure：shared system 中的能力不是彼此独立的标量

反复出现于：Continual Learning、distributed support、feedback concentration、replay、feature learning。

一个 action 是否伤害、保护或帮助另一能力，取决于它们在 functional support / tangent / receiver geometry 上的 overlap 与 signed interaction，而不是“两个任务不同”这一语义标签。

\[
\boxed{
\text{overlap determines interaction capacity; signed response determines direction.}
}
\]

#### I6. Local projection \(\neq\) complete dynamics：局部理论可以严格成立，但不能自动篡位成完整系统

反复出现于：NTK、Edge of Stability、Sharpness、Information Bottleneck、Dynamical Isometry、Scaling Laws、Spectral Bias、Fisher/EWC。

\[
\boxed{
\text{exact local theorem / projection / coordinate}
\not\Rightarrow
\text{complete neural ontology}.}
\]

TLIR 的统一方式不是否定这些理论，而是给出它们的 regime、作用对象、边界与超出边界后的失败原因。

#### I7. Scale transport 应保持 role-level functional dynamics，而不是复制表面尺度

由 μP、Scaling Laws、cross-system TL-G01/G02 共同加强：

\[
\boxed{
\text{真正需要 transport 的是 receiver/action/response/support/boundary role，}
\text{不是 parameter count、raw gradient norm 或某个固定 feature 的数值。}
}
\]

这解释为什么 Transformer→ResNet→Diffusion 的 role-level transport，在理论普遍性上比单纯 0.6B→7B 更有信息；也解释为什么大模型实验主要测试 scale transport / metrology / engineering value。

#### 归纳结论

因此现在最值得检验的高层命题已经不是“TLIR 能否解释第 15 个现象”，而是：

\[
\boxed{
\textbf{大量看似分散的 ML/RL 局部规律，}
\textbf{是否正是同一组 receiver–action–response–support–boundary–feedback 不变量}
\textbf{在不同 regime / projection / scale 下的 manifestations。}
}
\]

如果后续新的著名理论仍在**先冻结推导、允许失败**的条件下反复落回这些关系，这种 repeated derivational convergence 就不再只是“案例数量增加”，而是在逐渐识别统一理论真正的 invariant structure。

### 36.16 Theory-first evidence 的程序性边界与后续扩展纪律

需要诚实保留程序性边界：36.4 以后这一批新增案例是在会话内按“先写 TLIR 推导 / 冻结预测，再联网检索外部文献”的顺序执行；这已经明显强于 ordinary post-hoc compatibility，但并不是密码学意义上的 blind preregistration，因为执行推导的模型预训练知识可能包含部分经典文献。

因此若把这些案例升级成论文正式 theory-first evidence，最佳做法是：

1. 将下一条 frozen derivation、assumptions、equations、predicted boundary、falsifier **先写入 Git / timestamped artifact**；
2. 由隔离检索流程、不同模型窗口或独立研究者再做 external literature adjudication；
3. 外部结果只允许标记为 `proved / observed / partial / unresolved / contradicted`；
4. 变量、方向或 boundary 不同，仅能叫 analogy / compatibility，不能升级成 confirmation；
5. 若外部结果与推导冲突，保留冲突，区分是局部外推错、regime 假设错，还是 TLIR core relation 真正受到挑战；
6. 不以“已经能解释很多理论”为理由停止寻找 hard falsifier。

下一批值得做的对象包括：

- Neural Collapse：从 post-capability residual action + balanced symmetry 推 simplex-like boundary organization，并冻结 class-imbalance 破坏条件；
- Lottery Ticket / weight rewinding：检验 initialization 重要但不 sufficient，early receiver formation 是否解释 rewinding 优势；
- simplicity bias / spectral complexity；
- feature-learning phase transition；
- in-context learning / induction-head formation；
- adversarial robustness / robust feature formation 的更严格 theory-first 推导；
- Neural Collapse、Lottery Ticket 等若完成后，应继续更新 36.15 的 invariant map，而不是只继续堆编号。

这条证据路线的核心意义保持不变：

\[
\boxed{
\text{TLIR 不只能够整理既有历史；}
\text{它能够从已经由真实系统实验建立的统一关系，}
\text{重新导出由其他研究路线独立发现的局部 law / theorem / boundary。}
}
\]

当这种 theory-first external confirmation 在彼此独立的领域反复出现时，它构成独立于作者自身 experimental programme 的强外部证据轴；但其最高证据形态仍应通过真正 timestamped / blinded prediction 与允许 falsification 的外部 adjudication 来继续提高。

---
