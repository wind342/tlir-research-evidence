## 5.2 理论先行外推与独立研究对照

**方程来源约定（统一按 [TLIR_UNIFIED_DYNAMICS_V5.md](TLIR_UNIFIED_DYNAMICS_V5.md) 的编号）。** [T] 表示在本节明确写出的TLIR原方程；[A] 为被外推原研究已经使用的模型与数学条件；[R] 为在代入与计算之后恢复出的具体数学关系。公式中的等号、条件期望、近似号与渐近号不得互换。TLIR的统一动力学方程确定变量之间的关系；具体闭式系数、谱律、损失幂指数等来自被外推研究的原模型条件。每项都必须有实际的代入/消元步骤，不能只做概念映射。
理论先行外推从已建立的TLIR关系出发，使用被外推研究**原本采用的模型与数学条件**，推出相应的功能变化、稳定边界、风险或优化结果，不另立用于凑结论的假设。

**已完成核心数学关系推导（14项）：** 5.2.1–5.2.14。每项列出TLIR入口、被研究理论原有条件、计算步骤、所得核心关系与研究对照。其中5.2.2的多维最大间隔方向沿用原研究的定理，5.2.6由原Chinchilla经验损失律推出最优配置；5.2.10在ICLR 2024可解的线性Grokking模型下推出迟发时间；5.2.11推导μP原MLP示例的核心响应量级而非全部参数族定理。5.2.8已将原Information Bottleneck统一解释替换为Dropout自适应正则化，直接由原GLM与独立Bernoulli屏蔽条件得到相应的精确正则项和功能更新关系。

**判别标准：** 将“由数学条件直接得到的公式”与“将外部现象置于TLIR链条中的统一解释”分开；只有前者列为完成推导。

### 5.2.1 演化稳定性：Edge of Stability

**推导类型：原理论条件下的局部稳定阈值。**

**逐式来源标记。** **[T]** 原方程：\(\theta_{t+1}=\theta_t+\delta_t\)，\(Z_{t+1}=\mathcal T(Z_t,A_t)\)。**[A]** 原研究的局部二次模型与梯度下降：\(\delta_t=-\eta He_t\)，\(H=H^\top\)、\(e_t=\theta_t-\theta^\star\)。代入得到 \(e_{t+1}=(I-\eta H)e_t\)，再投影到 \(H\) 的每个特征方向。**[R]** 仅在正曲率方向的局部线性稳定边界：\(\lvert1-\eta\lambda_i\rvert<1\)，最高正曲率方向临界 \(\eta\lambda_{\max}=2\)。这里的 \([R]\) 是**局部边缘条件**，不是任意非凸训练全程都满足的稳定等式。

**TLIR作用对象。** 实际更新形成与状态转移：\(\theta_{t+1}=\theta_t+\delta_t\)，一次功能响应为\(R_q(\theta_t,\delta_t)=m_q(\theta_t+\delta_t)-m_q(\theta_t)\)。

**使用的原理论条件。** 普通梯度下降、固定学习率\(\eta>0\)，在驻点附近以局部二次损失近似，Hessian \(H=H^\top\) 在局部固定。这里分析的正是Edge of Stability文献的冻结局部线性稳定性问题。

**逐步推导。**

\[
\delta_t=-\eta\nabla L(\theta_t),\qquad
e_t=\theta_t-\theta^\star,\qquad
\nabla L(\theta_t)\approx He_t.
\]

将实际动作代入TLIR状态转移，得

\[
e_{t+1}=(I-\eta H)e_t.
\]

在\(H v_i=\lambda_i v_i\)的特征方向上，

\[
e_{i,t+1}=(1-\eta\lambda_i)e_{i,t}.
\]

正曲率方向局部收缩，当且仅当

\[
|1-\eta\lambda_i|<1
\iff 0<\eta\lambda_i<2.
\]

因此最高正曲率方向的局部边缘条件为

\[
\boxed{\lambda_{\max}(H)=2/\eta.}
\]

**推得的结论。** \(2/\eta\)并不是额外赋予网络的“能力门槛”，而是所用梯度下降映射的局部稳定边界。实际训练改变接收状态，曲率也随之变化；冻结\(H\)的上述推导确定瞬时边界，不单独计算越界后的长期曲率演化。

**研究对照。** Cohen等的Edge of Stability实验；Lewkowycz等的catapult轨迹。

**内部锚点。** TL-E03（实际动作的接收状态依赖）；TL-E04（有限更新响应）。

### 5.2.2 能力形成后的持续学习：Implicit Bias与最大间隔

**推导类型：直接证明“读出已正确而状态仍变化”；最大间隔方向采用该理论既有定理（不在此重证多维定理）。**

**逐式来源标记。** **[T]** 原方程：\(\theta_{t+1}=\theta_t+\delta_t\)、\(R_{t,q}=m_q(L_{t+1})-m_q(L_t)\)、\(C_{t,q}=\chi_q(m_q(L_t))\)。**[A]** 原论文线性可分 logistic 回归与梯度下降；在一维实例中 \(m(w)=w\)，\(\delta_t=\eta/(1+e^{w_t})\)。因此一旦 \(w_t>0\)，\(\chi(m)=1\) 持续不变，而 \(\delta_t>0\)；由 \(e^{w_{t+1}}-e^{w_t}\to\eta\) 得 \(w_t=\log t+O(1)\)。**[R]** 已正确的离散读出与持续状态更新可以同时成立；多维向最大间隔方向收敛则是 **Soudry 等的原有定理**，本一维计算并未单独重证多维定理。

**TLIR作用对象。** \(C_{t,q}=\mathbf1[m_q(\theta_t)>0]\)以及\(\theta_{t+1}=\theta_t+\delta_t\)。正确性是目标读出，不等于参数状态停止变化。

**使用的原理论条件。** 无显式正则化、线性可分样本、线性分类器、logistic损失、梯度下降。这是Soudry等最大间隔研究的原有设定。

**逐步推导。** 对样本\((x_i,y_i)\)（\(y_i\in\{-1,1\}\)）令\(m_i=y_iw^\top x_i\)，则

\[
L(w)=\sum_i\log(1+e^{-m_i}),\qquad
-\nabla L(w)=\sum_i\frac{y_i x_i}{1+e^{m_i}}.
\]

即使所有\(m_i>0\)、训练正确率已经为\(100\%\)，右侧一般仍不为零：实际更新仍会移动\(w\)及各个间隔。

为把这一点写成完全可解的例子，取同一原模型类中的一维可分样本\(x=y=1\)，采用梯度下降：

\[
w_{t+1}=w_t+\frac{\eta}{1+e^{w_t}}.
\]

\(w_t\)单调增长并趋于无穷；一旦\(w_t>0\)，任务的离散正确性始终为1，但更新仍严格为正。由

\[
e^{w_{t+1}}-e^{w_t}
=e^{w_t}\!\left(e^{\eta/(1+e^{w_t})}-1\right)
\longrightarrow\eta
\]

得到渐近关系

\[
\boxed{w_t=\log t+O(1),\qquad C_t=1\ \text{而}\ \Delta w_t>0.}
\]

在多维线性可分情形，Soudry等已证明在其原有步长与损失条件下，梯度下降的**方向**趋向硬间隔SVM方向。该多维定理是被对照理论已有的数学结果，不把它冒称为仅由TLIR定义直接证明的全新定理。

**推得的结论。** “外部正确性已饱和”与“内部学习状态持续演化”可以严格同时成立；优化轨迹还可选择特定的间隔方向。

**研究对照。** [Soudry等，JMLR 2018](https://jmlr.org/papers/v19/18-188.html)。

**内部锚点。** TL-E07（正确性不变时支撑仍可重组）；TL-E08（正确性边界）；RL-E06（已掌握后仍存在功能变化）。

### 5.2.3 递归反馈的信息收缩与模型坍塌：有限样本吸收态

**推导类型：在原模型坍塌论文的有限离散分布、逐代抽样、精确拟合设定下，直接推导尾部丢失概率和最终吸收态；对真实数据混合，推导原混合数据设定下的期望关系。**

**逐式来源标记。** **[T]** 原反馈闭环 \(Y_t=\mathcal I(L_t,x_t,c_t)\)、\(B_{t+1}=\mathcal B(B_t,Y_t,E_t)\)、\(A_{t+1}=\mathcal U(Z_{t+1},D_{t+1},B_{t+1})\)、\(Z_{t+2}=\mathcal T(Z_{t+1},A_{t+1})\)。这里把一次模型代际递归视为一次闭环转移，而非一个优化步。**[A]** 原文有限离散支持、固定 \(M\) 次独立采样、每代精确拟合经验频率：\(N_{t+1,i}\mid p_t\sim\mathrm{Bin}(M,q_t)\)，\(q_{t+1}=N_{t+1,i}/M\)。**[R]** \(\Pr(q_{t+1}=0\mid q_t)=(1-q_t)^M\)、\(\mathbb E(q_{t+1}\mid q_t)=q_t\)；有限状态吸收链最终在某一类别的点质量处吸收。添加固定比例真实数据后，缺失类别不再是吸收态，恢复概率由原混合抽样条件计算。**切勿把随机支持消失改写成每代期望概率必然下降。**

**TLIR作用对象。** TLIR式(4)描述冻结推理，式(5)描述推理后果回流反馈及后续训练数据\(D_{t+1}\)，进而由训练动作改变持久模型状态；式(1)的状态转移将此写为\(Z_{t+1}=\mathcal T(Z_t,\mathcal U(Z_t,D_t,B_t))\)。这里的\(t\)按**模型代际**计，而非一次优化器步。具体的生成、抽样与拟合映射使用被研究论文自己的定义。

**使用的原理论条件。** Shumailov等（Nature 2024）的「有限离散分布、无函数逼近误差」模型：有限类别集合\(\{1,\ldots,K\}\)、每代样本数固定为\(M\ge 1\)、从上一代模型分布\(p_t\)独立抽样\(M\)次、下一代模型精确拟合这批样本的经验分布。不额外设定信息保留率、噪声衰减指数或模型容量关系。

**逐步推导：一次代际反馈怎样丢失信息。** 对任一类别\(i\)，令其概率为\(q_t=p_t(i)\)，下一代计数\(N_{t+1,i}\)满足

\[
N_{t+1,i}\mid p_t\sim \operatorname{Binomial}(M,q_t),\qquad
q_{t+1}=\frac{N_{t+1,i}}{M}.
\]

因此

\[
\boxed{
\begin{aligned}
\Pr(q_{t+1}=0\mid q_t)&=(1-q_t)^M,\\
\mathbb E[q_{t+1}\mid q_t]&=q_t,\\
\operatorname{Var}(q_{t+1}\mid q_t)&=\frac{q_t(1-q_t)}{M}.
\end{aligned}}
\]

对于\(q_t\ll1/M\)，漏抽该类别的概率很高。**特别注意，单代期望概率并不下降**：这里的坍塌是有限抽样导致的随机支持集丢失，不是事先假定\(q_{t+1}=\rho q_t\)且\(\rho<1\)。

**逐步推导：多代递归为什么到达吸收态。** 对整个分布，

\[
p_{t+1}=\frac1M\,\operatorname{Multinomial}(M,p_t).
\]

一旦\(p_t(i)=0\)，纯递归抽样无法再产生类别\(i\)；一旦\(p_t=e_i\)（全部概率集中于一个类别），则\(p_{t+1}=e_i\)必然成立。因为\(M\)固定，经验分布只在有限个格点上取值；在任一非吸收态，下一代所有样本恰好相同的概率为\(\sum_i p_t(i)^M>0\)，因此以概率1最终进入某个\(e_i\)。再由各坐标的有界鞅性质，

\[
\boxed{
p_t\xrightarrow[t\to\infty]{\mathrm{a.s.}}e_I,\qquad
\Pr(I=i)=p_0(i).
}
\]

**核心结论：** 原有限离散模型中，纯生成数据逐代替换使支持集随机收缩，并最终坍塌到单一类别；该结果由原抽样—拟合映射推出，无需设定统一的逐代衰减率。

**原研究中的另一种数据组织：持续混合真实数据。** 对采用固定比例\(\alpha\in(0,1]\)真实源数据、其余比例使用上一代生成数据的原有混合设定，设真实分布在类别\(i\)的概率为\(r_i>0\)。混合抽样概率是\(\widetilde q_t=(1-\alpha)q_t+\alpha r_i\)，因此

\[
\boxed{
\mathbb E[q_{t+1}]=(1-\alpha)\mathbb E[q_t]+\alpha r_i
=r_i+(1-\alpha)^{t+1}(q_0-r_i)
}
\]

（最后一式取\(t\)从0开始）。并且在\(q_t=0\)时，下一代重新出现该类别的概率为

\[
\Pr(q_{t+1}>0\mid q_t=0)=1-(1-\alpha r_i)^M>0.
\]

因此真实数据使“缺失类别”不再是吸收态；结论是**长期期望与可恢复性**，而不是每代有限样本永远不丢失类别。另有Gerstgrasser等采用不同的「保留并累积历史数据」训练方案并证明相应线性模型的误差有界；两种数据方案不可混作同一个递推。

**独立研究对照。** [Shumailov等，AI models collapse when trained on recursively generated data，Nature 2024](https://doi.org/10.1038/s41586-024-07566-y)；[Seddik等，How Bad is Training on Synthetic Data? (2024)](https://arxiv.org/abs/2404.05090)；[Gerstgrasser等，Is Model Collapse Inevitable? (2024)](https://arxiv.org/abs/2404.01413)。

**内部实验锚点。** TLIR式(5)（输出回流训练）、式(1)（状态转移）；RL-E06（反馈数据组织改变持续功能结果）。

### 5.2.4 频率学习顺序：Spectral Bias与响应谱

**推导类型：原理论的核动力学条件下，直接推导模态收敛速率。**

**逐式来源标记。** **[T]** 原状态转移 \(\theta_{t+1}=\theta_t+\delta_t\)，功能响应 \(R_{t,q}=m_q(\theta_{t+1})-m_q(\theta_t)\)。**[A]** 平方损失、GD、原研究采用的局部线性化 \(J=\partial f/\partial\theta\) 和在相应区间固定的核 \(K=JJ^\top\)。代入 \(\delta_t=-\eta J^\top(f_t-y)\) 得 \(\Delta f_t\approx-\eta K(f_t-y)\)；沿 \(Kv_k=\lambda_kv_k\) 的模态重复迭代。**[R]** \(r_{k,t}\approx(1-\eta\lambda_k)^t r_{k,0}\)；只有在**原模型的傅里叶核谱**满足低频特征值较大时才得到低频先学，不是TLIR额外宣布的通用频率排序。

**TLIR作用对象。** 实际更新导致目标函数响应\(R_q(\theta,\delta)\)。对输出向量\(f_\theta\)作当前状态的一阶展开，可将参数动作映射到功能空间。

**使用的原理论条件。** 平方损失、梯度下降，以及核谱研究采用的局部线性化/固定切线核条件。记目标向量为\(y\)，残差\(r_t=f_{\theta_t}-y\)，输出参数雅可比\(J_t\)。

**逐步推导。**

\[
L=\tfrac12\|r_t\|^2,\quad
\delta_t=-\eta J_t^\top r_t,\quad
\Delta f_t\approx J_t\delta_t=-\eta J_tJ_t^\top r_t.
\]

定义\(K_t=J_tJ_t^\top\)。在冻结\(K_t=K\)的局部演化中，

\[
r_{t+1}\approx(I-\eta K)r_t.
\]

对\(K v_k=\lambda_kv_k\)作特征分解：

\[
\boxed{r_{k,t}\approx(1-\eta\lambda_k)^t r_{k,0}.}
\]

当\(0<\eta\lambda_k<1\)时，\(\lambda_k\)越大，该模态的残差衰减越快。原Spectral Bias研究中，若相应输入分布和核的**傅里叶模态**满足低频具有较大的\(\lambda_k\)，便得到低频优先的顺序。

**推得的结论。** TLIR的动作—响应关系在固定核区域分解为明确的模态形成速率；所谓“低频先学”来自原研究的核谱结构，而不是向TLIR另加一个普适的频率排序假设。

**研究对照。** Basri等、Cao等的核谱与频率收敛研究；Tancik等的Fourier Features研究。

**内部锚点。** TL-E03、TL-E04、TL-E07。

### 5.2.5 Benign Overfitting：插值解选择与总体风险

**推导类型：使用原过参数线性回归设定，直接得到最小范数解的总体风险分解。**

**逐式来源标记。** **[T]** 原训练转移 \(\theta_{t+1}=\theta_t+\delta_t\)；可将测试风险看作固定总体分布上的连续功能读出 \(m_{\rm test}(\theta)\)，于是 \(R_{t,\rm test}=m_{\rm test}(\theta_{t+1})-m_{\rm test}(\theta_t)\)。**[A]** 原最小范数线性回归：\(y=X\beta^\star+\varepsilon\)，零初始化平方损失梯度下降、\(p>n\)、\(X\) 满行秩、测试协方差 \(\Sigma\)。由原训练动作选取 \(\hat\beta=X^\top(XX^\top)^{-1}y\)，代入总体风险并对零均值独立噪声取条件期望。**[R]** 风险分解为 \(\beta^{\star\top}(I-P_X)\Sigma(I-P_X)\beta^\star+\sigma^2\operatorname{tr}[\Sigma X^\top(XX^\top)^{-2}X]\)。这是**条件期望等式**，不是单个测试样本的必然值。

**TLIR作用对象。** 训练动作选择最终参数状态；训练集和总体分布的功能表现是同一状态上的**不同读出**，不能以训练误差代替总体风险。

**使用的原理论条件。** \(y=X\beta^\star+\varepsilon\)、\(p>n\)、\(X\in\mathbb R^{n\times p}\)满行秩、平方损失、从零初始化的梯度下降收敛到最小范数插值解；测试输入协方差为\(\Sigma\)，训练噪声独立零均值且协方差\(\sigma^2I_n\)。这些均属于原良性过拟合/无正则最小二乘研究中的条件。

**逐步推导。** 梯度始终属于\(X^\top\)的列空间，零初始化的极限插值解为

\[
\hat\beta=X^\top(XX^\top)^{-1}y.
\]

令\(P_X=X^\top(XX^\top)^{-1}X\)，代入\(y\)得

\[
\hat\beta-\beta^\star
=-(I-P_X)\beta^\star+X^\top(XX^\top)^{-1}\varepsilon.
\]

总体超额平方预测风险为
\(R=(\hat\beta-\beta^\star)^\top\Sigma(\hat\beta-\beta^\star)\)。
对训练噪声取条件期望，交叉项消失：

\[
\boxed{
\begin{aligned}
\mathbb E_\varepsilon[R\mid X]
={}&\beta^{\star\top}(I-P_X)\Sigma(I-P_X)\beta^\star\\
&+\sigma^2\operatorname{tr}\!\left[
\Sigma X^\top(XX^\top)^{-2}X
\right].
\end{aligned}}
\]

**推得的结论。** 所有训练样本均可插值（训练误差为零），但总体风险由未覆盖信号和被放大的噪声方向共同决定。原论文关于有效秩/协方差谱的条件，决定第二项是否足够小，进而产生“良性”过拟合。

**研究对照。** [Hastie等，Annals of Statistics 2022](https://doi.org/10.1214/21-AOS2133)；Bartlett等的有效秩分析。

**内部锚点。** TL-E03（不同接收状态与实际更新）；TL-E08（目标依赖的读出）。

### 5.2.6 Scaling Laws与Chinchilla：给定原损失律时的计算最优解

**推导类型：原Chinchilla拟合函数条件下，直接求出最优模型—数据分配；幂律指数作为原研究输入。**

**逐式来源标记。** **[T]** 训练动作与状态转移 \(A_t=\mathcal U(Z_t,D_t,B_t)\)、\(Z_{t+1}=\mathcal T(Z_t,A_t)\)，训练后的损失是学习状态的连续功能读出。**[A]** **Hoffmann 等实测拟合的** \(\mathcal L(N,D)=E+AN^{-a}+BD^{-b}\)，以及计算预算约束 \(C=6ND\)，其中指数 \(a,b\) **不是TLIR自身推导出来的**。令 \(D=C/(6N)\)，对 \(N\) 求导得到 \(aAN^{-a}=bBD^{-b}\)。**[R]** \(N_\star=(aA/bB)^{1/(a+b)}(C/6)^{b/(a+b)}\)、\(D_\star=C/(6N_\star)\)，并有 \(\mathcal L_\star-E\propto C^{-ab/(a+b)}\)。验证范围是**在原经验损失律为真时恢复计算最优条件**，不等于推出规模幂律本身。

**TLIR作用对象。** 模型大小、训练数据与动作历史共同决定学习后的状态，损失是相应状态上的功能读出。TLIR本身没有固定某个跨模型族的经验损失指数。

**使用的原理论条件。** Hoffmann等使用的拟合形式（令\(N\)为参数规模、\(D\)为训练Token量）及计算量近似\(C=6ND\)：

\[
L(N,D)=E+AN^{-a}+BD^{-b},
\qquad A,B,a,b>0,\qquad C=6ND.
\]

这里\(a,b\)取自**被外推对象已经拟合的**尺度指数，而不是从TLIR之外另造一组新指数。

**逐步推导。** 固定计算量令\(M=C/6\)，\(D=M/N\)，于是

\[
L(N)=E+AN^{-a}+BM^{-b}N^b.
\]

求导并令其为零：

\[
-aAN^{-a-1}+bBM^{-b}N^{b-1}=0
\]

\[
\Longleftrightarrow\quad aAN^{-a}=bBD^{-b}.
\]

解得

\[
\boxed{
N_\star=
\left(\frac{aA}{bB}\right)^{1/(a+b)}
\left(\frac{C}{6}\right)^{b/(a+b)},\qquad
D_\star=\frac{C}{6N_\star}
\propto C^{a/(a+b)}.
}
\]

代回原损失律：

\[
\boxed{L_\star(C)-E\propto C^{-ab/(a+b)}.}
\]

**推得的结论。** 在该论文给出的经验损失律成立的范围内，计算最优的参数与数据规模必须满足两项损失的边际平衡。**这里推导出的是最优配置，不是仅从TLIR方程“证明了幂律存在”。** TLIR提供不同训练状态、功能目标和数据条件如何进入问题的统一表述。

**研究对照。** [Hoffmann等，Training Compute-Optimal Large Language Models (2022)](https://arxiv.org/abs/2203.15556)；Kaplan等的经验规模规律。

**内部锚点。** TL-E03、TL-E04、TL-E07；跨架构TLI测试。

### 5.2.7 Dynamical Isometry：深层网络的动作—响应传输

**推导类型：由原网络的分层复合与链式法则直接推出。**

**逐式来源标记。** **[T]** 原功能响应 \(R_q=m_q(\theta+\delta\theta)-m_q(\theta)\)，其局部形式是 \(R_q\approx\nabla_\theta m_q^\top\delta\theta\)。**[A]** 原分层网络 \(h^{\ell+1}=F_\ell(h^\ell,\theta_\ell)\) 与可微链式法则，层间Jacobian记为 \(J_{L\leftarrow\ell+1}\)。逐层传播 \(\delta h^{\ell+1}=B_\ell\delta\theta_\ell\)，得 **[R]** \(\delta f\approx\sum_\ell J_{L\leftarrow\ell+1}B_\ell\delta\theta_\ell\)。这里是**局部一阶响应**；动态等距仅约束对应 Jacobian 的奇异值传输，不单独保证所有更新具有良好泛化效果。若矩阵有非零核空间，**全空间**范数下界为零，不能把仅对非零奇异子空间有效的下界误写成对所有输入成立。

**TLIR作用对象。** 一个实际参数更新\(\delta\theta\)改变输出\(f_\theta\)，其局部功能响应由当前接收状态决定。

**使用的原理论条件。** 可微的分层网络\(h^{\ell+1}=F_\ell(h^\ell,\theta_\ell)\)，以及Dynamical Isometry研究所考察的跨层雅可比及其奇异值。

**逐步推导。** 对第\(\ell\)层参数做微小改动，首先有

\[
\delta h^{\ell+1}\approx
B_\ell\delta\theta_\ell,\qquad
B_\ell=\frac{\partial F_\ell}{\partial\theta_\ell}.
\]

余下层以链式法则传递，令\(J_{L\leftarrow\ell+1}=\partial h^L/\partial h^{\ell+1}\)，于是

\[
\boxed{\delta f\approx
\sum_{\ell}J_{L\leftarrow\ell+1}B_\ell\delta\theta_\ell.}
\]

同时，对从第\(\ell\)层发出的激活扰动\(v\)，有

\[
\|J_{L\leftarrow\ell}v\|
\in[\sigma_{\min}^{\mathrm{all}}(J)\|v\|,\ \sigma_{\max}(J)\|v\|]
\]

其中\(\sigma_{\min}^{\mathrm{all}}(J)=\inf_{\|v\|=1}\|Jv\|\)，当\(J\)有非零核空间时严格为0；只有限制在与核空间正交的有效子空间，才可以使用最小非零奇异值作下界。。反传梯度沿相应转置雅可比传递。若相关非零奇异值集中于1附近，则这些方向上的信号与梯度跨层传输不发生大的范数放大或衰减；若谱极端偏离1，则出现明显的方向性放大或衰减。

**推得的结论。** Dynamical Isometry描述的是固定当前状态上**计算和梯度的传输条件**，由原网络复合结构直接推出。它不是对每个实际参数更新必然高效的承诺，因为局部参数映射\(B_\ell\)也参与实际功能响应。

**研究对照。** [Pennington、Schoenholz与Ganguli，NeurIPS 2017](https://arxiv.org/abs/1711.04735)。

**内部锚点。** TL-E03（接收状态）；TL-E04（实际更新路径）；TL-E05（更新几何）。

### 5.2.8 Dropout：随机特征屏蔽为什么等价于自适应正则化

**推导类型：使用Dropout研究自身的独立Bernoulli特征屏蔽与广义线性模型（GLM）条件，直接推出标签无关的正则项；在原GLM包含的高斯线性回归中，精确推出数据依赖的对角二次正则化，以及它怎样改变真实训练动作。**

**逐式来源标记。** **[T]** 原训练动作 \(\theta_{t+1}=\theta_t+\delta_t\) 与功能响应 \(R_{t,q}=m_q(\theta_{t+1})-m_q(\theta_t)\)。**[A]** Wager 等的独立Bernoulli特征遮蔽、倒置缩放 \(\tilde x_j=z_jx_j/r\)、GLM对数配分函数 \(A(u)\) 凸。对遮蔽取期望，标签线性项不变，得到 **[R]** \(\overline L_{\rm drop}=L_0+\frac1n\sum_i\{\mathbb E_zA(w^\top\tilde x_i)-A(w^\top x_i)\}\ge L_0\)。原高斯线性特例 \(A(u)=u^2/2\) 时，该差值精确等于 \((1-r)/(2r)\sum_j s_j^2w_j^2\)。**[T→A]** 代回更新得 \(\mathbb E_z[\delta_t\mid w_t]=-\eta\nabla\overline L_{\rm drop}\)，单次抽到的实际动作并不等于该期望。对非线性网络不能把测试期望等同关系当成精确恒等式。

**TLIR作用对象。** 式(1)–(2)把实际训练写为\(A_t=\mathcal U(Z_t,D_t,B_t)\)、\(\theta_{t+1}=\theta_t+\delta_t\)。训练时随机屏蔽特征，直接改变实际动作的形成；式(3)再测量该动作对任一固定目标的连续功能响应。式(4)区分训练时的随机屏蔽和冻结模型的推理调用。

**使用的原理论条件。** Wager、Wang与Liang（NeurIPS 2013）研究广义线性模型的**独立特征Dropout**。令保留概率\(r\in(0,1]\)，各掩码\(z_{ij}\)独立服从\(\operatorname{Bernoulli}(r)\)，并使用原研究的倒置缩放\(\widetilde x_{ij}=x_{ij}z_{ij}/r\)，因此\(\mathbb E_z[\widetilde x_i]=x_i\)。GLM的单样本负对数似然为

\[
\ell_i(w;x_i,y_i)=A(w^\top x_i)-y_iw^\top x_i-\log h(y_i),
\]

其中\(A\)是原GLM的凸对数配分函数；不引入其他正则项或新的特征噪声模型。

**第一步：由原Dropout屏蔽机制计算目标函数。** 令\(L_0(w)=n^{-1}\sum_i\ell_i(w;x_i,y_i)\)。对屏蔽变量取期望：

\[
\begin{aligned}
\overline L_{\rm drop}(w)
&=\frac1n\sum_i\mathbb E_z[\ell_i(w;\widetilde x_i,y_i)]\\
&=L_0(w)+\mathcal R_{\rm drop}(w),\\
\boxed{\mathcal R_{\rm drop}(w)
&=\frac1n\sum_i\left[
\mathbb E_z A(w^\top\widetilde x_i)-A(w^\top x_i)
\right]\ge0.}
\end{aligned}
\]

最后的不等式来自\(A\)凸、\(\mathbb E_z[\widetilde x_i]=x_i\)以及Jensen不等式。由于标签项对屏蔽取期望后不变，**\(\mathcal R_{\rm drop}\)不包含\(y_i\)**。这是原GLM条件下的精确分解，而不是额外猜测Dropout会产生何种惩罚。

**第二步：在原GLM包含的高斯线性回归中算出正则项。** 对固定方差的高斯线性模型，归一化单位方差后\(A(u)=u^2/2\)，对应平方损失\(\ell_i=\tfrac12(y_i-w^\top x_i)^2+\mathrm{const}\)。因为独立屏蔽，

\[
\operatorname{Var}_z(w^\top\widetilde x_i)
=\frac{1-r}{r}\sum_{j=1}^d w_j^2x_{ij}^2.
\]

令\(s_j^2=n^{-1}\sum_i x_{ij}^2\)，则

\[
\boxed{
\overline L_{\rm drop}(w)
=L_0(w)+\frac{1-r}{2r}\sum_{j=1}^d s_j^2w_j^2.
}
\]

**这就是Dropout与数据依赖的对角\(L_2\)正则化之间的精确关系。** 特征坐标的惩罚系数由该坐标在训练数据中的平方均值和原始Dropout保留率决定。

**第三步：代回TLIR实际更新，得到功能响应。** 对上述期望目标使用原有的梯度下降训练动作（单次随机掩码SGD的条件期望梯度也等于下式梯度），记\(D_s=\operatorname{diag}(s_1^2,\ldots,s_d^2)\)，则

\[
\boxed{
\begin{aligned}
\mathbb E_z[\delta_t\mid w_t]
&=-\eta\nabla_w\overline L_{\rm drop}(w_t)\\
&=-\eta\left[\nabla L_0(w_t)+\frac{1-r}{r}D_sw_t\right].
\end{aligned}}
\]

这里的期望指**固定接收状态和训练数据、只平均Dropout掩码**；一次真实训练使用的仍是实际抽到的掩码所形成的动作\(\delta_t\)。

对于线性模型，在固定查询\(x_q\)上的连续功能读出\(m_q(w)=w^\top x_q\)，TLIR功能响应恰好为

\[
\boxed{
R_{t,q}
=m_q(w_t+\delta_t)-m_q(w_t)
=x_q^\top\delta_t.
}
\]

所以随机屏蔽通过改变训练动作，使不同查询产生不同的目标功能响应；其期望更新含有上述自适应收缩项。**正则化并不是另一个独立的学习过程，而是原有随机训练机制对状态更新施加的确定性平均效应。**

**第四步：训练与冻结推理的关系。** 对这一线性模型，因\(\mathbb E_z[\widetilde x]=x\)，

\[
\boxed{
\mathbb E_z[f_w(\widetilde x)]=f_w(x)=w^\top x.
}
\]

故训练时随机子模型预测的平均值，与冻结权重、关闭屏蔽后的线性推理**精确相同**。原Dropout论文在非线性深网中的测试时权重缩放/模型平均属于近似，不把线性模型的等式推广为所有非线性网络的恒等式。

**推得的核心关系。** 在原广义线性模型与独立Dropout条件下，**随机特征屏蔽 \(\Rightarrow\) 标签无关的非负正则项**；在原高斯线性特例下进一步得到**精确的自适应二次惩罚 \(\Rightarrow\) 改变的实际梯度更新 \(\Rightarrow\) 查询相关的功能响应**。这与TLIR的训练—学习—冻结推理链直接衔接。

**独立研究对照。** [Srivastava等，Dropout: A Simple Way to Prevent Neural Networks from Overfitting，JMLR 2014](https://www.jmlr.org/papers/v15/srivastava14a.html)；[Wager、Wang与Liang，Dropout Training as Adaptive Regularization，NeurIPS 2013](https://papers.nips.cc/paper_files/paper/2013/hash/38db3aed920cf82ab059bfccbd02be6a-Abstract.html)。

**内部实验锚点。** TL-E03（形成的实际动作及接收状态影响功能响应）；TL-E04（有限幅度功能响应）；INF-E01（固定学习状态上的查询相关推理）。

### 5.2.9 Catastrophic Forgetting：旧目标失效的精确边界

**推导类型：直接由TLIR功能响应和正确性读出式推出，SGD梯度解释采用遗忘理论原条件。**

**逐式来源标记。** **[T]** 原响应 \(R_{t,q}=m_q(L_{t+1})-m_q(L_t)\) 与能力读出 \(C_{t,q}=\mathbf1[m_q(L_t)>0]\)。精确求和得 \(m_q(L_T)=m_q(L_0)+\sum_{t<T}R_{t,q}\)，故对最初 \(m_q(L_0)>0\) 的目标，**[R]** 失效当且仅当 \(\sum_{t<T}R_{t,q}\le-m_q(L_0)\)。**[A]** 原持续学习分析的SGD更新 \(\delta_t=-\eta\nabla L_{\rm new}(\theta_t)\) 和可微读出，另外给出**一阶近似** \(R_{t,q}\approx-\eta\nabla m_q^\top\nabla L_{\rm new}\)；旧任务损失的相应变化为 \(\Delta L_{\rm old}\approx-\eta g_{\rm old}^\top g_{\rm new}\)。请区分前一条精确边界与后一条局部近似。

**TLIR作用对象。**

\[
R_{t,q}=m_q(\theta_{t+1})-m_q(\theta_t),\qquad
C_{t,q}=\mathbf1[m_q(\theta_t)>0].
\]

**第一步：精确累计，不需要线性近似。**

\[
\boxed{
m_q(\theta_T)=m_q(\theta_0)+\sum_{t=0}^{T-1}R_{t,q}.
}
\]

对于初始已正确的旧目标\(m_q(\theta_0)>0\)，在时刻\(T\)的目标失败当且仅当

\[
\boxed{
\sum_{t=0}^{T-1}R_{t,q}\leq-m_q(\theta_0).
}
\]

同一个公式还描述保持与恢复：后续的正向功能响应可以使间隔重新跨过零边界，重新得到正确结果。

**第二步：代入被研究对象采用的SGD。** 若新任务的损失为\(L_{\rm new}\)，其更新为\(\delta_t=-\eta g_{\rm new}\)。在可微读出和局部小步长条件下，

\[
R_{t,q}\approx-\eta\nabla_\theta m_q(\theta_t)^\top g_{\rm new}.
\]

如果旧目标采用可微损失\(L_{\rm old}\)，则

\[
\boxed{
\Delta L_{\rm old}
\approx-\eta g_{\rm old}^\top g_{\rm new}.
}
\]

梯度点积为负时，当前新任务更新使旧损失的一阶变化为正；为正时则为负。多个任务共享同一批参数，所以一次动作可同时改变许多目标的间隔。

**推得的结论。** 遗忘首先是旧目标函数的计算结果越过正确性边界；它不是“能力实体”从参数中消失。保护旧任务的方法本质上是在改变旧目标上累计的功能响应轨迹。

**研究对照。** Kirkpatrick等的EWC；Lopez-Paz与Ranzato的GEM；梯度投影与经验回放研究。

**内部锚点。** TL-E03、TL-E07、TL-E08；[RL-E06](experiments/gfg_rl_selective_positive_feedback_dose_recovery_v1/RESULTS.md)（先掌握、后退化、再恢复）。

### 5.2.10 Grokking：训练/测试读出的不同跨界时间

**推导类型：使用被外推Grokking可解研究的线性教师—学生模型、实际梯度流与随机矩阵谱，直接推出训练/测试损失轨迹以及延迟泛化时间。**

**逐式来源标记。** **[T]** 原状态更新 \(\theta_{t+1}=\theta_t+\delta_t\) 在原研究的梯度流极限中给出连续状态 \(D(t)\)；TLIR响应 \(R_{t,q}=m_q(L_{t+1})-m_q(L_t)\)，能力读出 \(C_{t,q}=\chi_q(m_q(L_t))\)。**[A]** Levi、Beck与Bar-Sinai的高斯线性教师—学生模型、零权重衰减、平方损失梯度流、训练协方差 \(\Sigma_{\rm tr}\)、总体协方差 \(I\)，以及原文使用的Marčenko–Pastur谱和同一正确率阈值。代入更新得 \(D(t)=e^{-2\eta_0\Sigma_{\rm tr}t}D_0\)，同一状态的两个连续读出分别是 \(L_{\rm tr}(t)=D_0^\top e^{-4\eta_0\Sigma_{\rm tr}t}\Sigma_{\rm tr}D_0\) 与 \(L_{\rm gen}(t)=D_0^\top e^{-4\eta_0\Sigma_{\rm tr}t}D_0\)。对**原研究的**准确率阈值 \(L=\epsilon/4\)，可等价选择TLIR读出 \(m_q=\epsilon/4-L_q\)、\(\chi_q(m)=\mathbf1[m\ge0]\)。由原谱边缘渐近得到 **[R]** \(\Delta t_{\rm grok}\simeq-\ln(1-\sqrt\lambda)/[2\eta_0(1-\sqrt\lambda)^2]\)，其中 \(0<\lambda<1\)。这个时间公式属于**特定渐近区域的结果**，不是仅由TLIR形式方程独立确定的常数。

**TLIR作用对象。** 式(2)实际参数更新和式(3)/(17)的目标连续读出、离散正确性边界。训练集目标和测试分布目标是**同一个学习状态**的两类读出；正确性可以先后跨越各自的阈值。

**使用的原理论条件。** Levi、Beck、Bar-Sinai（ICLR 2024）的无偏置线性教师—学生设定：输入\(x\sim\mathcal N(0,I_d)\)，教师参数\(T\)、学生参数\(S_t\)；平方损失，标量输出，\(n\)个固定训练样本，零权重衰减，梯度流；训练协方差\(\Sigma_{\rm tr}=\frac1n\sum_jx_jx_j^\top\)，总体协方差为\(I_d\)，高维极限\(d,n\to\infty\)，\(\lambda=d/n\in(0,1)\)固定；原论文的Marčenko–Pastur谱、各向同性初始化，以及误差阈值\(\epsilon\)。

**第一步：由TLIR更新代入原模型的优化动作。** 记\(D_t=S_t-T\)，训练与测试的两个连续读出为

\[
L_{\rm tr}(D)=D^\top\Sigma_{\rm tr}D,\qquad
L_{\rm gen}(D)=D^\top D.
\]

原梯度流在时间尺度\(\eta_0\)下满足

\[
\dot D(t)=-2\eta_0\Sigma_{\rm tr}D(t),\qquad
\boxed{D(t)=e^{-2\eta_0\Sigma_{\rm tr}t}D_0}.
\]

代入同一状态的两个读出，得**精确**损失演化：

\[
\boxed{
\begin{aligned}
L_{\rm tr}(t)&=D_0^\top e^{-4\eta_0\Sigma_{\rm tr}t}\Sigma_{\rm tr}D_0,\\
L_{\rm gen}(t)&=D_0^\top e^{-4\eta_0\Sigma_{\rm tr}t}D_0.
\end{aligned}}
\]

**第二步：代入原论文使用的协方差谱。** 设\(\nu\)遵循\(\lambda=d/n\)下的Marčenko–Pastur谱，在原论文的各向同性初始方向及大维度近似下：

\[
L_{\rm tr}(t)\approx\mathbb E_\nu[\nu e^{-4\eta_0\nu t}],\qquad
L_{\rm gen}(t)\approx\mathbb E_\nu[e^{-4\eta_0\nu t}].
\]

其正谱下端为\(\nu_-=(1-\sqrt\lambda)^2\)。当\(\eta_0t\gg\sqrt\lambda/(1-\sqrt\lambda)^2\)时，原论文的谱端点渐近给出

\[
\boxed{
L_{\rm tr}(t)\simeq
\frac{e^{-4\eta_0(1-\sqrt\lambda)^2t}}
{16\sqrt{\pi}\lambda^{3/4}(\eta_0t)^{3/2}},
\qquad
L_{\rm gen}(t)\simeq\frac{L_{\rm tr}(t)}{(1-\sqrt\lambda)^2}.
}
\]

**第三步：用原论文的正确率读出计算迟发时间。** 每个输入的正确性由误差是否小于\(\epsilon\)定义：

\[
C_x(t)=\mathbf1[(D(t)^\top x)^2<\epsilon].
\]

在相应高斯读出近似下

\[
A(t)=\operatorname{erf}\!\left(\sqrt{\frac{\epsilon}{2L(t)}}\right).
\]

原论文用\(A=\operatorname{erf}(\sqrt2)\approx95.45\%\)作为跨界标准，对应\(L(t^*)=\epsilon/4\)。分别求\(L_{\rm tr}(t_{\rm tr}^*)\)和\(L_{\rm gen}(t_{\rm gen}^*)\)到达该阈值的时间，在小\(\epsilon\)渐近下：

\[
\boxed{
\Delta t_{\rm grok}=t_{\rm gen}^*-t_{\rm tr}^*
\simeq-\frac{\ln(1-\sqrt\lambda)}
{2\eta_0(1-\sqrt\lambda)^2}>0,\quad 0<\lambda<1.
}
\]

**核心结论。** 训练、测试都从同一条连续参数轨迹得到；在原模型的两种读出几何及同一正确性阈值下，测试正确率可远晚于训练正确率跃升。时间差不仅存在，而且由原研究的\(\lambda,\eta_0\)条件定量给出。整个过程无需假设外部读出跃升时突然生成独立“能力”。

**研究对照。** [Levi、Beck与Bar-Sinai，Grokking in Linear Estimators—A Solvable Model that Groks without Understanding，ICLR 2024](https://proceedings.iclr.cc/paper_files/paper/2024/hash/8c0a85ee3eddb1858d9142f0d51d67b2-Abstract-Conference.html)；[Žunkovič与Ilievski，Grokking phase transitions in learning local rules with gradient descent，JMLR 2024](https://www.jmlr.org/papers/v25/22-1228.html)。

**内部实验锚点。** TL-E03（接收状态决定响应）；TL-E07（训练继续时支撑可重组）；TL-E08（连续间隔与离散边界）；TL-P01（目标边界预测）。

### 5.2.11 μP：宽度增长时保持非退化功能更新

**推导类型：使用Tensor Programs IV的原MLP参数化与SGD条件，直接得到更新量级和输出稳定性；进一步使用该论文的第一层修改，说明各层特征学习的条件。**

**逐式来源标记。** **[T]** 原实际动作 \(\theta_{t+1}=\theta_t+\delta_t\) 和功能响应 \(R_{t,q}=m_q(\theta_t+\delta_t)-m_q(\theta_t)\)。**[A]** Yang与Hu《Tensor Programs IV》附录E.1的线性MLP参数化、输出缩放 \(f=n^{-1/2}v\bar h\)、相关权重尺度与学习率 \(\eta_n\propto n^{-c}\)。对该论文的**同一SGD动作**按链式法则计算中间特征变化和输出响应，得到 **[R]** \(\Delta\bar h_j=\Theta(n^{-c})\)、\(\Delta f=\Theta(n^{-c})\)；当 \(c=0\) 时单步两者均为 \(\Theta(1)\)。这是原附录的**单步、特定参数化量级**；不据此直接声称完整μP所有层及μTransfer定理都已由此独立证明。

**TLIR作用对象。** 式(2) \(\theta_{t+1}=\theta_t+\delta_t\)和式(3) \(R_q(\theta,\delta)=m_q(\theta+\delta)-m_q(\theta)\)。跨宽度比较必须检查实际参数动作在**功能/表征空间**造成的响应，而不能只比较参数增量数值。

**使用的原理论条件。** Yang与Hu（ICML 2021，Tensor Programs IV）附录E的两隐藏层线性MLP演示及其SGD宽度极限：隐藏宽度\(n\to\infty\)，\(h=U\xi\)、\(\bar h=Wh\)，\(h^\top h=\Theta(n)\)，中间层与输出层原始高斯初始化尺度为\(n^{-1/2}\)，损失对输出的导数\(\chi=\mathcal L'(f,y)=\Theta(1)\)。该附录给出输出缩放\(f=n^{-1/2}v\bar h\)，学习率\(\eta_n\propto n^{-c}\)，并研究固定输入上的单次实际SGD动作。

**第一步：把原参数化与动作代入TLIR响应。** 对中间层的一个SGD动作，该论文附录E.1直接按链式求导得到

\[
\Delta\bar h
=-n^{-c-1/2}\chi\,v^\top(h^\top h),\qquad
\Delta f
=-n^{-c-1}\chi(vv^\top)(h^\top h).
\]

在它自己的初始化尺度下，每个\(v_j=\Theta(n^{-1/2})\)、\(vv^\top=\Theta(1)\)、\(h^\top h=\Theta(n)\)，故

\[
\boxed{
\Delta\bar h_j=\Theta(n^{-c}),\qquad
\Delta f=\Theta(n^{-c}).
}
\]

若\(c>0\)，固定步数内特征与输出更新趋零；若\(c=0\)，则

\[
\boxed{\Delta\bar h_j=\Theta(1),\qquad \Delta f=\Theta(1).}
\]

这从原前向计算和实际反向更新直接给出**宽度增长但功能更新仍非退化**的量级条件。

**第二步：为什么必须连输出缩放一起调整。** 原论文将上述\(n^{-1/2}\)的输出缩放与标准参数化对照：按原标准输出缩放、却让隐藏特征发生\(\Theta(1)\)的更新时，输出权重与已更新特征的相关项可使logit达到\(\Theta(\sqrt n)\)并失去稳定性；采用附录E.1的μP输出缩放时该项为\(\Theta(1)\)。因此不能只要求某一层“大幅更新”，还要同时控制输出的稳定量级。

**第三步：第一层也参与特征学习。** 原论文进一步把第一层参数化指数改为\(a_1=-1/2,b_1=1/2\)（并在输出层取\(a_{L+1}=1/2,c=0\)），使输入层的实际更新传入隐藏特征也保持非退化；不同层须分别满足相应的初始化和更新缩放条件。该条件来自Tensor Programs IV的参数化定义与证明，而不是额外假设各层采用相同学习率。

**核心结论。** μP的核心关系是：**在参数宽度增加时，调整各层的实际更新与前向缩放，使学习形成的功能变化保持有限而不消失、同时不导致logit发散。** 上述原MLP例子完成了这一关系的量级推导。Tensor Programs V的μTransfer进一步研究此参数化下最优超参数跨宽度的稳定性，其完整参数族证明与实测迁移效果仍属于原研究的进一步结果，不把“\(\Theta(1)\)响应”直接等同于“任意超参数必然可迁移”。

**研究对照。** [Yang与Hu，Tensor Programs IV: Feature Learning in Infinite-Width Neural Networks，ICML 2021](https://proceedings.mlr.press/v139/yang21c.html)，特别是[附录E](https://proceedings.mlr.press/v139/yang21c/yang21c-supp.pdf)；[Yang等，Tensor Programs V: Tuning Large Neural Networks via Zero-Shot Hyperparameter Transfer，NeurIPS 2021](https://proceedings.nips.cc/paper_files/paper/2021/hash/8df7c2e3c3c3be098ef7b382bd2c37ba-Abstract.html)。

**内部实验锚点。** TL-E03（接收状态决定实际功能响应）；TL-E04（有限幅度更新的非线性）；TL-E05（目标与动作几何）。

### 5.2.12 Sharpness与SAM：同一函数的不同参数几何与训练动作

**推导类型：使用原研究的函数保持重参数化与SAM的邻域优化定义，得到两个可以同时成立的结论。**

**逐式来源标记。** **[T]** 原训练动作 \(\theta_{t+1}=\theta_t+\delta_t\)、响应 \(R_{t,q}=m_q(\theta_{t+1})-m_q(\theta_t)\)。**[A]** Dinh等的函数保持重参数化，以及Foret等SAM的参数邻域目标。取原思想的二参数乘积特例 \(f(a,b)=ab\)：在 \(ab=1\) 时，坐标重标定 \((a,b)\mapsto(ca,b/c)\) 保持输出却把Hessian非零特征值改成 **[R]** \(\lambda_{\max}'=c^2a^2+b^2/c^2\)。原SAM目标 \(\max_{\|\epsilon\|\le\rho}L(\theta+\epsilon)\) 的一阶扰动 \(\epsilon^\star\approx\rho g/\|g\|\) 使训练动作变为 \(\delta_{\rm SAM}\approx-\eta\nabla L(\theta+\epsilon^\star)\)。两者描述不同对象：前者是参数几何，后者是执行动作；**两个近似号不能改成精确等号**。

**TLIR作用对象。** 参数状态与目标函数读出不是一回事。参数坐标可变化而输入—输出函数不变；训练策略则通过改变实际动作\(\delta_t\)改变后续状态。

**使用的原理论条件。** Sharpness重参数化研究使用的正齐次/乘积参数冗余；SAM论文使用的参数邻域最坏损失目标。以下取相同原理的最简单二参数乘积计算。

**逐步推导：函数不变，参数锐度改变。** 令模型输出\(f(a,b)=ab\)，平方损失

\[
L(a,b)=\tfrac12(ab-1)^2.
\]

对任意\(c>0\)，变换\((a,b)\mapsto(ca,b/c)\)保持\(ab\)及函数结果完全不变。在达到拟合\(ab=1\)的点，

\[
H=\nabla^2L=
\begin{pmatrix}b^2&ab\\ab&a^2\end{pmatrix}
\]

的特征值为\(0\)和\(a^2+b^2\)。重参数化后，最大特征值成为

\[
\boxed{\lambda_{\max}'=c^2a^2+b^2/c^2,}
\]

可随\(c\)显著改变，**同一功能结果却有不同参数锐度**。

**逐步推导：SAM改变实际训练动作。** SAM的原目标是

\[
L_{\rm SAM}(\theta)=
\max_{\|\epsilon\|\leq\rho}L(\theta+\epsilon).
\]

对当前点做一阶展开，若\(g=\nabla L(\theta)\ne0\)，最坏扰动近似为

\[
\epsilon^\star\approx \rho\frac{g}{\|g\|},
\qquad
\delta_{\rm SAM}\approx-\eta\nabla L(\theta+\epsilon^\star).
\]

这与普通SGD的\(-\eta g\)是不同的训练动作，从而可产生不同的实际功能响应。

**推得的结论。** “参数锐度不是唯一的函数性质”与“使用邻域损失设计训练动作可以改变训练结果”不矛盾：一个说的是参数坐标度量，另一个说的是状态更新规则。

**研究对照。** [Dinh等，Sharp Minima Can Generalize For Deep Nets](https://arxiv.org/abs/1703.04933)；[Foret等，SAM，ICLR 2021](https://research.google/pubs/sharpness-aware-minimization-for-efficiently-improving-generalization/)。

**内部锚点。** TL-E03、TL-E04、TL-E05。

### 5.2.13 Gradient Noise Scale：随机梯度如何变成目标功能响应噪声

**推导类型：沿用原批量梯度独立采样条件，直接计算参数与功能空间方差。**

**逐式来源标记。** **[T]** 原响应 \(R_{t,q}=m_q(\theta_t+\delta_t)-m_q(\theta_t)\)。**[A]** McCandlish等使用的独立样本梯度与普通SGD条件：\(g_i\) 的均值 \(g\)，协方差 \(\Sigma_g\)，批量大小 \(B\)；此处 \(B\) 是**batch size**，不同于TLIR反馈变量 \(B_t\)。由 \(\delta=-\eta B^{-1}\sum_{i=1}^{B}g_i\) 得精确的参数动作协方差 \(\operatorname{Cov}(\delta)=\eta^2\Sigma_g/B\)。再将 **[T]** 的 \(R\) 在接收点一阶展开，**[R]** \(\operatorname{Var}(R_{t,q})\approx\eta^2 j_q^\top\Sigma_gj_q/B\)，\(\operatorname{Cov}(\Delta f)\approx\eta^2 J\Sigma_gJ^\top/B\)；原梯度噪声尺度是 \(\operatorname{tr}\Sigma_g/\|g\|^2\)。这里的功能方差关系是**线性化近似**。

**TLIR作用对象。** 实际动作\(\delta_t\)作用于当前接收状态，目标响应\(R_q(\theta_t,\delta_t)\)由该动作和目标决定。

**使用的原理论条件。** 从同一固定训练分布独立抽取\(B\)个样本；单样本梯度\(g_i\)有均值\(g\)、协方差\(\Sigma_g\)；采用普通SGD、小步长局部线性近似。这是梯度噪声规模分析的基础条件。

**逐步推导。** 平均梯度及其协方差：

\[
\hat g_B=\frac1B\sum_{i=1}^B g_i,\qquad
\mathbb E[\hat g_B]=g,\qquad
\operatorname{Cov}(\hat g_B)=\frac{\Sigma_g}{B}.
\]

实际参数更新为\(\delta=-\eta\hat g_B\)，故

\[
\boxed{
\operatorname{Cov}(\delta)=\frac{\eta^2}{B}\Sigma_g.
}
\]

对一个固定目标\(q\)，令其功能读出的参数梯度为\(j_q=\nabla_\theta m_q(\theta)\)，则

\[
R_q(\theta,\delta)\approx j_q^\top\delta
=-\eta j_q^\top\hat g_B.
\]

因而

\[
\boxed{
\begin{aligned}
\mathbb E[R_q]&\approx-\eta j_q^\top g,\\
\operatorname{Var}(R_q)&\approx
\frac{\eta^2}{B}j_q^\top\Sigma_gj_q.
\end{aligned}}
\]

对输出向量\(f\)及其雅可比\(J\)，同理得到

\[
\boxed{
\operatorname{Cov}(\Delta f)\approx
\frac{\eta^2}{B}J\Sigma_gJ^\top.
}
\]

**推得的结论。** 批量大小不仅改变更新噪声，还通过当前模型的目标敏感方向，改变**具体任务的功能响应噪声**。原研究中的梯度噪声规模可由\(\operatorname{tr}\Sigma_g/\|g\|^2\)表征其典型量级；是否为有效批量瓶颈仍由原优化问题决定。

**研究对照。** [McCandlish等，An Empirical Model of Large-Batch Training](https://arxiv.org/abs/1812.06162)。

**内部锚点。** TL-E03（接收状态）；TL-E05（目标—更新几何）。

### 5.2.14 Double Descent：高斯线性原模型中的显式风险曲线

**推导类型：只采用原研究的各向同性高斯设计和最小范数/最小二乘解，直接推导插值边界两侧的期望风险。**

**逐式来源标记。** **[T]** 原状态更新 \(\theta_{t+1}=\theta_t+\delta_t\)；将测试总体风险记作同一最终学习状态的功能读出 \(m_{\rm test}(L)\)，其步间变化由 \(R_{t,\rm test}=m_{\rm test}(L_{t+1})-m_{\rm test}(L_t)\) 给出。**[A]** Hastie等各向同性零信号高斯回归的原始模型 \(X_{ij}\sim N(0,1)\)、\(y=\varepsilon\sim N(0,\sigma^2I_n)\)、独立测试协方差 \(I_p\)、零初始化最小二乘/最小范数解。通过原训练动作收敛到对应解：欠参数 \(\hat\beta=(X^\top X)^{-1}X^\top\varepsilon\)，过参数 \(\hat\beta=X^\top(XX^\top)^{-1}\varepsilon\)。代入**原模型的**总体风险读出并使用Wishart逆期望，得到 **[R]** \(\mathbb E R_{\rm test}=\sigma^2 p/(n-p-1)\)（\(p<n-1\)）或 \(\sigma^2 n/(p-n-1)\)（\(p>n+1\)）。这是噪声**超额风险**而非包含不可约噪声的总MSE；曲线的定量形式来自原高斯设定，不由TLIR单独给出。

**TLIR作用对象。** 模型规模\(p\)改变可选的参数化函数空间；训练动作选择一个最终参数解；测试误差是该解对新输入的功能读出。它不是独立的“能力实体”。

**使用的原理论条件。** 取经典Double Descent研究中的纯噪声各向同性线性模型：\(X\in\mathbb R^{n\times p}\)各元素独立服从\(N(0,1)\)，\(y=\varepsilon\)且\(\varepsilon\sim N(0,\sigma^2I_n)\)，测试\(x\sim N(0,I_p)\)与训练独立；\(p<n\)用普通最小二乘，\(p>n\)用最小范数插值解。此处计算的是测试**超额预测风险**，不包含新标签自身的不可约噪声。

**逐步推导：欠参数区间\(p<n-1\)。**

\[
\hat\beta=(X^\top X)^{-1}X^\top\varepsilon.
\]

对训练噪声与独立测试输入取期望，

\[
\mathbb E R
=\sigma^2\,\mathbb E_X
\operatorname{tr}[(X^\top X)^{-1}].
\]

利用该原高斯设计的Wishart逆矩阵期望，得到

\[
\boxed{\mathbb ER=\sigma^2\frac{p}{n-p-1},\quad p<n-1.}
\]

**逐步推导：过参数区间\(p>n+1\)。**

\[
\hat\beta=X^\top(XX^\top)^{-1}\varepsilon,\qquad
\mathbb E R=\sigma^2\mathbb E_X
\operatorname{tr}[(XX^\top)^{-1}],
\]

从而

\[
\boxed{\mathbb ER=\sigma^2\frac{n}{p-n-1},\quad p>n+1.}
\]

在插值阈值附近，这些逆矩阵期望发散；远离阈值、\(p\)继续增大后，第二式下降。这样便直接得到**先升至峰值、再下降**的Double Descent风险形状。

**推得的结论。** 这个反常曲线是原线性回归的解与训练/测试分布共同确定的计算结果；无需另加“参数变多后噪声支撑必然重新分配”的新假设。对非零真实信号或一般协方差，可继续使用5.2.5的偏差—方差分解及原研究条件。

**研究对照。** [Hastie、Montanari、Rosset与Tibshirani，Annals of Statistics 2022](https://doi.org/10.1214/21-AOS2133)；Belkin等的插值阈值研究。

**内部锚点。** TL-E03（参数更新与状态）；TL-E08（目标依赖读出）。
