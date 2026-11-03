# TLIR：14项理论外推逐项来源与数学核验报告

**审核日期：** 2026-11-03  
**基准：** `TLIR_UNIFIED_DYNAMICS_V5.md`、`TLIR_THEORY_FIRST_14_DERIVATIONS.md` 与 `TLIR_SECTION5_21_PAIRS_14_EXTRAPOLATIONS.md`。  
**标记：** <sup>[T]</sup> TLIR原方程；<sup>[A]</sup> 原研究数学模型／假设；<sup>[R]</sup> 经过代入和运算恢复的关系。

## 总体结果

14项都保留了原研究对象与主要闭式公式，并补入了TLIR原方程、原研究数学条件与具体代入链。以下“核验”是对明确列出的数学关系及适用边界的核验，不等于已经重跑原论文全部实验或复证其全部定理。

| 编号 | 主题 | 推导状态与限制 | 核验要点 |
|---|---|---|---|
| 1 | Edge of Stability | **局部线性化** | 代入原GD动作后，特征方向变为 $e_{i,t+1}=(1-\eta\lambda_i)e_{i,t}$；$2/\eta$是正曲率方向的局部稳定边缘，不是任意非凸轨迹的全局常数。 |
| 2 | Implicit Bias | **一维直接推导；多维借用原定理** | 正确性饱和时更新仍正、$w_t=\log t+O(1)$；最大间隔方向是Soudry等独立证明的原有定理，不能由一维例子单独推出。 |
| 3 | Model Collapse | **有限离散原模型下精确；混合数据结果是期望关系** | $\Pr(q_{t+1}=0\mid q_t)=(1-q_t)^M$ 且 $E[q_{t+1}\mid q_t]=q_t$ 同时成立；支持集随机收缩不等于平均概率逐代衰减。 |
| 4 | Spectral Bias | **冻结核近似，频率顺序有条件** | 模态残差按 $(1-\eta\lambda_k)^t$ 衰减；低频优先必须由原模型频谱顺序提供。 |
| 5 | Benign Overfitting | **噪声条件期望下精确** | 由实际GD选出的最小范数解，导出偏差和方差两项；不能混称单次实验风险。 |
| 6 | Chinchilla | **在原经验幂律下精确求最优分配** | 微分得到 $aAN^{-a}=bBD^{-b}$、最优$N,D$与最优损失指数；TLIR不单独推出经验幂指数。 |
| 7 | Dynamical Isometry | **一阶动作响应，奇异值界已修正** | 由链式法则推出局部输出响应；Jacobian若有核空间，全空间最小增益为0，不能使用“最小非零奇异值”为所有方向提供下界。 |
| 8 | Dropout | **GLM期望分解精确；高斯线性正则精确** | 独立Bernoulli遮蔽使标签线性项期望不变；平方损失下得到特征平方均值加权的$L_2$项。单步实际遮蔽动作不等于条件期望动作。 |
| 9 | Catastrophic Forgetting | **累计响应/跨界条件精确；梯度内积一阶** | 望远镜求和与旧目标错误边界不依赖梯度近似；新旧任务梯度夹角解释依赖小步长。 |
| 10 | Grokking | **原线性教师—学生模型的谱端点渐近** | 原状态演化和训练/泛化两个读出与ICLR 2024数学式一致；延迟公式保留$\sqrt\lambda$及原阈值的适用条件。 |
| 11 | μP | **原附录E的单步量级推导** | $\Delta\bar h_j,\Delta f=\Theta(n^{-c})$；不能由此替代整套μP参数族定理或μTransfer的独立证明。 |
| 12 | SAM/Sharpness | **乘积模型Hessian精确；SAM动作一阶近似** | $(a,b)\mapsto(ca,b/c)$保持函数但改变Hessian特征值；SAM最坏邻域扰动与优化动作使用近似号。 |
| 13 | Gradient Noise Scale | **参数动作协方差精确；功能方差一阶** | 采用独立采样时 $\operatorname{Cov}(\delta)=\eta^2\Sigma_g/B$，雅可比投影给出查询相关响应噪声。$B$为批大小，区别于反馈$B_t$。 |
| 14 | Double Descent | **高斯零信号最小二乘模型下的有限样本期望** | Wishart逆矩阵期望在 $p<n-1$ 和 $p>n+1$ 产生分段噪声超额风险，不能当作所有网络的普适精确风险律。 |

## 数学来源对照（部分原始论文/正式页面）

- Edge of Stability：[Cohen et al., ICLR 2021](https://openreview.net/forum?id=jh-rTtvkGeM)。
- Implicit Bias：[Soudry et al., JMLR 2018](https://jmlr.org/papers/v19/18-188.html)。
- 递归模型坍塌：[Shumailov et al., *Nature* 2024](https://www.nature.com/articles/s41586-024-07566-y)。
- 计算最优规模分配：[Hoffmann et al., NeurIPS 2022](https://proceedings.neurips.cc/paper_files/paper/2022/hash/c1e2faff6f588870935f114ebe04a3e5-Abstract.html)。
- Dynamical Isometry：[Pennington et al., NeurIPS 2017](https://research.google/pubs/resurrecting-the-sigmoid-in-deep-learning-through-dynamical-isometry-theory-and-practice/)。
- Dropout正则化：[Wager et al., NeurIPS 2013](https://papers.nips.cc/paper_files/paper/2013/hash/38db3aed920cf82ab059bfccbd02be6a-Abstract.html)。
- Grokking：[Levi, Beck & Bar-Sinai, ICLR 2024](https://proceedings.iclr.cc/paper_files/paper/2024/hash/8c0a85ee3eddb1858d9142f0d51d67b2-Abstract-Conference.html)，其延迟公式为正文式(11)。
- μP：[Yang & Hu, ICML 2021](https://proceedings.mlr.press/v139/yang21c.html)，见论文附录E。
- Sharpness：[Dinh et al., ICML 2017](https://proceedings.mlr.press/v70/dinh17b.html)；SAM：[Foret et al., ICLR 2021](https://research.google/pubs/sharpness-aware-minimization-for-efficiently-improving-generalization/)。
- Gradient Noise：[McCandlish, Kaplan & Amodei, 2018](https://openai.com/index/how-ai-training-scales/)。
- Double Descent：[Hastie et al., *Annals of Statistics* 2022](https://pmc.ncbi.nlm.nih.gov/articles/PMC9481183/)。

## 本次代码与形式核验

独立脚本 `verify_14.py` 运行 **21/21 项检查通过**。内容包括：局部Hessian稳定边界、logistic训练保持更新、Binomial条件矩、冻结核残差、偏差—方差矩阵迹恒等式、Chinchilla边际平衡、Jacobian核空间、Dropout对所有有限Bernoulli掩码的精确枚举、旧能力边界、Grokking谱积分的小阈值渐近方向、SAM坐标变换、梯度噪声投影，以及Double Descent高斯Wishart两侧的随机样本模拟。

**检查范围：** 数值抽查可发现公式错误或符号误用，但不能取代原论文证明；对原有定理借用处明确标注了范围。重点改正：V5编号混用、Dynamical Isometry零空间下界、严格结果与一阶近似的归属，以及Grokking/μP/Chinchilla数学条件。

## 仓库同步范围

- `TLIR_THEORY_FIRST_14_DERIVATIONS.md`：14项全部添加逐式来源和适用范围。
- `TLIR_SECTION5_21_PAIRS_14_EXTRAPOLATIONS.md`：同步上述14项，保留原21项正反解释。
- `TLIR_UNIFIED_DYNAMICS_V5.md`：原始动力学方程组保持不变。

**结论：** 14项均已被改写为可追踪的“TLIR原式 [T] → 原研究条件 [A] → 实际代入与计算 → 恢复关系 [R]”，并对各自的数学强度与边界做了标记；它们作为统一方程在不同模型下的可恢复结果进行检验。