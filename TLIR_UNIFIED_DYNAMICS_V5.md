# Unified Dynamics of Training, Learning, Inference, and Reinforcement (TLIR) — V5

**An experimentally established dynamical theory of neural-network functional organization**

## 1. Scientific objects and notation

TLIR relates four processes within one learned system: **training** forms and executes updates; **learning** is the persistent functional reorganization produced by training; **inference** recruits and combines the support already present in a fixed learned state; and **reinforcement** returns behavioral consequences to subsequent training. The same persistent-state transition gives rise to target-specific functional responses, changes in distributed support, and observable capability outcomes.

The theory distinguishes a **training action**, the **state receiving that action**, and the **functional consequences** measured after it. These objects are separately accessible through the realized-update interventions, component probes, exact state restorations, and matched feedback branches used in the experiments.

### 1.1 State and training quantities

| Symbol | Definition |
|---|---|
| \(t\) | Index of an executed training transition. |
| \(L_t\) | Persistent learned state that determines the model's functional behavior. In the tested parameterized networks, it is represented primarily by parameters \(\theta_t\), together with any persistent model variables relevant to the function. |
| \(\omega_t\) | Training-mechanism state, such as Adam/AdamW moments, SGD momentum, or Adagrad accumulators. |
| \(Z_t=(L_t,\omega_t)\) | Complete receiving state for the registered training transition. |
| \(D_t\) | Training information supplied at the transition: identified inputs, targets, and other registered training data. |
| \(B_t\) | Feedback information available to the training mechanism, including relevant consequences of preceding interactions. |
| \(A_t\) | Actually formed and executed training action, carrying the realized parameter and training-state changes required by the transition. |
| \(\mathcal U\) | Action-formation rule for the executed training mechanism. |
| \(\mathcal T\) | Persistent-state transition induced by executing \(A_t\) on \(Z_t\). |
| \(\delta_t=\theta_{t+1}-\theta_t\) | Realized parameter increment, measured from exact before/after parameter versions. |

### 1.2 Functional, inference, and feedback quantities

| Symbol | Definition |
|---|---|
| \(q\) | An identified evaluation target, including its input and target identity. |
| \(m_q(L)\) | Continuous target-specific functional readout of learned state \(L\); in boundary experiments, a signed target margin. |
| \(R_{t,q}\) | Actual change in \(m_q\) caused by the registered training transition. |
| \(\mathcal J_q\) | Fixed intervention-and-measurement protocol used to measure target \(q\)'s functional support. |
| \(\mathcal P_q(L;\mathcal J_q)\) | Intervention-measured functional support profile, possibly vector-valued, at learned state \(L\). |
| \(P_{t,q},\Delta P_{t,q}\) | Measured support and its change across one persistent-state transition. |
| \(\chi_q\) | Target-specific rule converting the continuous readout into an observable capability result. |
| \(C_{t,q}\) | Capability result for \(q\) at state \(L_t\). |
| \(x,c\) | Inference query/input and its context. The \(c\) in this pair denotes context, not a competing class. |
| \(L^*\) | Learned state held fixed during an inference occurrence. |
| \(\Pi_{x,c}\) | Query-conditioned recruitment of functional support from \(L^*\). The symbol denotes recruitment/selection, including nonlinear operations, rather than requiring a linear projection. |
| \(S^{\mathrm{active}}_{x,c}\) | Functionally active support recruited for the particular query and context. |
| \(H_{x,c}\) | Execution and combination of the recruited support to produce an inference result. |
| \(Y_{x,c},Y_t\) | Output of a particular frozen inference, or the output participating in feedback cycle \(t\). |
| \(E_t\) | Environment consequence of the associated behavior. |
| \(\mathcal B\) | Rule that forms the next feedback state from prior feedback, behavior, and consequence. |

**Readout conventions.** For a classified target \(q=(x,y)\) with logits \(z_k(L,x)\), a concrete signed margin is

\[
m_q(L)=z_y(L,x)-\max_{k\ne y}z_k(L,x),
\qquad
\chi_q(m)=\mathbf 1[m>0].
\]

The diffusion experiment uses a distinct, frozen residual-error boundary,

\[
m_q(L)=\min_k\bigl(E_k(L)-E_{\mathrm{true}}(L)\bigr),
\]

where \(q\) identifies an image, diffusion timestep, and noise occurrence; its registered equality convention is included in \(\chi_q\). Thus \(m_q\) is the system's measured functional readout, and \(\chi_q\) specifies its observed decision rule.

---

## 2. Coupled dynamical equations

### 2.1 Training: realized action and persistent state

\[
\boxed{
\begin{aligned}
Z_t&=(L_t,\omega_t),\\
A_t&=\mathcal U(Z_t,D_t,B_t),\\
Z_{t+1}&=\mathcal T(Z_t,A_t).
\end{aligned}}
\tag{1}
\]

Training is the concrete execution of \(A_t\). The receiving state matters both in forming the action and in determining its realized effect. The optimizer-memory coordinate \(\omega_t\) is retained because experiments that exchange Adam/AdamW, momentum, or Adagrad states change subsequent action formation while other registered inputs are held fixed.

For a parameterized implementation, the parameter component of the same transition can be read as

\[
\boxed{\delta_t=\Psi(\theta_t,\omega_t,D_t,B_t),\qquad
\theta_{t+1}=\theta_t+\delta_t.}
\tag{2}
\]

Here \(\Psi\) names the *executed optimizer-specific update formation*. The functional response to a formed \(\delta_t\) is a separate object from the process that formed it.

### 2.2 Learning: three coupled observations of the same transition

\[
\boxed{
\begin{aligned}
R_{t,q}
&=m_q(L_{t+1})-m_q(L_t),\\
P_{t,q}
&=\mathcal P_q(L_t;\mathcal J_q),\\
\Delta P_{t,q}
&=\mathcal P_q(L_{t+1};\mathcal J_q)
 -\mathcal P_q(L_t;\mathcal J_q),\\
C_{t,q}
&=\chi_q\bigl(m_q(L_t)\bigr),\\
C_{t+1,q}
&=\chi_q\bigl(m_q(L_t)+R_{t,q}\bigr).
\end{aligned}}
\tag{3}
\]

These are **different measurements of one transition** \(Z_t\rightarrow Z_{t+1}\):

- \(R_{t,q}\) measures the change in a target's continuous functional response.
- \(\Delta P_{t,q}\) measures the persistent redistribution of function across intervened components.
- \(C_{t+1,q}\) records the capability boundary reached after that response.

Learning is the **persistent functional reorganization** caused by actual training. It can be detected through changes in intervention-measured support even while an external correctness result remains unchanged. The measured response, support change, and capability result share the state transition as their source.

### 2.3 Inference: recruitment from a fixed learned state

\[
\boxed{
\begin{aligned}
S^{\mathrm{active}}_{x,c}&=\Pi_{x,c}(L^*),\\
Y_{x,c}&=H_{x,c}\bigl(S^{\mathrm{active}}_{x,c}\bigr),\\
L^*_{\mathrm{after}}&=L^*_{\mathrm{before}}.
\end{aligned}}
\tag{4}
\]

A frozen inference is an executed occurrence: it reads a particular learned version, recruits support conditioned on the query and context, and combines the resulting contributions. Recurrent computation, denoising steps, token generation, and other transient computations can evolve inside \(H_{x,c}\) while the persistent learned state \(L^*\) remains fixed.

In the experiments, causal recruitment is measured by component gates, non-additive joint gates, and exact substitution of earlier learned component versions.

### 2.4 Reinforcement: behavioral consequences re-enter training

\[
\boxed{
\begin{aligned}
Y_t&=\mathcal I(L_t,x_t,c_t),\\
B_{t+1}&=\mathcal B(B_t,Y_t,E_t),\\
A_{t+1}&=\mathcal U(Z_{t+1},D_{t+1},B_{t+1}),\\
Z_{t+2}&=\mathcal T(Z_{t+1},A_{t+1}).
\end{aligned}}
\tag{5}
\]

\(\mathcal I\) denotes the realized inference/behavioral process. The feedback \(B_{t+1}\) affects the next training action, which in turn reorganizes later learned states and their functional support. RL-E05 and RL-E06 test this loop by experimentally controlling the allocation, duration, and redirection of *correct positive feedback* in a shared four-capability GRU.

---

## 3. Experimentally established dynamical relations

### 3.1 Realized updates and receiving-state dependence — TL-E01–E03

TL-E01 measures formation, post-formation decline, and recovery using exact training-state records and sealed capability evaluations. Its cross-run experiments show why loss, accuracy, absolute step, or a single parameter summary cannot serve as a sufficient description of the learning trajectory. TL-E02 applies interventions that pause effective parameter–optimizer evolution and compares them with clipping controls; the pause shifts the observed capability-formation event.

TL-E03 isolates an actual update from the state receiving it. For a formed update \(\delta\) and a target-specific readout,

\[
\boxed{
R_q(\theta,\delta)
=m_q(\theta+\delta)-m_q(\theta),
}
\tag{6}
\]

with the experimentally observed receiving-state dependence

\[
\boxed{
\exists\,\theta_A,\theta_B,\delta,q:
\quad R_q(\theta_A,\delta)\ne R_q(\theta_B,\delta).
}
\tag{7}
\]

The reciprocal causal-fork protocols restore exact pre-states, execute full/skip and parameter-/optimizer-separated branches, transplant realized update increments, and compare branch-minus-skip functional responses across registered horizons. They distinguish the *formation of an update* from the *effect of applying that update to a particular receiving state*.

**Experimental anchors:** [TL-E01/02 theory ledger](experiments/gfg_nanogpt_autonomous_capability_discovery_v1/research_archive/THEORY_LEDGER.md); [TL-E02 intervention contract](experiments/gfg_nanogpt_autonomous_capability_discovery_v1/contracts/intervention_api.json); [TL-E03 reciprocal protocol](experiments/gfg_nanogpt_stepwise_support_transition_v1/RECIPROCAL_MATCHED_PAIR_PROTOCOL_V2.json); [independent reciprocal validator](experiments/gfg_nanogpt_stepwise_support_transition_v1/reciprocal_validator.py).

### 3.2 Finite-amplitude response and its conditioning factors — TL-E04–E05

The actual update is applied at controlled amplitudes to one unchanged receiving pre-state:

\[
\boxed{
r_{t,q}(\alpha)
=m_q(\theta_t+\alpha\delta_t)-m_q(\theta_t).
}
\tag{8}
\]

\(\alpha\) scales a **single recorded parameter increment**, not elapsed training time. The experiments preserve the response paths, including near-linear, saturating, accelerating, turnback, and sign-reversal morphologies. Finite-amplitude nonlinearity is directly observed:

\[
\boxed{
\exists\,t,q,\alpha:\quad
r_{t,q}(\alpha)\ne\alpha\,r_{t,q}(1).
}
\tag{9}
\]

TL-E05 then studies *why the response of the same identified target differs between real training histories*. Its frozen analysis separates seven prior-information blocks, matches identical target identities across runs, tests alternative factor orders and leave-one-block-out comparisons, and uses run-pair-clustered bootstrap uncertainty. The principal response-conditioning families are

\[
\boxed{
\mathcal F_{t,q}
=\bigl(F1_{t,q},F3_{t,q},F5_t\bigr).
}
\tag{10}
\]

- **F1 — target boundary:** the pre-update margin, competing outcomes, and proximity to a capability boundary;
- **F3 — target–update geometry:** the realized update's magnitude, component distribution, and directional relationship to the identified target;
- **F5 — receiving parameter/optimizer state:** the pre-update parameter configuration, optimizer memory, preconditioning summaries, and their interactions with the formed update.

The frozen TL-E05 analysis uses 12 development histories, 72 update sections, 15,264 identity-aligned target responses, and 2,000 clustered bootstrap replicates. \(\mathcal F_{t,q}\) records measured families of response-conditioning information; their operational components can be adapted to the native state of a different optimizer and architecture.

**Experimental anchors:** [amplitude-path protocol](experiments/gfg_nanogpt_stepwise_support_transition_v1/B_UPDATE_AMPLITUDE_PATH_PROTOCOL.json); [amplitude-path validator](experiments/gfg_nanogpt_stepwise_support_transition_v1/amplitude_path_validator.py); [TL-E05 frozen factor contract](experiments/gfg_nanogpt_response_factor_analysis_v1/FACTOR_ANALYSIS_CONTRACT.md); [factor-analysis implementation](experiments/gfg_nanogpt_response_factor_analysis_v1/runner.py).

### 3.3 Distributed support and component interaction — TL-E06

Fix a learned state \(L\), target \(q\), and component-intervention protocol \(\mathcal J_q\). Let \(f_q(L;\mathrm{full})\) be the continuous functional readout with all registered components active. Write \(f_q(L;-i)\) for the same state with component \(i\)'s output disabled, and \(f_q(L;-i,-j)\) for simultaneous removal of \(i\) and \(j\).

The exact measured two-component interaction is

\[
\boxed{
\begin{aligned}
I_{ij,q}(L)={}&f_q(L;\mathrm{full})
-f_q(L;-i)\\
&-f_q(L;-j)+f_q(L;-i,-j).
\end{aligned}}
\tag{11}
\]

\[
\boxed{\exists\,L,i,j,q:\quad I_{ij,q}(L)\ne0.}
\tag{12}
\]

A nonzero \(I_{ij,q}\) records a joint functional effect that differs from the sum of the separate gate effects. Its value and sign come from measured outputs.

For the nanoGPT CSRG-4C instrument, component necessity and pair backup are evaluated as

\[
\begin{aligned}
N_{j,q}&=\max\bigl(0,f_{\mathrm{full},q}-f_{-j,q}\bigr),\\
Q_{ij,q}&=\max\bigl(0,(f_{\mathrm{full},q}-f_{-i,-j,q})
-N_{i,q}-N_{j,q}\bigr).
\end{aligned}
\tag{13}
\]

When \(\sum_j N_{j,q}>0\), the normalized allocation and effective component support are

\[
w_{j,q}=\frac{N_{j,q}}{\sum_kN_{k,q}},
\qquad
n^{\mathrm{eff}}_q=\frac{1}{\sum_jw_{j,q}^{\,2}}.
\tag{14}
\]

The full measured profile \(\mathcal P_q(L;\mathcal J_q)\) also contains the registered single-/double-gate margins, backup/substitution effects, and failure slack. It is constructed from **actual intervention results**, with each absent or undefined statistic retained as such in the evidence record.

**Experimental anchors:** [CSRG-4C frozen capture contract](experiments/gfg_nanogpt_support_redundancy_v1/capture_contract_v2.json); [intervention and support builder](experiments/gfg_nanogpt_support_redundancy_v1/builder.py); [independent archive verifier](experiments/gfg_nanogpt_support_redundancy_v1/verify_archive.py).

### 3.4 Persistent redistribution and capability readout — TL-E07–E08

Measure the same registered support profile on both sides of a *single actual training transition*:

\[
\boxed{
\Delta P_{t,q}
=\mathcal P_q(L_{t+1};\mathcal J_q)
-\mathcal P_q(L_t;\mathcal J_q).
}
\tag{15}
\]

TL-E07 compares the identical receiving pre-state with \(\alpha=0\) (skipped parameter increment) and \(\alpha=1\) (the complete realized increment). Across 72 sections, it records 1,656 target-group transitions, 1,636 defined allocation comparisons, and **228 primary-support handoffs**. The measured allocation can reorganize while the observed target capability is unchanged:

\[
\boxed{
\exists\,t,q:\quad
\Delta P_{t,q}\ne0
\quad\land\quad
C_{t+1,q}=C_{t,q}.
}
\tag{16}
\]

For signed target margins, TL-E08 connects the actual response to the observable boundary:

\[
\boxed{
\begin{aligned}
m_q(L_{t+1})&=m_q(L_t)+R_{t,q},\\
C_{t,q}&=\mathbf1[m_q(L_t)>0],\\
C_{t+1,q}&=\mathbf1[m_q(L_t)+R_{t,q}>0].
\end{aligned}}
\tag{17}
\]

The four observed outcomes—remaining correct, correct-to-wrong, remaining wrong, and wrong-to-correct—are determined by the two boundary readouts. The underlying continuous margin records how an update can move a target toward or away from the boundary before the discrete outcome changes.

**Experimental anchors:** [TL-E07 analysis](experiments/gfg_nanogpt_support_reallocation_audit_v1/analysis.py); [TL-E07 independent checker](experiments/gfg_nanogpt_support_reallocation_audit_v1/independent.py); [TL-E08 identity-aligned experiment](experiments/gfg_nanogpt_identity_aligned_margin_crossing_v1/experiment.py).

### 3.5 Query-conditioned frozen inference — INF-E01

For a frozen learned state \(L^*\), inference executes Equation (4). A direct version-intervention measurement is

\[
\boxed{
D_{j,q}^{\mathrm{rollback}}
=f_q(L^*;\mathrm{full})
-f_q\bigl(L^{(j\leftarrow\mathrm{pre})};\mathrm{full}\bigr),
}
\tag{18}
\]

where \(L^{(j\leftarrow\mathrm{pre})}\) replaces only the identified component \(j\) with its exact pre-formation version while retaining all other registered trained components.

INF-E01 tests 13 nanoGPT histories at four mechanically selected checkpoint phases each: **52 phase-specific comparisons**. Frozen forward executions are paired with component gating, query-specific support profiles, pairwise interactions, exact earlier-version substitution, and byte-exact trained-version restoration. Thus the inference result is experimentally linked to the particular learned parameter versions recruited for that occurrence.

The same frozen-inference relations are tested in ResNet, diffusion U-Net, and routed Vision-MoE. In diffusion, complete denoising trajectories change transient sample state while the model's persistent learned state remains fixed.

**Experimental anchors:** [INF-E01 frozen protocol](experiments/gfg_nanogpt_training_learning_inference_projection_v1/PROTOCOL_FREEZE.md); [INF-E01 analysis](experiments/gfg_nanogpt_training_learning_inference_projection_v1/analysis.py); [cross-system frozen inference](experiments/gfg_cross_system_frozen_inference_projection_v1/RESULTS.md); [Vision-MoE inference results](experiments/gfg_moe_cifar_training_learning_inference_generalization_v1/RESULTS.md).

### 3.6 Feedback-directed support reorganization — RL-E05–E06

The feedback branch of Equation (5) is tested in a shared, finite, four-capability GRU after all four capabilities have reached mastery. Registered component-coalition interventions measure exact Shapley contributions to correct-action margins. Aggregating positive contributions across registered targets gives each capability's measured positive-support mass \(M_q^+\), and its **relative** support share

\[
\boxed{
s_q=\frac{M_q^+}{\sum_k M_k^+}.
}
\tag{19}
\]

RL-E05 compares selective, balanced, and no-persistent-update branches from byte-identical parameter–optimizer states. It records increased relative support concentration, reduced performance on unreinforced capabilities, and directionally informative component-version rollback interventions. Its stricter pre-registered temporal gate, requiring a specified 0.03 support-share increase by the first observed accuracy loss, was satisfied in **2/12** seeds.

RL-E06 directly varies the *distribution* of 60 correct-feedback episodes per update while holding the initial mastered state, episode budget, optimizer opportunities, and reward rule fixed. The four allocations across capabilities \(0,1,2,3\) are

\[
(15,15,15,15),\quad(30,10,10,10),\quad
(45,5,5,5),\quad(60,0,0,0).
\tag{20}
\]

At the registered dose levels \(d_1<d_2<d_3<d_4\), all 12 formal seeds exhibit the measured ordering

\[
\boxed{
\begin{aligned}
s_0(d_1)&<s_0(d_2)<s_0(d_3)<s_0(d_4),\\
\overline m_{\mathrm{other}}(d_1)
&>\overline m_{\mathrm{other}}(d_2)
>\overline m_{\mathrm{other}}(d_3)
>\overline m_{\mathrm{other}}(d_4).
\end{aligned}}
\tag{21}
\]

The mean reinforced-capability support shares are **25.54%, 26.47%, 27.50%, 35.20%**; the other-capability mean margins are **9.474, 9.130, 8.493, 3.962**.

At exclusive-feedback update 800, two recovery branches begin from the **same exact parameter–AdamW state**. Rebalanced feedback restores the other capabilities' mean accuracy from **70.31% to 99.48%** while retaining capability 0 at **100%**; training only the other three restores them to **99.83%** while capability 0 reaches **89.06%**. The associated functional-support profiles move toward the corresponding balanced configurations.

Together these interventions establish feedback allocation as a causal control of *persistent functional reorganization* in the tested shared system.

**Experimental anchors:** [RL-E05 protocol](experiments/gfg_rl_selective_positive_feedback_support_concentration_v1/PROTOCOL_FREEZE.md); [RL-E05 results](experiments/gfg_rl_selective_positive_feedback_support_concentration_v1/RESULTS.md); [RL-E06 frozen dose/recovery protocol](experiments/gfg_rl_selective_positive_feedback_dose_recovery_v1/PROTOCOL_FREEZE.md); [RL-E06 formal results](experiments/gfg_rl_selective_positive_feedback_dose_recovery_v1/RESULTS.md); [RL-E06 independent checker](experiments/gfg_rl_selective_positive_feedback_dose_recovery_v1/independent_checker.py).

---

## 4. Quantitative prediction from the established dynamics — TL-P01

The realized-update and target-boundary measurements yield a concrete prediction task: **after an update has been formed, predict its effect on an identified target's decision boundary before evaluating the updated network on that target**.

For a network with hidden representation \(h(\theta,x)\) and linear output rows \(w_k\), fix correct class \(y\) and competitor \(c\). At the receiving pre-state, set

\[
\begin{aligned}
b_c&=w_y-w_c,&
\Delta b_c&=\Delta w_y-\Delta w_c,\\
h&=h(\theta_t,x),&
h'&=Dh(\theta_t,x)[\delta_t],\\
&&h''&=D^2h(\theta_t,x)[\delta_t,\delta_t].
\end{aligned}
\tag{22}
\]

Here \(h'\) and \(h''\) are the first and second *directional responses of the pre-update hidden representation along the already formed actual update*. An affine output head can be written in the same form by augmenting \(h\) with a constant coordinate.

The frozen quadratic-complete endpoint-gap estimator is

\[
\boxed{
\begin{aligned}
\widehat g_{y,c}
={}&b_c^\top h
+\Delta b_c^\top h
+b_c^\top h'\\
&+\Delta b_c^\top h'
+\tfrac12\,b_c^\top h''.
\end{aligned}}
\tag{23}
\]

All competitors are evaluated separately; the predicted endpoint capability is

\[
\boxed{
\widehat C_{t+1,q}
=\mathbf1\!\left[\min_{c\ne y}\widehat g_{y,c}>0\right].
}
\tag{24}
\]

The predictor combines the **initial target boundary**, **the target-specific geometry of the actual update**, and **the receiving-state directional response**. It is evaluated at the pre-response information boundary: the actual update is available, while the updated target output is revealed only during adjudication.

In the nanoGPT four-run confirmation split, the fixed algorithm obtains **4,652/5,088 = 91.43%** post-update boundary accuracy and **91.49%** macro recall across the four target-transition categories. Its primary score is a target-boundary result rather than next-token task accuracy. The same algorithm is independently applied to all 15,264 target records.

**Experimental anchors:** [TL-P01 frozen protocol](experiments/gfg_nanogpt_actual_update_boundary_v1/PROTOCOL_FREEZE.md); [results](experiments/gfg_nanogpt_actual_update_boundary_v1/RESULTS.md); [independent checker](experiments/gfg_nanogpt_actual_update_boundary_v1/INDEPENDENT_CHECKER.py).

## 5. Cross-system experimental tests

The core dynamical relations were remeasured under different architectures, learning mechanisms, targets, and inference operations. The table reports the **single-update boundary predictor's** accuracy and four-transition macro recall, alongside the system's experimental role.

| System | Native test of the dynamics | Boundary-prediction accuracy | Four-transition macro recall |
|---|---|---:|---:|
| nanoGPT / Adam | Receiving-state exchanges, amplitude paths, component support, learned-version inference | **91.43%** | **91.49%** |
| ResNet-18 / CIFAR-100 / SGD momentum | Image-classification target boundaries, momentum exchanges, residual-stage coalitions | **99.28%** | **97.99%** |
| Diffusion U-Net / CIFAR-10 / AdamW | Identified image–timestep–noise targets, residual-error boundaries, denoising support | **99.60%** | **97.98%** |
| Vision-MoE / CIFAR-100 / AdamW | Top-2 routing, expert coalitions, routed support, exact rollback | **99.63%** | **96.95%** |
| Message-passing GNN / RMSprop | Graph-level targets, native functional support and update exchanges | **98.54%** | **98.62%** |
| Hard-spiking SNN / Adagrad | Exact spike events, hard-threshold forward dynamics, native support interventions | **95.31%** | **70.74%** |

The ResNet, diffusion, and Vision-MoE experiments independently reproduce the tested receiving-state response, nonlinear paths, and support reallocation under frozen protocols. The GNN retains one failed finite-amplitude frequency gate in its first seed (**8/9** gates passed there), while its other reported relation gates and the core intervention observations remain measured. The hard-spiking SNN passes **12/12** operational gates in each of four seeds; its lower boundary-transition recall identifies the distinct behavior of a local-derivative predictor when real hard-spike threshold crossings are involved.

This separation between **underlying functional dynamics** and **a particular approximate endpoint predictor** is itself experimentally observable.

**Experimental anchors:** [ResNet formal results](experiments/gfg_resnet_cifar_training_learning_generalization_v1/RESULTS.md); [diffusion formal results](experiments/gfg_ddpm_cifar_training_learning_generalization_v1/RESULTS.md); [Vision-MoE formal results](experiments/gfg_moe_cifar_training_learning_inference_generalization_v1/RESULTS.md); [GNN/SNN prediction transport](independent_stress_tests/tlp01_prediction/REPORT.md); [cross-system evidence scope](independent_stress_tests/EVIDENCE_SCOPE.md).

## 6. Equation-to-evidence map

| Dynamical object or relationship | Equations | Direct experimental basis |
|---|---|---|
| Realized training actions, persistent receiving states, optimizer memory | (1)–(2) | TL-E01–E03; TL-G01–G03; GNN/SNN |
| State-dependent response to the same formed update | (6)–(7) | TL-E03 and cross-system reciprocal update exchanges |
| Actual finite-amplitude nonlinear response | (8)–(9) | TL-E04; TL-G01–G03; GNN/SNN |
| Response-conditioning factor families F1/F3/F5 | (10) | TL-E05; run-isolated cross-system factor tests |
| Distributed support and non-additive component effects | (11)–(14) | TL-E06; native coalition experiments in other systems |
| Persistent support redistribution under actual updates | (3), (15)–(16) | TL-E07; cross-system before/after support interventions |
| Target capability and boundary crossings | (3), (17) | TL-E08 and identity-aligned true endpoint records |
| Query-conditioned frozen inference and learned-version dependence | (4), (18) | INF-E01; INF-G01; INF-G02; GNN/SNN |
| Feedback-directed concentration and recovery | (5), (19)–(21) | RL-E05–E06 |
| Quantitative prediction of an actual update's target boundary | (22)–(24) | TL-P01 and cross-system prediction tests |

## 7. Unified dynamical interpretation and source navigation

The experimentally grounded relation can be summarized as

\[
\boxed{
\begin{gathered}
\text{receiving state}+\text{realized training action}\\
\Downarrow\\
\text{one persistent-state transition}\\
\Downarrow\\
\left\{
\begin{array}{l}
\text{target-specific functional response}\\
\text{distributed functional-support reorganization}\\
\text{capability-boundary outcome}
\end{array}
\right.\\
\Downarrow\\
\text{query-conditioned use of the learned state}\\
\Downarrow\\
\text{behavioral consequence and subsequent feedback}\\
\Downarrow\\
\text{next realized training action}.
\end{gathered}}
\tag{25}
\]

**TLIR unifies training, learning, inference, and reinforcement through their experimentally resolved relationships.** A realized training action acts on a concrete receiving state; its finite functional response depends on that state; the transition reorganizes distributed functional support and changes target-specific readout boundaries; inference recruits support from the learned version in use; and feedback changes the future distribution of training actions, hence the subsequent learned functional organization.

### Repository evidence and manuscript

- [Final Science manuscript evidence map](TLIR_PAPER_EVIDENCE_MAP.md)
- [Primary experiment index](EXPERIMENT_INDEX.md)
- [Experimental instruments and native implementations](EXPERIMENTAL_INSTRUMENTS.md)
- [Public evidence and reproduction matrix](PUBLIC_EVIDENCE_MATRIX.md)
- [Historical positive/negative theory comparisons and derivations](TLIR_SECTION5_21_PAIRS_14_EXTRAPOLATIONS.md)
- [Publication evidence archive](https://doi.org/10.5281/zenodo.22005307)

## 8. A Simple Theory

The basic logic of TLIR can be seen directly from how a neural network computes and how its parameters are updated.

### 8.1 How a neural network works

An artificial neural network is a connected system of numerical computations. Each neuron or computational component receives numerical signals, transforms them using operations such as multiplication, addition and nonlinear activation, and passes the result onward. The parameters determine the numerical behavior of these computations. Layers and components connect them into a complete function, written as \(f_\theta(x)\), where \(x\) is an input and \(\theta\) denotes the current model parameters.

The same network processes many different inputs. Its neurons, parameter matrices and components are not generally reserved for just one training example or one capability. Different inputs can therefore use overlapping parts of the same computational system.

Training proceeds just as simply. The network makes a prediction; a loss measures its disagreement with the training target; backpropagation applies the chain rule to compute how that loss depends on the parameters; and an optimizer executes a parameter update. For elementary gradient descent on target \(A\),

\[
\boxed{
\theta_{t+1}
=\theta_t-\eta\nabla_\theta\ell_A(\theta_t).
}
\tag{26}
\]

Other optimizers, such as Adam, also use stored optimization history to form the realized update. Backpropagation calculates gradients; the optimizer turns them into an actual change of state.

### 8.2 The elementary deduction

**First: updating one target can affect other targets.** Training on \(A\) changes parameters used by the *whole* network, not merely by \(A\). For any other input or target \(q\), including \(B\) and \(C\), the resulting change is simply

\[
\boxed{
\Delta f_q
=f_{\theta_t+\delta_A}(x_q)-f_{\theta_t}(x_q).
}
\tag{27}
\]

This change can be positive, negative or zero. The update was formed from \(A\), but its effects are determined by how the shared network computes each target.

A two-parameter example makes the point without advanced mathematics. Suppose three target scores produced by one network are

\[
f_A=a+b,\qquad f_B=a,\qquad f_C=1-b.
\]

Initially \(a=0.2\), \(b=0.1\). Train **only A** toward the target value \(1.3\) using \(\ell_A=\tfrac12(f_A-1.3)^2\) and one gradient-descent step with \(\eta=0.2\). The gradients are both \(-1\), so the optimizer updates \(a\) to \(0.4\) and \(b\) to \(0.3\).

| Target score | Before training A | After training A | Change |
|---|---:|---:|---:|
| \(f_A=a+b\) | 0.3 | 0.7 | +0.4 |
| \(f_B=a\) | 0.2 | 0.4 | +0.2 |
| \(f_C=1-b\) | 0.9 | 0.7 | -0.2 |

No training instruction mentioned B or C. Yet B increased and C decreased because they used the parameters changed by training A. This is the simplest origin of cross-target functional transfer and interference.

**Second: functional support can be distributed, combined and reorganized.** A target's output can depend on more than one neuron or component. In the additive example \(f_A=a+b\), both paths contribute; their relative contributions change as the parameters change. In a different elementary computation, \(f_q=ab\), the two paths interact multiplicatively. Removing either path changes what the other path can contribute. Thus shared computation naturally permits distributed contributions, replacement or backup, and non-additive joint effects. Their actual causal roles can be measured by disabling components separately and together, rather than inferred from parameter size.

Because training changes the parameters inside these same computational paths, it can also change **which components support a target, by how much, and in what combinations**. No rewiring of the network's program is necessary: the functional organization changes through ordinary parameter updates. This is the elementary connection between training and persistent learning.

**Third: a gradual functional change can appear as a sudden new ability.** If a target's correct-answer score initially lies below a competitor's score, successive updates may move the difference toward and eventually across zero:

\[
\boxed{
m_q(\theta_t)<0,\qquad
m_q(\theta_{t+1})>0
\quad\Longrightarrow\quad
C_{t,q}=0,\ C_{t+1,q}=1.
}
\tag{28}
\]

The continuous readout changes numerically; the correct/incorrect judgment changes discretely. A capability transition therefore need not coincide with a sudden creation of its entire internal support.

**Fourth: inference uses the organization already formed.** After training, hold the model parameters and other persistent learned variables fixed. Input \(x_A\) and input \(x_B\) can still produce different activations, computational paths and causal component contributions:

\[
\boxed{
Y_A=f_{\theta^*}(x_A),\qquad
Y_B=f_{\theta^*}(x_B),
\qquad \theta^*\ \text{unchanged}.
}
\tag{29}
\]

The model is using its existing learned functional organization, not performing another persistent parameter update. Attention, convolution, graph message passing, spiking and expert routing are different ways of executing computations inside this more general relation; the relation itself does not require Attention.

**Fifth: reinforcement closes the same loop.** When the network's output affects an environment or receives a reward, that consequence can become the information used to form a later training action. The new action changes shared parameters, their functional organization and subsequent behavior:

\[
\boxed{
\text{current function}
\ \longrightarrow\ \text{behavior and feedback}
\ \longrightarrow\ \text{actual update}
\ \longrightarrow\ \text{changed function}.
}
\tag{30}
\]

### 8.3 The simple conclusion

The reasoning begins with ordinary facts: neurons and components compute together; parameters are shared across many inputs; backpropagation calculates gradients; optimizers update those parameters; and the same updated network subsequently processes old and new inputs.

From these facts, the core TLIR relationships follow naturally: an actual update has target- and receiving-state-dependent functional consequences; training can reorganize distributed functional support; observed capability depends on the target's readout boundary; inference makes input-dependent use of the learned state without changing it; and feedback can drive another training transition.

**The theory is simple because its starting mechanisms are simple.** The experimental program measures the particular responses, support interactions, reorganizations, boundary transitions and feedback effects realized by actual neural networks.
