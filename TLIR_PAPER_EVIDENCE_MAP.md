# TLIR paper: section, figure and evidence map

This navigation follows **Unified Training–Learning–Inference–Reinforcement Dynamics Across Neural Systems**, the supplied eight-page, three-figure version. It does not replace the protocols, redefine scientific quantities or turn archival checks into new experiments. The confidential review PDF is not redistributed here.

Read in discovery order: **nine nanoGPT TLI experiments → TL-E05-derived prediction → five additional TLI systems → GRU feedback interventions → cross-field interpretation**. Six systems test TLI; GRU is a seventh system for feedback closure, not an additional complete TLI chain.

## 1. Introduction

The introduction motivates the common dynamical relationship. Its experimental authorities are the following sections, not additional experiment IDs. GFG foundations and four instruments remain available under [EXPERIMENT_INDEX.md](EXPERIMENT_INDEX.md) and [EXPERIMENTAL_INSTRUMENTS.md](EXPERIMENTAL_INSTRUMENTS.md).

## 2. Experimental establishment of TLI open-loop dynamics

### 2.1 Training: realized updates and receiving states

| Experiment | Question / evidence | Protocol and execution | Results / checks |
|---|---|---|---|
| TL-E01 | Insufficiency of familiar observables; formation, decline and recovery | [Contracts](experiments/gfg_nanogpt_autonomous_capability_discovery_v1/contracts/) | [Theory ledger](experiments/gfg_nanogpt_autonomous_capability_discovery_v1/research_archive/THEORY_LEDGER.md), [public summary](experiments/gfg_nanogpt_autonomous_capability_discovery_v1/reports/public_summary.json) |
| TL-E02 | Pausing parameter–optimizer evolution delays formation; clipping control | [Intervention contract](experiments/gfg_nanogpt_autonomous_capability_discovery_v1/contracts/intervention_api.json), [checkpoint forks](experiments/gfg_nanogpt_autonomous_capability_discovery_v1/checkpoint_fork.py) | [Archived executions and causal evaluations](experiments/gfg_nanogpt_autonomous_capability_discovery_v1/research_archive/entries/) |
| TL-E03 | Same formed update, different receiving states; separate optimizer-memory exchanges | [Reciprocal protocol](experiments/gfg_nanogpt_stepwise_support_transition_v1/RECIPROCAL_MATCHED_PAIR_PROTOCOL_V2.json), [P2 local response](experiments/gfg_nanogpt_stepwise_support_transition_v1/P2_RECIPROCAL_LOCAL_RESPONSE_PROTOCOL.json), [optimizer exchange](experiments/gfg_nanogpt_stepwise_support_transition_v1/H20_RECIPROCAL_OPTIMIZER_EXCHANGE_PROTOCOL.json) | [Reciprocal validator](experiments/gfg_nanogpt_stepwise_support_transition_v1/reciprocal_validator.py), [independent replay](experiments/gfg_nanogpt_stepwise_support_transition_v1/independent_replay_validator.py) |
| TL-E04 | Finite-amplitude response paths, including nonlinear morphologies | [Amplitude-path protocol](experiments/gfg_nanogpt_stepwise_support_transition_v1/B_UPDATE_AMPLITUDE_PATH_PROTOCOL.json) | [Path validator](experiments/gfg_nanogpt_stepwise_support_transition_v1/amplitude_path_validator.py), [GFG validator](experiments/gfg_nanogpt_stepwise_support_transition_v1/amplitude_path_gfg_validator.py) |
| TL-E05 | Conditioning by target boundary F1, target-update geometry F3, receiving state F5 | [Contract](experiments/gfg_nanogpt_response_factor_analysis_v1/FACTOR_ANALYSIS_CONTRACT.md), [analysis](experiments/gfg_nanogpt_response_factor_analysis_v1/analysis.py) | [Validator](experiments/gfg_nanogpt_response_factor_analysis_v1/validator.py); preserve the disclosed metadata-access boundary in the [public evidence matrix](PUBLIC_EVIDENCE_MATRIX.md) |

### 2.2 Learning: distributed support and capability boundaries

| Experiment | Question / evidence | Protocol / code / checker |
|---|---|---|
| TL-E06 | Distributed necessity, backup, substitution, synergy and failure slack | [CSRG-4C experiment](experiments/gfg_nanogpt_support_redundancy_v1/README.md), [capture contract](experiments/gfg_nanogpt_support_redundancy_v1/capture_contract_v2.json), [archive verifier](experiments/gfg_nanogpt_support_redundancy_v1/verify_archive.py) |
| TL-E07 | Actual updates reorganize support: 72 sections, 1,656 transitions, 1,636 defined allocations, 228 handoffs | [Scope](experiments/gfg_nanogpt_support_reallocation_audit_v1/ANALYSIS_SCOPE.md), [analysis](experiments/gfg_nanogpt_support_reallocation_audit_v1/analysis.py), [checker](experiments/gfg_nanogpt_support_reallocation_audit_v1/INDEPENDENT_CHECKER.py) |
| TL-E08 | Identity-aligned measured margins cross target readout boundaries | [Ledger construction](experiments/gfg_nanogpt_identity_aligned_margin_crossing_v1/experiment.py), [15,264-row ledger checking](experiments/gfg_nanogpt_actual_update_boundary_v1/INDEPENDENT_CHECKER.py) |

Undefined allocations are not zero support. TL-E08's measured endpoint facts are distinct from TL-P01's predictions, even when the archived ledger supports both analyses.

### 2.3 Inference: frozen recruitment of learned support

**INF-E01:** 13 nanoGPT histories, 52 historical checkpoint comparisons; target/query-conditioned recruitment, non-additive combinations, exact learned-version rollback and restoration.

- [Frozen protocol](experiments/gfg_nanogpt_training_learning_inference_projection_v1/PROTOCOL_FREEZE.md)
- [Analysis](experiments/gfg_nanogpt_training_learning_inference_projection_v1/analysis.py)
- [Independent checker](experiments/gfg_nanogpt_training_learning_inference_projection_v1/independent.py), [strict audit](experiments/gfg_nanogpt_training_learning_inference_projection_v1/strict_audit.py)
- [Public evidence boundary](experiments/gfg_nanogpt_training_learning_inference_projection_v1/PUBLIC_EVIDENCE.md)

## 3. Prediction and cross-system validation

### 3.1 Prediction derived from TL-E05

**TL-P01:** [protocol](experiments/gfg_nanogpt_actual_update_boundary_v1/PROTOCOL_FREEZE.md), [results](experiments/gfg_nanogpt_actual_update_boundary_v1/RESULTS.md), [analysis](experiments/gfg_nanogpt_actual_update_boundary_v1/analysis.py), [independent checker](experiments/gfg_nanogpt_actual_update_boundary_v1/INDEPENDENT_CHECKER.py).

Confirmation: **4,652/5,088 = 91.43% accuracy; 91.49% four-transition macro recall**. The 92.17% confirmation balanced accuracy is a different metric. The all-run total is 14,069/15,264 with 91.30% four-transition macro recall.

The direct predictor uses the already-formed update and pre-response internal quantities. It is not a trained probe or KNN. Separate coordinate-transport ablations in cross-system experiments do use KNN; do not conflate them with the direct predictor or Table 2.

### 3.2 Six TLI systems

| System | Identity | Protocol / code | Results / check |
|---|---|---|---|
| nanoGPT / Adam | TL-E01–08, INF-E01, TL-P01 | Sections 2 and 3.1 above | Original frozen release and Zenodo archive |
| ResNet-18 / CIFAR-100 / SGD momentum | TL-G01; inference also INF-G01 | [Protocol](experiments/gfg_resnet_cifar_training_learning_generalization_v1/PROTOCOL_FREEZE.md), [implementation](experiments/gfg_resnet_cifar_training_learning_generalization_v1/) | [Results](experiments/gfg_resnet_cifar_training_learning_generalization_v1/RESULTS.md), [checker](experiments/gfg_resnet_cifar_training_learning_generalization_v1/INDEPENDENT_CHECKER.py) |
| Diffusion U-Net / CIFAR-10 / AdamW | TL-G02; inference also INF-G01 | [Protocol](experiments/gfg_ddpm_cifar_training_learning_generalization_v1/PROTOCOL_FREEZE.md), [implementation](experiments/gfg_ddpm_cifar_training_learning_generalization_v1/) | [Results](experiments/gfg_ddpm_cifar_training_learning_generalization_v1/RESULTS.md), [checker](experiments/gfg_ddpm_cifar_training_learning_generalization_v1/INDEPENDENT_CHECKER.py) |
| Vision-MoE / CIFAR-100 / AdamW | **TL-G03 / INF-G02** | [Protocol](experiments/gfg_moe_cifar_training_learning_inference_generalization_v1/PROTOCOL_FREEZE.md), [model contract](experiments/gfg_moe_cifar_training_learning_inference_generalization_v1/MODEL_CONTRACT.json), [model](experiments/gfg_moe_cifar_training_learning_inference_generalization_v1/model.py), [runner](experiments/gfg_moe_cifar_training_learning_inference_generalization_v1/runner.py) | [Results](experiments/gfg_moe_cifar_training_learning_inference_generalization_v1/RESULTS.md), [machine results](experiments/gfg_moe_cifar_training_learning_inference_generalization_v1/FORMAL_RESULTS.json), [checker](experiments/gfg_moe_cifar_training_learning_inference_generalization_v1/INDEPENDENT_CHECKER.py), [historical check receipt](experiments/gfg_moe_cifar_training_learning_inference_generalization_v1/INDEPENDENT_CHECK_RESULT.json) |
| GNN / graph reachability / RMSprop | Independent stress test; no invented TL-G ID | [Frozen protocol](independent_stress_tests/gnn/gnn_falsification_frozen.json), [native code](independent_stress_tests/gnn/gnn_tlir_counterexample.py) | [Formal results](independent_stress_tests/gnn/gnn_tlir_formal_results.json), [replay](independent_stress_tests/gnn/gnn_tlir_formal_replay.json), [checker](independent_stress_tests/gnn/gnn_independent_checker.py), [audit](independent_stress_tests/gnn/gnn_falsification_audit.json) |
| Hard-spiking SNN / digits / Adagrad | Independent stress test; no invented TL-G ID | [Frozen protocol](independent_stress_tests/snn/snn_tlir_frozen_protocol.json), [native code](independent_stress_tests/snn/snn_tlir_adversarial.py) | [Formal summary](independent_stress_tests/snn/formal/FORMAL_SUMMARY.json), [checker](independent_stress_tests/snn/snn_tlir_independent_checker.py), [native-state restoration](independent_stress_tests/snn/native_parts/README.md) |

ResNet and diffusion inference are separately documented in **INF-G01**: [protocol](experiments/gfg_cross_system_frozen_inference_projection_v1/PROTOCOL_FREEZE.md), [results](experiments/gfg_cross_system_frozen_inference_projection_v1/RESULTS.md), [checker](experiments/gfg_cross_system_frozen_inference_projection_v1/INDEPENDENT_CHECKER.py).

**MoE is included in full as publicly committed**, not merely linked externally. All three formal seeds (20260910–12) passed the frozen joint verdict. It tests deterministic top-2 selection among four experts; all expert branches execute, with zero weight outside the selection. It is not a sparse-kernel speed benchmark. See the [scientific assessment](experiments/gfg_moe_cifar_training_learning_inference_generalization_v1/SCIENTIFIC_ASSESSMENT.md) and [runtime manifest](experiments/gfg_moe_cifar_training_learning_inference_generalization_v1/FORMAL_RUNTIME_MANIFEST.json). That manifest identifies runtime files absent from the public Git snapshot; the historical PASS receipt is not a fresh replay in this repository.

GNN preserves its failed nonlinear-frequency gate for seed 20261024 (4/256, 1.56%); its other eight gates passed. SNN preserves 12/12 operational gates in each of four seeds. Both stress tests explicitly **do not run GFG Core v3 or the original four-instrument implementation**; see [their evidence scope](independent_stress_tests/EVIDENCE_SCOPE.md).

### Table 2: boundary prediction, not task accuracy

| System | Accuracy (%) | Four-way macro recall (%) | Source |
|---|---:|---:|---|
| nanoGPT | 91.43 | 91.49 | TL-P01 confirmation |
| ResNet | 99.28 | 97.99 | TL-G01 direct second-order predictor |
| Diffusion | 99.60 | 97.98 | TL-G02 direct second-order predictor |
| MoE | 99.63 | 96.95 | TL-G03 direct second-order predictor |
| GNN | 98.54 | 98.62 | Independent prediction transport |
| Hard-spiking SNN | 95.31 | 70.74 | Independent prediction transport; frozen transport verdict NOT_SUPPORTED |

GNN/SNN prediction authorities: [protocol](independent_stress_tests/tlp01_prediction/FROZEN_PROTOCOL.json), [code](independent_stress_tests/tlp01_prediction/tlp01_cross_system.py), [report](independent_stress_tests/tlp01_prediction/REPORT.md), [machine metrics](independent_stress_tests/tlp01_prediction/ADJUDICATION_REPORT.json), [checker](independent_stress_tests/tlp01_prediction/independent_checker.py). The public subtree does not contain the full sealed target ledger or all pretarget/native endpoint files required by that checker. Do not describe this import as a fresh 10,272-target recomputation.

## 4. Reinforcement: feedback closure in the shared GRU

| Experiment | Scope | Protocol | Results / checker |
|---|---|---|---|
| RL-E05 | Selective correct feedback, support concentration, cross-skill interference, component-version rollback | [Protocol](experiments/gfg_rl_selective_positive_feedback_support_concentration_v1/PROTOCOL_FREEZE.md) | [Results](experiments/gfg_rl_selective_positive_feedback_support_concentration_v1/RESULTS.md), [assessment](experiments/gfg_rl_selective_positive_feedback_support_concentration_v1/SCIENTIFIC_ASSESSMENT.md), [checker](experiments/gfg_rl_selective_positive_feedback_support_concentration_v1/independent_checker.py) |
| RL-E06 | Fixed dose/duration and matched update-800 recovery forks | [Protocol](experiments/gfg_rl_selective_positive_feedback_dose_recovery_v1/PROTOCOL_FREEZE.md) | [Results](experiments/gfg_rl_selective_positive_feedback_dose_recovery_v1/RESULTS.md), [checker](experiments/gfg_rl_selective_positive_feedback_dose_recovery_v1/independent_checker.py) |

RL-E05's stronger temporal criterion remains **NOT_SUPPORTED** (2/12 seeds met the timing gate; 10/11 composite gates passed). Its observed intervention trade-offs remain evidence, not a relabelled pass.

RL-E06 retains all 12 seeds. The update-800 other-skill accuracy is 70.31%; rebalanced recovery reaches 99.48%, preserving skill 0 at 100%. Others-only repair reaches 99.83% but skill 0 falls to 89.06%.

**Table 3** is the fixed 60-episode allocation: balanced 15/15/15/15, mild 30/10/10/10, high 45/5/5/5, exclusive 60/0/0/0. It is not a new experiment.

## 5–6. Cross-field explanations and conclusion

The 21 historical positive/negative comparisons discussed in Section 5 are literature interpretations after experimental discovery, **not 21 new native experiments**. No separate complete 21-pair machine-readable comparison ledger was identified in the two imported public source snapshots. Do not manufacture experimental IDs or a checker for it. The conclusion synthesizes the linked evidence rather than introducing another test.

## Figures: exact panel mapping

**Figure 1** is a theory schematic, not additional measurements. **Figure 3** summarizes the six within-system TLI chains in Section 3.2, retaining the GNN partial gate outcome. **Table 1** is the system/optimizer/component inventory; it is not a performance comparison.

**Figure 2 contains 14 plots and two colorbars, not 16 independent experiments.**

| Panel | Evidence identity | What the plotted measurement means |
|---|---|---|
| a1 | TL-E02 | 11 pause pairs plus one clipping control; paired formation steps, not trajectories |
| a2 | TL-E03 P2 | Receiver-conditioned q10 response to recorded updates at alpha=0.125; not alpha=1 or a same-query cross-run comparison |
| a3 | TL-E04 | Five illustrative measured path morphologies, each normalized by its own maximum absolute response; not class frequencies |
| a4 | TL-E05 | Stated F1/F3/F5 conditioning contrasts and 95% intervals; not TL-P01 prediction accuracy |
| b1 | TL-E06 measurements retained in TL-E07 | Example single/pair gates: group-q10 failure boundary; not correctness of every individual |
| b2 | TL-E07 | A measured before/after support handoff; not an average or conserved physical resource |
| b3 | TL-E08 | All 15,264 measured pre/post margins, not predicted endpoints |
| c1 | INF-E01 | 23 target-group profiles: q10(full) minus q10(single gate) |
| c2 | INF-E01 | q10(full) minus both single-gate q10s plus joint-gate q10; zero means additive for this readout |
| c3 | INF-E01 | 52 historical-component rollback/restoration comparisons; exact restoration |
| d1–d2 | RL-E06 | Feedback dose, relative support share and other-skill margins across 12 seeds |
| d3 | RL-E05 | All 168 proper-subset rollbacks (14 × 12); not the failed temporal criterion |
| d4 | RL-E06 | Common update-800 state, continued exclusive feedback versus two recovery allocations |

The new Figure 2 plotting bundle is not part of the imported public snapshots. This map identifies its experiment authorities; it does not claim to redistribute private plotting sources or all historical trajectories. The already-public [v8 figure sources](manuscripts/tlir_v8/README.md) are an **older version** and must not be substituted for this three-figure PDF.

## Availability and verification

- [Import provenance, limits and Release mirror](EVIDENCE_IMPORT.md)
- [Historical public verification boundaries](PUBLIC_EVIDENCE_MATRIX.md)
- [Historical archive authority](PUBLIC_ARCHIVE.md)
- [Import checker](tools/verify_tlir_evidence_import.py): verifies copied objects and archive identities, **not scientific native reruns**.

