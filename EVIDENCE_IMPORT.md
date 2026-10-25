# Evidence import and preservation

## Sources

This repository is an owner-authorized copy and evidence-navigation layer over the existing public [research-evidence repository](https://github.com/wind342/gfg-training-learning-inference-experiments). Nothing is moved out of or deleted from that repository.

| Imported scope | Immutable source |
|---|---|
| Complete public main snapshot, including GFG Core, all original experiments, ResNet, diffusion, **MoE**, GRU feedback, existing documentation and public v8 manuscript assets | commit `766715a9728cce4e949714969f118fc7f16a0b41` |
| Complete `independent_stress_tests/` subtree, including GNN, SNN, native SNN binary parts and prediction-transport reports/code | commit `a35d143ea4952c48fce425161acd8f3ead408317` |
| Original publication archive | [Zenodo 22032772](https://doi.org/10.5281/zenodo.22032772), 10 original ZIPs and original archive metadata |
| Original frozen release | annotated tag `paper-experiments-cross-system-feedback-release`, commit `36dab5ce347dbbdac157ef23205f556606d18294` |

The source main history is retained. The annotated tag object is copied unchanged, not moved or recreated. The new combined repository has a new tip/tree identity: it is **not** the old frozen release. Historical manifests continue to identify their historical scope.

All imported scientific files retain their Git blob identity. Only the root README is extended with a new entrypoint; its original bytes remain in source history. No original protocol, result, validator, experimental code, manifest, licence notice or manuscript asset is edited. The GNN/SNN subtree is imported exactly, not merged into or relabelled as a Core-v3 experiment.

## Public archive mirror

The [GitHub Release mirror](https://github.com/wind342/tlir-research-evidence/releases/tag/zenodo-22032772-mirror) contains the ten original ZIPs and eight original metadata files from Zenodo. ZIPs are not recompressed or inserted into normal Git history. Zenodo remains the archival authority; the mirror does not create a new scientific version. Compare every payload with the original `ARCHIVE_MANIFEST.json`.

The local import check also compares these bytes with the public Zenodo API checksums before publication. The separate public manuscript PDF attached to Zenodo is not duplicated in the evidence mirror.

## Explicit availability limits

1. **MoE:** the complete 20-file public experiment directory is included. Its `FORMAL_RESULTS.json`, historical independent-check receipt and 34-entry runtime manifest are present. The three raw final checkpoints, per-event records and compact GFG runtime files listed by that manifest were not in the source Git snapshot, were not in Zenodo 22032772, and were not located in the inspected local evidence locations. No reconstruction or claimed fresh replay is substituted for them.
2. **GNN/SNN prediction transport:** all five public files are included. The checker additionally requires `PREPARED_MANIFEST.json`, `PREDICTIONS_SEALED.json`, `PREDICTIONS_SEAL_RECEIPT.json`, `ADJUDICATION.json` and matching native/pretarget states; those are not in that public subtree. The report's historical audit claim is not a newly executed audit here.
3. **SNN structural tests:** all four distinct native states are present as 16 binary fragments. Their concatenated SHA-256 identities can be checked without deserializing PyTorch files. The existing restoration script reconstructs formal/replay copies.
4. **nanoGPT / original release:** public archives and compact records retain the scope documented in `PUBLIC_EVIDENCE_MATRIX.md`; not every historical training checkpoint or full CSRG runtime is included.
5. **New paper figures and historical literature comparisons:** the supplied confidential three-figure review PDF and private plotting project are not uploaded. The existing public v8 manuscript assets are preserved but are not this PDF. No complete separate 21-pair literature ledger was found in the imported snapshots.

These are availability statements, not modifications to scientific outcomes. GNN's failed nonlinear gate, SNN's negative prediction-transport verdict and RL-E05's negative composite verdict remain unchanged.

## Recheck this import

From a Git checkout with history:

```bash
python tools/verify_tlir_evidence_import.py
python tools/verify_tlir_evidence_import.py --archive-dir /path/to/downloaded/release/assets
```

This checks source object identity, copied files, documentation links, SNN binary assembly hashes, retained negative verdicts and (when supplied) archive hashes. It does not train a model, load PyTorch checkpoints or rerun the native scientific experiments. The current import receipt is [EVIDENCE_IMPORT_CHECKS.json](EVIDENCE_IMPORT_CHECKS.json).

## Rights and privacy

The original [proprietary notice](PROPRIETARY_NOTICE.md) and third-party authority notices remain in force. No new open-source licence is applied. This copy was requested by the repository owner; public access does not grant third parties unrestricted redistribution rights. No newly supplied confidential manuscript, local user path, credential or private working directory is added.

