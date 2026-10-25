"""Verify the evidence copy, not the scientific experiments. Standard library only."""
from __future__ import annotations

import argparse
import hashlib
import json
import re
import subprocess
from pathlib import Path
from urllib.parse import unquote

ROOT = Path(__file__).resolve().parents[1]


def git(*args: str) -> bytes:
    return subprocess.check_output(['git', *args], cwd=ROOT)


def source_files(commit: str, prefix: str = '') -> dict[str, str]:
    entries = git('ls-tree', '-r', '-z', commit, '--', prefix or '.').split(b'\0')
    return {item.split(b'\t', 1)[1].decode('utf-8'): item.split(b'\t', 1)[0].split()[2].decode()
            for item in entries if item}


def mismatches(expected: dict[str, str], actual: dict[str, str]) -> list[str]:
    return [name for name, blob in expected.items() if actual.get(name) != blob]


def digest(path: Path) -> str:
    h = hashlib.sha256()
    with path.open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def check(archive_dir: Path | None = None) -> dict:
    manifest = json.loads((ROOT / 'EVIDENCE_IMPORT_MANIFEST.json').read_text(encoding='utf-8'))
    errors = []
    def require(condition, message):
        if not condition:
            errors.append(message)

    index = {}
    for row in git('ls-files', '--stage', '-z').split(b'\0'):
        if row:
            meta, name = row.split(b'\t', 1)
            mode, blob, stage = meta.split()
            require(stage == b'0', 'UNMERGED_INDEX')
            index[name.decode('utf-8')] = blob.decode()
    original = source_files(manifest['source_main_commit'])
    require(len(original) == manifest['source_main_file_count'], 'SOURCE_COUNT')
    require(git('rev-parse', manifest['source_main_commit'] + '^{tree}').decode().strip()
            == manifest['source_main_tree'], 'SOURCE_TREE')
    preserved = {p: h for p, h in original.items() if p not in manifest['allowed_source_file_changes']}
    errors.extend('SOURCE_CHANGED:' + p for p in mismatches(preserved, index))

    # The imported subtree itself is the immutable authority; no remote access required.
    expected_tree = manifest['stress_tree']
    expected = source_files(expected_tree)
    expected = {manifest['stress_prefix'] + '/' + p: h for p, h in expected.items()}
    require(len(expected) == manifest['stress_file_count'], 'STRESS_COUNT')
    actual_stress = {p: h for p, h in index.items() if p.startswith(manifest['stress_prefix'] + '/')}
    require(actual_stress == expected, 'STRESS_TREE_CHANGED')

    # Check unstaged changes as well as the staged/committed Git objects.
    diff = git('diff', '--name-only', '-z').decode('utf-8').split('\0')
    require(not (set(diff) & (set(preserved) | set(expected))), 'UNSTAGED_EVIDENCE_CHANGES')
    tag = manifest['frozen_tag']
    require(git('rev-parse', tag).decode().strip() == manifest['frozen_tag_object'], 'TAG_OBJECT_CHANGED')
    require(git('rev-parse', tag + '^{}').decode().strip() == manifest['frozen_commit'], 'FROZEN_COMMIT_CHANGED')

    checked_links = 0
    for name in ('README.md', 'TLIR_PAPER_EVIDENCE_MAP.md', 'EVIDENCE_IMPORT.md'):
        content = (ROOT / name).read_text(encoding='utf-8')
        for link in re.findall(r'\]\(([^)]+)\)', content):
            if '://' in link or link.startswith('#'):
                continue
            target = unquote(link.split('#', 1)[0])
            require((ROOT / target).exists(), 'BROKEN_LINK:' + name + ':' + target)
            checked_links += 1

    # No torch.load: reconstruct hashes in memory from existing public binary parts.
    snn = ROOT / 'independent_stress_tests/snn'
    parts = json.loads((snn / 'native_parts/MANIFEST.json').read_text())
    native_hashes = {}
    for seed, info in parts['seeds'].items():
        h = hashlib.sha256()
        count = 0
        for part in info['parts']:
            data = (snn / part['path']).read_bytes()
            require(len(data) == part['bytes'], 'SNN_PART_SIZE:' + part['path'])
            h.update(data)
            count += len(data)
        require(count == parts['original_file_size'], 'SNN_NATIVE_SIZE:' + seed)
        require(h.hexdigest() == info['full_sha256'], 'SNN_NATIVE_SHA256:' + seed)
        native_hashes[seed] = h.hexdigest()

    moe = ROOT / 'experiments/gfg_moe_cifar_training_learning_inference_generalization_v1'
    moe_manifest = json.loads((moe / 'FORMAL_RUNTIME_MANIFEST.json').read_text())
    moe_summary = json.loads((moe / 'FORMAL_RESULTS.json').read_text())
    require(digest(moe / 'FORMAL_RESULTS.json') == moe_manifest['files']['GENERALIZATION_RESULTS.json']['sha256'],
            'MOE_AGGREGATE_RUNTIME_IDENTITY')
    require(moe_summary['seeds'] == [20260910, 20260911, 20260912], 'MOE_SEEDS')
    require(all(moe_summary['training_learning']['tests'].values())
            and moe_summary['inference']['all_seeds_pass'], 'MOE_RECORDED_VERDICTS')

    gnn = json.loads((ROOT / 'independent_stress_tests/gnn/gnn_falsification_audit.json').read_text())
    bad_seed = next(x for x in gnn['seed_results'] if x['seed'] == 20261024)
    require(bad_seed['all_gates']['finite_amplitude_nonlinearity'] is False
            and bad_seed['nonlinear_count'] == 4 and bad_seed['nonlinear_total'] == 256,
            'GNN_NEGATIVE_VERDICT_NOT_PRESERVED')
    e05 = (ROOT / 'experiments/gfg_rl_selective_positive_feedback_support_concentration_v1/RESULTS.md').read_text()
    require('**NOT_SUPPORTED**' in e05, 'RL_E05_NEGATIVE_VERDICT_NOT_PRESERVED')
    prediction = (ROOT / 'independent_stress_tests/tlp01_prediction/REPORT.md').read_text(encoding='utf-8')
    require('PREDICTION_TRANSPORT_NOT_SUPPORTED' in prediction, 'SNN_NEGATIVE_VERDICT_NOT_PRESERVED')

    archives = {'status': 'NOT_RUN', 'reason': 'Pass --archive-dir to verify Release payloads.'}
    if archive_dir is not None:
        archive_manifest = json.loads((ROOT / manifest['archive']['manifest']).read_text())
        checked = []
        for name, spec in {**archive_manifest['archives'], **archive_manifest['files']}.items():
            path = archive_dir / name
            if not path.is_file():
                errors.append('MISSING_ARCHIVE_FILE:' + name)
                continue
            actual = digest(path)
            require(path.stat().st_size == spec['bytes'] and actual == spec['sha256'], 'ARCHIVE_MISMATCH:' + name)
            checked.append({'name': name, 'bytes': path.stat().st_size, 'sha256': actual})
        require(digest(archive_dir / 'ARCHIVE_MANIFEST.json') == digest(ROOT / manifest['archive']['manifest']),
                'ARCHIVE_MANIFEST_MISMATCH')
        archives = {'status': 'PASS' if not any('ARCHIVE' in e for e in errors) else 'FAIL',
                    'files': checked, 'manifest_sha256': digest(archive_dir / 'ARCHIVE_MANIFEST.json')}

    # Negative controls for the comparison routine; never mutate experimental files.
    require(mismatches({'x': 'abc'}, {}) == ['x'], 'MISSING_FILE_NEGATIVE_CONTROL')
    require(mismatches({'x': 'abc'}, {'x': 'def'}) == ['x'], 'CHANGED_BLOB_NEGATIVE_CONTROL')
    require(mismatches({'x': 'abc'}, {'x': 'abc'}) == [], 'EXACT_COPY_CONTROL')
    return {'schema': 'tlir-evidence-import-check-v1', 'status': 'PASS' if not errors else 'FAIL',
            'errors': errors, 'source_main_commit': manifest['source_main_commit'],
            'unchanged_source_files': len(preserved), 'imported_stress_files': len(expected),
            'moe_files': len([p for p in preserved if p.startswith('experiments/gfg_moe_cifar_training_learning_inference_generalization_v1/')]),
            'moe_aggregate_sha256_matches_runtime_manifest': digest(moe / 'FORMAL_RESULTS.json') == moe_manifest['files']['GENERALIZATION_RESULTS.json']['sha256'],
            'local_document_links_checked': checked_links, 'snn_native_sha256': native_hashes,
            'negative_result_preservation': not any('NEGATIVE_VERDICT' in e for e in errors),
            'comparison_negative_controls': 'PASS' if not any('CONTROL' in e for e in errors) else 'FAIL',
            'archives': archives, 'native_scientific_experiments_rerun': False,
            'scope': 'Import integrity and navigation only; historical check receipts are unchanged, not rerun.'}


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--archive-dir', type=Path)
    args = parser.parse_args()
    result = check(args.archive_dir)
    print(json.dumps(result, ensure_ascii=False, sort_keys=True, indent=2))
    raise SystemExit(0 if result['status'] == 'PASS' else 1)
