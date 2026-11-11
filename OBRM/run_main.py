#!/usr/bin/env python3
"""Build and execute the OBRM main experiment once, writing outside this tree."""
import argparse
import csv
import hashlib
import json
import math
import os
import shutil
import statistics
import struct
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

# Do not create bytecode artifacts in the code-and-data submission directory.
sys.dont_write_bytecode = True
from experiment_settings import load_settings

PROJECT = Path(__file__).resolve().parent
PACKAGE_ROOT = PROJECT.parent
SGX_PROJECT = PACKAGE_ROOT / 'OBRMSGX'
PARAMETERS = {"K", "N", "B", "m", "r", "hidden", "epochs", "lr", "seed", "data_encoding"}


def file_info(path, required, dimension, value_bytes=4):
    with path.open('rb') as f:
        raw = f.read(4)
        if len(raw) != 4 or struct.unpack('<i', raw)[0] != dimension:
            raise ValueError(f'Unexpected dimension in {path}')
    stride = 4 + dimension * value_bytes
    size = path.stat().st_size
    if size % stride or size // stride < required:
        raise ValueError(f'Insufficient or truncated vectors: {path}')
    digest = hashlib.sha256()
    remaining = required * stride
    with path.open('rb') as f:
        while remaining:
            block = f.read(min(remaining, 1024 * 1024))
            if not block:
                raise ValueError(f'Truncated prefix: {path}')
            digest.update(block)
            remaining -= len(block)
    return {'path': str(path), 'dimension': dimension, 'available_records': size // stride,
            'used_prefix_records': required, 'value_bytes': value_bytes, 'prefix_sha256': digest.hexdigest()}


def save_csv(path, rows):
    if not rows:
        return
    with path.open('w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def groundtruth_info(path, query_count, base_count):
    with path.open('rb') as f:
        raw = f.read(4)
        if len(raw) != 4:
            raise ValueError(f'Empty groundtruth: {path}')
        k = struct.unpack('<i', raw)[0]
        if k < 10:
            raise ValueError(f'Groundtruth needs at least 10 neighbors: {path}')
        info = file_info(path, query_count, k)
        f.seek(0)
        for _ in range(query_count):
            row = f.read(4 * (k + 1))
            values = struct.unpack('<' + 'i' * (k + 1), row)
            if values[0] != k or any(i < 0 or i >= base_count for i in values[1:]) or len(set(values[1:])) != k:
                raise ValueError(f'Invalid neighbor identifiers in {path}')
    return info


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path,
                        help='Optional external JSON override; defaults to experiment_settings.py')
    parser.add_argument('--data-root', type=Path)
    parser.add_argument('--sgx-sdk', type=Path)
    parser.add_argument('--sgx-mode', choices=['SIM', 'HW'], default='SIM')
    parser.add_argument('--output-dir', type=Path)
    parser.add_argument('--datasets', nargs='+', help='Defaults to all configured datasets')
    parser.add_argument('--scheme', choices=['base', 'optimized'], required=True,
                        help='Run one OBRM design and print only its query times')
    parser.add_argument('--recompute-groundtruth', action='store_true',
                        help='Recompute exact neighbors instead of using the packaged groundtruth')
    parser.add_argument('--jobs', type=int, default=2, help='Build parallelism; queries run serially')
    args = parser.parse_args()
    config = json.loads(args.config.read_text()) if args.config else load_settings()
    sizes, nq, minimum = config['sizes'], config['query_count'], config['minimum_recall']
    if (not sizes or len(set(sizes)) != len(sizes) or any(type(n) is not int or n < 10 for n in sizes)
            or type(nq) is not int or nq <= 0 or not 0 <= minimum <= 1 or args.jobs < 1):
        raise ValueError('Invalid experiment sizes, query count, recall threshold, or build jobs')
    selected = args.datasets or list(config['datasets'])
    if len(set(selected)) != len(selected) or any(name not in config['datasets'] for name in selected):
        raise ValueError('Unknown or repeated dataset')
    data_root = (args.data_root or PROJECT / config['data_root']).resolve()
    sdk = (args.sgx_sdk or Path(os.environ.get('SGX_SDK', str(PACKAGE_ROOT / config['sgx_sdk'])))).resolve()
    if not (sdk / 'buildenv.mk').is_file() or ' ' in str(sdk):
        raise ValueError('Supply an installed Intel SGX SDK path without spaces using --sgx-sdk')
    for command in ('cmake', 'make', 'g++', 'openssl'):
        if not shutil.which(command):
            raise ValueError(f'Missing build dependency: {command}')
    stamp = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S_%fZ')
    out = (args.output_dir or PACKAGE_ROOT.parent / 'OBRM main results' / stamp).resolve()
    if out == PACKAGE_ROOT or PACKAGE_ROOT in out.parents:
        raise ValueError('Output must be outside the OBRM main code-and-data directory')
    if out == data_root or out == sdk or data_root in out.parents or sdk in out.parents:
        raise ValueError('Output must not be inside the dataset or SDK directory')
    inputs, cases = {}, []
    for name in selected:
        d = config['datasets'][name]
        inputs[name] = {}
        for n in sizes:
            params = dict(d['parameters'])
            params.update(d.get('by_size', {}).get(str(n), {}))
            required_encoding = params.get('data_encoding')
            if params.get('data_encoding') != required_encoding:
                raise ValueError(f'{name}: all vector coordinates must use {required_encoding}')
            width = 1 if required_encoding == 'uint8' else 2
            replacements = {'size': n, 'size_k': n // 1000, 'dataset': name}
            base, query = (data_root / d[key].format(**replacements) for key in ('base', 'query'))
            extension = '.bvecs' if required_encoding == 'uint8' else '.hvecs'
            if base.suffix != extension or query.suffix != extension:
                raise ValueError(f'{name}: base and query must both use {extension}')
            supplied_gt = data_root / d['groundtruth'].format(**replacements) if d.get('groundtruth') else None
            case_inputs = {'coordinate_encoding': required_encoding,
                           'base': file_info(base, n, d['dimension'], width),
                           'query': file_info(query, nq, d['dimension'], width)}
            if supplied_gt:
                if case_inputs['base']['available_records'] != n:
                    raise ValueError(f'Scale-specific base file must contain exactly {n} vectors: {base}')
                case_inputs['groundtruth'] = groundtruth_info(supplied_gt, nq, n)
            inputs[name][str(n)] = case_inputs
            if set(params) != PARAMETERS:
                raise ValueError(f'Unexpected or missing parameters for {name} {n}')
            if any(type(params[k]) is not int or params[k] < 1 for k in ('K', 'N', 'B', 'm', 'r', 'hidden', 'epochs')):
                raise ValueError('Positive integer index parameters required')
            if (params['K'] < 2 or params['r'] > 32 or type(params['seed']) is not int
                    or not 0 <= params['seed'] <= 0xffffffff or not math.isfinite(params['lr']) or params['lr'] <= 0
                    or params['data_encoding'] not in ('float16', 'uint8')):
                raise ValueError('Invalid index parameter range')
            cases.append((name, n, base, query, supplied_gt, params))
    out.mkdir(parents=True, exist_ok=False)
    diagnostics = out / 'diagnostics'
    diagnostics.mkdir()
    for folder in ('logs', 'groundtruth', 'per_query', 'neighbors'):
        (diagnostics / folder).mkdir()
    for folder in ('per_query', 'work'):
        (out / folder).mkdir()
    env = os.environ.copy()
    env['LD_LIBRARY_PATH'] = str(sdk / 'lib64') + ':' + env.get('LD_LIBRARY_PATH', '')
    manifest = {'status': 'running', 'configuration': config, 'selected_datasets': selected,
                'inputs': inputs, 'sgx_mode': args.sgx_mode, 'sgx_sdk': str(sdk), 'commands': [],
                'query_execution': 'one pass, serial queries, no excluded warm-up',
                'source_sha256': {str(p.relative_to(PACKAGE_ROOT)): hashlib.sha256(p.read_bytes()).hexdigest()
                                  for p in sorted(PACKAGE_ROOT.rglob('*'))
                                  if p.is_file() and PROJECT / 'datasets' not in p.parents
                                  and '__pycache__' not in p.parts}}
    manifest_path = diagnostics / 'manifest.json'

    def save_manifest():
        manifest_path.write_text(json.dumps(manifest, indent=2))

    def run(label, command, cwd=None):
        command = list(map(str, command))
        manifest['commands'].append({'label': label, 'argv': command, 'cwd': str(cwd or PROJECT)})
        save_manifest()
        logfile = diagnostics / 'logs' / (label + '.log')
        with logfile.open('w') as f:
            result = subprocess.run(command, cwd=cwd or PROJECT, env=env, stdout=f, stderr=subprocess.STDOUT)
        if result.returncode:
            raise RuntimeError(f'{label} failed; see {logfile}')
        return dict(line.split('=', 1) for line in logfile.read_text().splitlines()
                    if '=' in line and line.count('=') == 1)

    rows, detailed_rows, failures = [], [], []
    try:
        build = out / 'build'
        run('cmake_configure', ['cmake', '-S', PROJECT, '-B', build, '-DCMAKE_BUILD_TYPE=Release'])
        run('cmake_build', ['cmake', '--build', build, '-j', args.jobs])
        enclave = out / 'work' / 'sgx'
        shutil.copytree(SGX_PROJECT, enclave)
        run('sgx_build', ['make', '-j', args.jobs, f'SGX_SDK={sdk}', f'SGX_MODE={args.sgx_mode}', 'SGX_DEBUG=0'], enclave)
        manifest['binary_sha256'] = {
            str(p.relative_to(out)): hashlib.sha256(p.read_bytes()).hexdigest()
            for p in (build / 'obrm_export', build / 'obrm_groundtruth', enclave / 'app', enclave / 'enclave.signed.so')}
        for name, n, base, query, supplied_gt, params in cases:
            label = f'{name}_{n}'
            gt = diagnostics / 'groundtruth' / (label + '.ivecs')
            if supplied_gt and not args.recompute_groundtruth:
                shutil.copyfile(supplied_gt, gt)
            else:
                run(label + '_groundtruth', [build / 'obrm_groundtruth', base, query, gt, n, nq, params['data_encoding']])
            prefix = out / 'work' / label
            options = [item for key, value in params.items()
                       for item in ('--' + key.replace('data_encoding', 'data-encoding'), value)]
            run(label + '_export', [build / 'obrm_export', base, prefix, n] + options)
            for scheme in (args.scheme,):
                per_query = out / 'per_query' / f'{label}_{scheme}.csv'
                raw_per_query = diagnostics / 'per_query' / f'{label}_{scheme}.csv'
                neighbors = diagnostics / 'neighbors' / f'{label}_{scheme}.ivecs'
                values = run(label + '_' + scheme, [enclave / 'app', scheme,
                             str(prefix) + f'.{scheme}.blocks', query, gt, nq, params['r'], raw_per_query, neighbors], enclave)
                with raw_per_query.open() as f:
                    individual = list(csv.DictReader(f))
                times = [float(r['time_ms']) for r in individual]
                if len(times) != nq or any(not math.isfinite(t) or t < 0 for t in times):
                    raise RuntimeError('Invalid per-query timing output')
                recall = float(values['recall_at_10'])
                if not math.isfinite(recall) or not 0 <= recall <= 1:
                    raise RuntimeError('Invalid recall output')
                per_query_recalls = [float(record['recall_at_10']) for record in individual]
                if (any(not math.isfinite(value) or not 0 <= value <= 1 for value in per_query_recalls)
                        or abs(statistics.mean(per_query_recalls) - recall) > 1e-9
                        or [int(record['query_index']) for record in individual] != list(range(nq))):
                    raise RuntimeError('Invalid per-query validation output')
                save_csv(per_query, [{'query_index': int(record['query_index']), 'time_ms': time}
                                     for record, time in zip(individual, times)])
                passed = recall >= minimum
                detailed_rows.append({'dataset': name, 'data_size': n, 'scheme': scheme, 'query_count': nq,
                             **params, 'execution_mode': values['execution_mode'], 'recall_at_10': recall,
                             'meets_recall_target': passed, 'mean_query_ms': statistics.mean(times),
                             'median_query_ms': statistics.median(times),
                             'p95_query_ms': sorted(times)[math.ceil(0.95 * nq) - 1],
                             'oram_path_accesses': int(values['oram_path_accesses']),
                             'stash_hits': int(values['stash_hits']),
                             'query_cache_hits': int(values['query_cache_hits']),
                             'pending_stash_peak': int(values['pending_stash_peak']),
                             'per_query_file': str(raw_per_query.relative_to(out))})
                save_csv(diagnostics / 'measurements.csv', detailed_rows)
                rows.append({'dataset': name, 'data_size': n, 'scheme': scheme,
                             'mean_query_ms': statistics.mean(times)})
                save_csv(out / 'measurements.csv', rows)
                if not passed:
                    failures.append(f'{label} {scheme}: Recall@10={recall:.6f} < {minimum}')
                print(f'{name} {n // 1000}K {scheme}: {statistics.mean(times):.6f} ms/query', flush=True)
            # Serialized blocks are temporary runtime data, not reusable results.
            Path(str(prefix) + f'.{args.scheme}.blocks').unlink()
        if failures:
            manifest['status'] = 'failed_validation'
            manifest['failures'] = failures
            save_manifest()
            raise RuntimeError('Result validation failed; see diagnostics/manifest.json. No main_results.csv was generated.')
        save_csv(out / 'main_results.csv', rows)
        manifest['status'] = 'complete'
        save_manifest()
    except Exception as error:
        if manifest['status'] == 'running':
            manifest['status'] = 'failed'
        manifest['error'] = str(error)
        save_manifest()
        raise


if __name__ == '__main__':
    try:
        main()
    except (OSError, ValueError, KeyError, RuntimeError) as error:
        print(f'error: {error}', file=sys.stderr)
        sys.exit(1)
