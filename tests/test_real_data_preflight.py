"""只使用临时合成 SNP/CSV；不读取真实数据、不训练或筛选。"""

import hashlib
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd

from experiments.spec import ExperimentSpec, digest, file_digest
from utils.RealDataManifest import (RealDataManifest, check_ids, discover_ids, experiment_template,
                                    inspect_genotype, manifest_hash, redacted_example, scan_data,
                                    time_grid, verify_manifest, write_manifest)


class RealDataPreflightTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.data = self.root / 'data'
        self.pheno = self.data / 'pheno'
        self.snp = self.data / 'snp' / 'raw.npy'
        self.pheno.mkdir(parents=True)
        self.snp.parent.mkdir()
        self.x = np.array([[0, 0, -1, 2, 1], [1, 0, 2, 2, 1],
                           [2, 0, np.nan, 2, 1], [0, 0, 0, 2, 1]], dtype=float, order='F')
        np.save(self.snp, self.x)
        self.y = np.arange(12).reshape(4, 3).astype(float)
        for trait in ('Trait_A', 'Trait_B'):
            pd.DataFrame(self.y, columns=['X2', 'X10', 'X12']).to_csv(self.pheno / f'{trait}.csv', index=False)
        self.out = self.root / 'outputs'

    def scan(self, **kwargs):
        dest = scan_data(self.snp, self.pheno, self.out, expected_traits=2, block_columns=2,
                         attestation_date=kwargs.pop('attestation_date', '2026-10-22'), **kwargs)
        return dest, json.loads((dest / 'real_data_manifest.json').read_text())

    def test_stream_hash_and_column_statistics_match_direct(self):
        self.assertEqual(file_digest(self.snp), hashlib.sha256(self.snp.read_bytes()).hexdigest())
        info = inspect_genotype(self.snp, 4, block_columns=2)
        self.assertEqual(info['shape'], [4, 5])
        self.assertEqual(info['order'], 'F')
        self.assertEqual(info['missing_count'], 2)
        self.assertEqual(info['dosage_counts']['nan'], 1)
        self.assertEqual(info['dosage_counts']['minus_one'], 1)
        self.assertAlmostEqual(info['missing_rate'], .1)
        observed = self.x.copy()
        observed[observed == -1] = np.nan
        self.assertAlmostEqual(info['column_variance']['mean'], float(np.nanvar(observed, axis=0).mean()))
        frequencies = np.nanmean(observed, axis=0) / 2
        self.assertAlmostEqual(info['column_maf']['mean'], np.minimum(frequencies, 1-frequencies).mean())
        self.assertEqual(info['constant_columns'], 3)
        self.assertFalse(info['selection_indices_generated'])

    def test_mmap_readonly_and_block_size_invariance(self):
        original = np.load
        calls = []
        def recording(*args, **kwargs):
            calls.append(kwargs)
            return original(*args, **kwargs)
        with patch('utils.RealDataManifest.np.load', side_effect=recording):
            small = inspect_genotype(self.snp, 4, block_columns=1)
        self.assertTrue(calls)
        self.assertTrue(all(c['mmap_mode'] == 'r' and c['allow_pickle'] is False for c in calls))
        large = inspect_genotype(self.snp, 4, block_columns=5)
        for name in ('column_maf', 'column_missing_rate', 'column_variance', 'dosage_counts'):
            self.assertEqual(small[name], large[name])

    def test_invalid_dosage_inf_and_fraction_rejected(self):
        for value in (3, -2, np.inf, .5):
            x = self.x.copy()
            x[0, 0] = value
            np.save(self.snp, x)
            with self.assertRaisesRegex(ValueError, '非法剂量'):
                inspect_genotype(self.snp, 4, block_columns=2)

    def test_all_missing_column_explicit_undefined_not_zero(self):
        x = self.x.copy()
        x[:, 0] = -1
        np.save(self.snp, x)
        info = inspect_genotype(self.snp, 4)
        self.assertEqual(info['all_missing_columns'], 1)
        self.assertEqual(info['column_maf']['undefined_count'], 1)
        self.assertEqual(info['constant_columns'], 3)

    def test_transpose_reports_blocked_without_rewriting(self):
        np.save(self.snp, self.x.T)
        source_hash = file_digest(self.snp)
        dest, payload = self.scan()
        self.assertEqual(payload['status'], 'blocked')
        self.assertIn('疑似转置', payload['errors'][0])
        self.assertEqual(payload['genotype']['shape'], [5, 4])
        self.assertEqual(file_digest(self.snp), source_hash)
        self.assertTrue((dest / 'scan_report.json').is_file())
        with self.assertRaisesRegex(ValueError, '阻塞'):
            verify_manifest(dest / 'real_data_manifest.json')

    def test_time_natural_order_duplicate_unparseable(self):
        info = time_grid(['T10', 'T2'])
        self.assertEqual(info['natural_labels'], ['T2', 'T10'])
        self.assertFalse(info['already_natural'])
        for names in (['T2', 'X2'], ['T2', 'T2'], ['flower', 'T2'], []):
            with self.assertRaises(ValueError):
                time_grid(names)
        pd.DataFrame(self.y, columns=['X10', 'X2', 'X12']).to_csv(self.pheno / 'Trait_A.csv', index=False)
        _, payload = self.scan()
        self.assertIn('禁止静默重排', payload['errors'][0])
        self.assertEqual(payload['phenotypes'][0]['time']['natural_permutation'], [1, 0, 2])

    def test_duplicate_csv_header_and_grid_mismatch_block(self):
        (self.pheno / 'Trait_A.csv').write_text('X2,X2\n1,2\n')
        _, payload = self.scan()
        self.assertIn('重复列名', payload['errors'][0])
        pd.DataFrame(self.y, columns=['X2', 'X10', 'X13']).to_csv(self.pheno / 'Trait_A.csv', index=False)
        _, payload = self.scan()
        self.assertIn('时间网格不一致', payload['errors'][0])

    def test_phenotype_missing_blocks_before_genotype_scan(self):
        y = self.y.copy()
        y[0, 0] = np.nan
        pd.DataFrame(y, columns=['X2', 'X10', 'X12']).to_csv(self.pheno / 'Trait_A.csv', index=False)
        with patch('utils.RealDataManifest.inspect_genotype', side_effect=AssertionError('must stop')):
            _, payload = self.scan()
        self.assertEqual(payload['status'], 'blocked')
        self.assertEqual(payload['phenotypes'][0]['nonfinite_count'], 1)

    def test_ids_unique_same_set_same_order(self):
        check_ids(['A', 'B'], ['A', 'B'])
        for actual in (['B', 'A'], ['A', 'C'], ['A', 'A'], ['A', '']):
            with self.assertRaises(ValueError):
                check_ids(['A', 'B'], actual)

    def test_biological_id_preferred_and_wrong_order_rejected(self):
        mapping = self.data / 'rows.csv'
        pd.DataFrame({'sample_id': ['a', 'b', 'c', 'd']}).to_csv(mapping, index=False)
        for trait in ('Trait_A', 'Trait_B'):
            frame = pd.DataFrame(self.y, columns=['X2', 'X10', 'X12'])
            frame.insert(0, 'sample_id', ['a', 'b', 'c', 'd'])
            frame.to_csv(self.pheno / f'{trait}.csv', index=False)
        dest, payload = self.scan(snp_ids_file=mapping)
        self.assertEqual(verify_manifest(dest / 'real_data_manifest.json')['identity']['alignment_status'], 'verified_by_biological_id')
        with self.assertRaisesRegex(ValueError, 'runner 只支持'):
            experiment_template(payload)
        frame['sample_id'] = ['b', 'a', 'c', 'd']
        frame.to_csv(self.pheno / 'Trait_B.csv', index=False)
        _, payload = self.scan(snp_ids_file=mapping)
        self.assertIn('错序', payload['errors'][0])

    def test_attestation_binds_all_hashes_and_source_change_invalidates(self):
        dest, payload = self.scan()
        result = verify_manifest(dest / 'real_data_manifest.json')
        identity = result['identity']
        self.assertEqual(identity['evidence_type'], 'user_source_order_attestation')
        self.assertEqual(identity['alignment_status'], 'confirmed_by_user_source_order')
        self.assertEqual(identity['sample_id_source'], 'row_index')
        self.assertEqual(json.loads((dest / 'sample_ids.json').read_text()), [0, 1, 2, 3])
        self.assertEqual(identity['sample_ids_sha256'], digest([0, 1, 2, 3]))
        self.assertEqual(len(identity['confirmation_scope']['phenotypes']), 2)
        pd.DataFrame(self.y[::-1], columns=['X2', 'X10', 'X12']).to_csv(self.pheno / 'Trait_A.csv', index=False)
        with self.assertRaisesRegex(ValueError, '确认已失效'):
            verify_manifest(dest / 'real_data_manifest.json')

    def test_changed_sources_cannot_reuse_old_attestation_date(self):
        self.scan()
        pd.DataFrame(self.y + 1, columns=['X2', 'X10', 'X12']).to_csv(self.pheno / 'Trait_A.csv', index=False)
        _, blocked = self.scan()
        self.assertIn('旧日期不能续用', blocked['errors'][0])
        receipt = self.root / 'renewed_attestation.json'
        receipt.write_text(json.dumps({'evidence_type': 'user_source_order_attestation',
            'attestation_date': '2026-10-23', 'source_hashes': {
                'snp': file_digest(self.snp),
                'phenotypes': {p.name: file_digest(p) for p in self.pheno.glob('*.csv')}}}))
        dest, ready = self.scan(attestation_file=receipt)
        self.assertEqual(ready['status'], 'ready')
        self.assertEqual(verify_manifest(dest / 'real_data_manifest.json')['identity']['attestation_date'], '2026-10-23')

    def test_no_attestation_and_external_mapping_candidates_block(self):
        _, payload = self.scan(attestation_date=None)
        self.assertIn('缺少用户', payload['errors'][0])
        pd.DataFrame({'id': ['a']}).to_csv(self.data / 'unknown.csv', index=False)
        _, payload = self.scan()
        self.assertIn('外部映射候选', payload['errors'][0])

    def test_auxiliary_only_and_new_trait_invalidates_inventory(self):
        pd.DataFrame({'trait': ['A'], 'h2': [.4]}).to_csv(self.data / 'all_phenotype_h2.csv', index=False)
        pd.DataFrame({'gene': ['a'], 'snp': [1]}).to_csv(self.data / 'gene_snp_mapping.csv', index=False)
        dest, payload = self.scan()
        self.assertEqual(len(payload['auxiliary']), 2)
        self.assertEqual(len(payload['phenotypes']), 2)
        pd.DataFrame(self.y, columns=['X2', 'X10', 'X12']).to_csv(self.pheno / 'Extra.csv', index=False)
        with self.assertRaisesRegex(ValueError, '清单变化'):
            verify_manifest(dest / 'real_data_manifest.json')

    def test_canonical_hash_serialization_tamper_and_no_overwrite(self):
        dest, payload = self.scan()
        reverse = dict(reversed(list(payload.items())))
        self.assertEqual(manifest_hash(reverse), payload['manifest_sha256'])
        RealDataManifest.model_validate(payload)
        with self.assertRaises(FileExistsError):
            write_manifest(payload, self.out, sample_ids=[0, 1, 2, 3])
        payload['environment']['tamper'] = True
        (dest / 'real_data_manifest.json').write_text(json.dumps(payload))
        with self.assertRaisesRegex(ValueError, 'canonical'):
            verify_manifest(dest / 'real_data_manifest.json')

    def test_schema_and_public_example_never_copy_private_information(self):
        _, payload = self.scan()
        example = redacted_example(payload)
        text = json.dumps(example)
        for secret in (str(self.root), 'Trait_A', payload['genotype']['sha256'], 'per_time'):
            self.assertNotIn(secret, text)
        self.assertEqual(example['manifest_sha256'], manifest_hash(example))
        self.assertEqual(RealDataManifest.model_json_schema()['properties']['schema_version']['const'], 1)

    def test_experiment_template_uses_frozen_refs_only(self):
        _, payload = self.scan()
        template = experiment_template(payload)
        self.assertEqual(template['data']['snp']['sha256'], payload['genotype']['sha256'])
        template['splits']['manifests'] = [{'path': 'future.npz', 'sha256': '0'*64}]
        template['experiments'][0].update(seed=73, traits=['Trait_A'], k=3, gwas_k=5, tfpp={'screen_missing_strategy': 'mode'})
        spec = ExperimentSpec.model_validate(template)
        self.assertEqual(spec.data.n_snps, 5)
        self.assertIn('<FILL_OUTPUT_ROOT>', spec.output_root)

    def test_larger_array_still_uses_mmap_without_read_bytes(self):
        large = self.data / 'large.npy'
        mapped = np.lib.format.open_memmap(large, mode='w+', dtype=np.int8, shape=(32, 100003), fortran_order=True)
        mapped[:] = 1
        mapped.flush()
        del mapped
        with patch.object(Path, 'read_bytes', side_effect=AssertionError('no whole-file reads')):
            info = inspect_genotype(large, 32, block_columns=127)
        self.assertEqual(info['n_snps'], 100003)
        self.assertEqual(info['constant_columns'], 100003)
        self.assertEqual(info['loading'], 'numpy_mmap_read_only_column_blocks')

    def test_input_directories_are_never_output_locations(self):
        with self.assertRaisesRegex(ValueError, '原始输入'):
            scan_data(self.snp, self.pheno, self.pheno / 'out')

    def test_cli_synthetic_scan_and_verify_no_writes(self):
        cli = Path(__file__).resolve().parents[1] / 'inspect_real_data.py'
        command = [sys.executable, '-B', str(cli)]
        result = subprocess.run(command + ['--snp_file', str(self.snp), '--pheno_dir', str(self.pheno),
            '--output_root', str(self.out), '--expected_traits', '2', '--block_columns', '2',
            '--attestation_date', '2026-10-22'], check=True, capture_output=True, text=True)
        manifest = Path(json.loads(result.stdout)['manifest'])
        before = {str(p): file_digest(p) for p in self.root.rglob('*') if p.is_file()}
        subprocess.run(command + ['--verify_manifest', str(manifest)], check=True, capture_output=True, text=True)
        after = {str(p): file_digest(p) for p in self.root.rglob('*') if p.is_file()}
        self.assertEqual(before, after)
        self.assertTrue((manifest.parent / 'experiment.template.json').is_file())
        self.assertTrue((manifest.parent / 'next_steps.txt').is_file())


if __name__ == '__main__':
    unittest.main()
