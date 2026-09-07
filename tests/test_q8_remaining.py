"""Boundary, transfer-integrity, score-direction and missing-data checks."""

import gzip
import hashlib
import io
from pathlib import Path
import struct
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd

try:
    from . import q8_dbnsfp as db, q8_remaining as remaining
except ImportError:
    from notebooks.src import q8_dbnsfp as db, q8_remaining as remaining


class TabixTests(unittest.TestCase):
    def test_one_based_boundary_bins_and_overlapping_chunks(self):
        # A minimal wire-format TBI: positions 1..16384 are in bin 4681;
        # position 16385 starts bin 4682. Deliberately overlapping chunks merge.
        raw = b'TBI\x01' + struct.pack('<8i', 1, 0, 1, 2, 2, 35, 1, 2) + b'1\0'
        raw += struct.pack('<i', 2)
        raw += struct.pack('<IiQQQQ', 4681, 2, 100, 200, 180, 250)
        raw += struct.pack('<IiQQ', 4682, 1, 300, 400)
        raw += struct.pack('<iQQ', 2, 100, 300)
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / 'test.tbi'
            path.write_bytes(gzip.compress(raw))
            self.assertEqual(db.tabix_chunks(path, ['GRCh38:1:16384:A:C']), [(100, 250)])
            self.assertEqual(db.tabix_chunks(path, ['GRCh38:1:16385:A:C']), [(300, 400)])
            self.assertEqual(db.tabix_chunks(path, ['GRCh38:2:1:A:C']), [])
            with self.assertRaisesRegex(ValueError, 'GRCh38'):
                db.tabix_chunks(path, ['GRCh37:1:1:A:C'])
            changed = bytearray(raw)
            struct.pack_into('<i', changed, 8, 0x10000)  # BED coordinates cannot masquerade as 1-based.
            path.write_bytes(gzip.compress(changed))
            with self.assertRaisesRegex(ValueError, 'coordinate schema'):
                db.tabix_chunks(path, ['GRCh38:1:1:A:C'])


class RangeTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.content = b'0123456789abcdef'
        self.source = {'url': 'https://example.invalid/data', 'bytes': 16, 'etag': 'etag', 'generation': '1'}
        self.store = db.BlockStore(Path(self.tmp.name), self.source, block_size=4)

    def response(self, request, **kwargs):
        start, end = map(int, request.get_header('Range').removeprefix('bytes=').split('-'))
        response = io.BytesIO(self.content[start:end + 1])
        response.status = 206
        response.headers = {'ETag': '"etag"', 'x-goog-generation': '1',
                            'Content-Range': f'bytes {start}-{end}/16'}
        return response

    def test_seek_cross_block_reads_and_offline_verified_reuse(self):
        with patch.object(db.urllib.request, 'urlopen', side_effect=self.response) as fetch:
            reader = db.RangeReader(self.store)
            self.assertEqual(reader.read(0), b'')
            reader.seek(3)
            self.assertEqual(reader.read(10), self.content[3:13])
            reader.seek(-3, 2)
            self.assertEqual(reader.read(100), b'def')
            self.assertEqual(reader.read(1), b'')
            self.assertEqual(fetch.call_count, 4)
        with patch.object(db.urllib.request, 'urlopen', side_effect=AssertionError('network forbidden')):
            self.assertEqual(db.RangeReader(self.store).read(16), self.content)
        with self.assertRaisesRegex(ValueError, 'Unbounded'):
            db.RangeReader(self.store).read()

    def test_cached_corruption_is_rejected(self):
        with patch.object(db.urllib.request, 'urlopen', side_effect=self.response):
            self.store.get(0)
        (Path(self.tmp.name) / '00000000.bin').write_bytes(b'xxxx')
        with self.assertRaisesRegex(ValueError, 'changed'):
            self.store.get(0)

    def test_wrong_identity_or_range_is_never_cached(self):
        for header, value in [('ETag', 'changed'), ('x-goog-generation', '2'), ('Content-Range', 'bytes 1-4/16')]:
            def wrong(request, **kwargs):
                response = self.response(request)
                response.headers[header] = value
                return response
            with self.subTest(header=header), patch.object(db.urllib.request, 'urlopen', side_effect=wrong):
                with self.assertRaises(ValueError):
                    self.store.get(0)
            self.assertFalse((Path(self.tmp.name) / '00000000.bin').exists())


class ScoreTests(unittest.TestCase):
    def raw(self):
        rows = []
        for pos, sift, poly, eve in [(10, '.;0.8;0.1', '0.2;0.7', '0.4'), (20, '.', '.', '.')]:
            row = {name: '.' for name in db.FIELDS}
            row.update({'#chr': '1', 'pos(1-based)': str(pos), 'ref': 'A', 'alt': 'C',
                        'variant_key': f'GRCh38:1:{pos}:A:C', 'Ensembl_transcriptid': 'T1;T2;T3',
                        'SIFT4G_score': sift, 'Polyphen2_HVAR_score': poly, 'EVE_score': eve,
                        'clinvar_clnsig': 'must not become a feature', 'Polyphen2_HDIV_score': '1'})
            rows.append(row)
        return pd.DataFrame(rows)

    def test_direction_model_choice_and_missing_scores(self):
        raw = self.raw()
        keys = raw.variant_key.tolist() + ['GRCh38:1:30:A:C']
        for slug, expected in [('sift4g', .9), ('polyphen2', .7), ('eve', .4)]:
            with self.subTest(slug=slug):
                scores, annotations = remaining.score_alleles(raw, keys, remaining.METHODS[slug])
                self.assertAlmostEqual(scores.score.iloc[0], expected)
                self.assertTrue(scores.score.iloc[1:].isna().all())
                self.assertEqual(scores.status.tolist(), ['scored', 'score_missing', 'no_exact_allele_match'])
                self.assertNotIn('clinvar_clnsig', annotations)
                self.assertNotIn('Polyphen2_HDIV_score', annotations)
                self.assertNotIn('label', scores)
                self.assertIn('raw_score', annotations)

    def test_wrong_allele_and_nonfinite_score_fail(self):
        raw = self.raw()
        keys = raw.variant_key.tolist()
        raw.loc[0, 'alt'] = 'G'
        with self.assertRaisesRegex(ValueError, 'allele fields'):
            remaining.score_alleles(raw, keys, remaining.METHODS['eve'])
        for value in ['nan', 'inf', '-0.1', '1.1']:
            raw = self.raw()
            raw.loc[0, 'EVE_score'] = value
            with self.subTest(value=value), self.assertRaises(ValueError):
                remaining.score_alleles(raw, keys, remaining.METHODS['eve'])


if __name__ == '__main__':
    unittest.main()
