"""Retrieve exact pilot alleles from the public, indexed GRCh38 dbNSFP4.9a release.

Uses Biopython's BGZF reader and the documented Tabix bin index. HTTP byte ranges
avoid a 39.5 GB download. Only genomic/protein mapping and three score fields are
retained; clinical annotations are excluded from the extracted score table.
"""

from concurrent.futures import ThreadPoolExecutor
import gzip
import hashlib
import io
import json
from pathlib import Path
import struct
import threading
import time
import urllib.request

from Bio import bgzf
import pandas as pd

from . import comparison as c

DATA = c.ROOT / 'data/dbnsfp4.9a'
BASE = 'https://grr.iossifovlab.com/hg38/scores/dbNSFP4.9a/'
SOURCE = {
    'release': 'dbNSFP4.9a', 'assembly': 'GRCh38', 'resource_version': 0,
    'url': BASE + 'dbNSFP4.9a.gz', 'bytes': 39507803250,
    'etag': '1f3e8e14b401b05cf61b25decdaa8692', 'generation': '1781155602928795',
    'documentation': BASE + 'index.html',
    'index': {'name': 'dbNSFP4.9a.gz.tbi', 'md5': '4d7b91406fd25a7e2b1dc76e7d9645f5'},
    'readme': {'name': 'dbNSFP4.9a.readme.txt', 'md5': '0024508a2ed89eed969d799a40620105'},
    'resource': {'name': 'genomic_resource.yaml', 'md5': '09e1de6a896b96706d717e7d690fa26f'},
    'license': 'Academic dbNSFP branch; upstream tool restrictions apply. Public GRR distribution of the 2024 release.',
    'integrity': 'Published MD5 for index and documentation; data ranges require the pinned ETag, '
                 'generation, total length and exact Content-Range. Cached blocks have SHA-256 checksums; '
                 'BGZF verifies decompressed block CRCs. The full data-file MD5 is not recomputed.',
}
FIELDS = ['#chr', 'pos(1-based)', 'ref', 'alt', 'aaref', 'aaalt', 'aapos',
          'Ensembl_transcriptid', 'Ensembl_proteinid', 'Uniprot_acc',
          'SIFT4G_score', 'Polyphen2_HVAR_score', 'EVE_score']
BLOCK_SIZE = 512 * 1024


def metadata_files(directory):
    directory.mkdir(parents=True, exist_ok=True)
    for key in ['index', 'readme', 'resource']:
        spec = SOURCE[key]
        path = directory / spec['name']
        if not path.exists():
            with urllib.request.urlopen(BASE + spec['name'], timeout=60) as response:
                data = response.read()
            c.require(hashlib.md5(data).hexdigest() == spec['md5'], f'Invalid dbNSFP {key} download')
            c.atomic_write(path, data)
        c.require(hashlib.md5(path.read_bytes()).hexdigest() == spec['md5'], f'Invalid cached dbNSFP {key}')


def tabix_chunks(index, variant_keys):
    """Read little-endian TBI bins; query six enclosing bins for each point SNV."""
    raw = gzip.decompress(Path(index).read_bytes())
    c.require(raw[:4] == b'TBI\x01', 'Invalid Tabix index magic')
    n_ref, fmt, seq, beg, end, meta, skip, n_names = struct.unpack_from('<8i', raw, 4)
    c.require((fmt, seq, beg, end, meta, skip) == (0, 1, 2, 2, 35, 1), 'Unexpected Tabix coordinate schema')
    names = raw[36:36 + n_names].decode().rstrip('\0').split('\0')
    c.require(len(names) == n_ref, 'Tabix reference names differ')
    wanted = {}
    for key in variant_keys:
        assembly, chrom, pos, ref, alt = key.split(':')
        c.require(assembly == 'GRCh38' and pos.isdigit() and int(pos) > 0 and
                  ref in list('ACGT') and alt in list('ACGT') and ref != alt, 'Expected a GRCh38 SNV')
        point = int(pos) - 1
        bins = {0, 1 + (point >> 26), 9 + (point >> 23), 73 + (point >> 20),
                585 + (point >> 17), 4681 + (point >> 14)}
        wanted.setdefault(chrom, set()).update(bins)
    offset, chunks = 36 + n_names, []
    for chrom in names:
        n_bins, = struct.unpack_from('<i', raw, offset)
        offset += 4
        for _ in range(n_bins):
            bin_id, n_chunks = struct.unpack_from('<Ii', raw, offset)
            offset += 8
            if bin_id in wanted.get(chrom, set()):
                chunks.extend(struct.unpack_from('<QQ', raw, offset + 16 * i) for i in range(n_chunks))
            offset += 16 * n_chunks
        n_intervals, = struct.unpack_from('<i', raw, offset)
        offset += 4 + 8 * n_intervals
    merged = []
    for start, stop in sorted(set(chunks)):
        c.require(start < stop, 'Invalid Tabix chunk interval')
        if merged and start <= merged[-1][1]:
            merged[-1] = (merged[-1][0], max(stop, merged[-1][1]))
        else:
            merged.append((start, stop))
    return merged


class BlockStore:
    """Shared, checksummed cache of HTTP byte ranges from one pinned object."""
    def __init__(self, directory, source=SOURCE, block_size=BLOCK_SIZE):
        self.directory, self.source, self.block_size = Path(directory), source, block_size
        self.directory.mkdir(parents=True, exist_ok=True)
        self.guard, self.locks = threading.Lock(), {}

    def get(self, number):
        with self.guard:
            lock = self.locks.setdefault(number, threading.Lock())
        with lock:
            path = self.directory / f'{number:08d}.bin'
            checksum = path.with_suffix('.sha256')
            start = number * self.block_size
            stop = min(start + self.block_size, self.source['bytes']) - 1
            if path.exists() and checksum.exists():
                data = path.read_bytes()
                c.require(hashlib.sha256(data).hexdigest() == checksum.read_text().strip(), 'Cached dbNSFP block changed')
                c.require(len(data) == stop - start + 1, 'Cached dbNSFP block size differs')
                return data
            request = urllib.request.Request(self.source['url'], headers={
                'Range': f'bytes={start}-{stop}', 'If-Match': '"' + self.source['etag'] + '"',
                'Accept-Encoding': 'identity'})
            for attempt in range(3):
                try:
                    with urllib.request.urlopen(request, timeout=60) as response:
                        c.require(response.status == 206, 'Server did not honor the dbNSFP byte range')
                        c.require(response.headers.get('ETag', '').strip('"') == self.source['etag'] and
                                  response.headers.get('x-goog-generation') == self.source['generation'],
                                  'Remote dbNSFP identity changed')
                        c.require(response.headers.get('Content-Range') ==
                                  f'bytes {start}-{stop}/{self.source["bytes"]}', 'Wrong dbNSFP byte range')
                        data = response.read(stop - start + 2)
                    break
                except (OSError, TimeoutError):
                    if attempt == 2:
                        raise
                    time.sleep(attempt + 1)
            c.require(len(data) == stop - start + 1, 'Truncated dbNSFP byte range')
            c.atomic_write(path, data)
            c.atomic_write(checksum, hashlib.sha256(data).hexdigest() + '\n')
            return data


class RangeReader(io.RawIOBase):
    def __init__(self, store):
        self.store, self.position, self.number, self.block = store, 0, None, b''

    def tell(self):
        return self.position

    def seek(self, offset, whence=0):
        self.position = offset + (self.position if whence == 1 else self.store.source['bytes'] if whence == 2 else 0)
        c.require(0 <= self.position <= self.store.source['bytes'], 'Invalid dbNSFP seek')
        return self.position

    def read(self, size=-1):
        c.require(size >= 0, 'Unbounded reads of the remote database are disabled')
        remaining = min(size, self.store.source['bytes'] - self.position)
        pieces = []
        while remaining:
            number, start = divmod(self.position, self.store.block_size)
            if number != self.number:
                self.block, self.number = self.store.get(number), number
            count = min(remaining, len(self.block) - start)
            c.require(count > 0, 'Unexpected end of dbNSFP block')
            pieces.append(self.block[start:start + count])
            self.position += count
            remaining -= count
        return b''.join(pieces)


def fetch_annotations(variant_keys, directory=DATA):
    keys, directory = list(variant_keys), Path(directory)
    c.require(len(keys) == len(set(keys)), 'Duplicate pilot keys')
    identity = {'source': SOURCE, 'variant_keys_sha256': hashlib.sha256(json.dumps(keys).encode()).hexdigest(),
                'implementation_sha256': c.digest(__file__), 'retained_columns': FIELDS}
    directory.mkdir(parents=True, exist_ok=True)
    table_path, manifest_path = directory / 'pilot_annotations.tsv.gz', directory / 'acquisition.json'
    if manifest_path.exists():
        manifest = c.read_json(manifest_path)
        c.require(manifest['identity'] == identity, 'dbNSFP extraction identity changed; preserve the old extraction first')
        c.verified(table_path, manifest['table_sha256'])
        print('Reusing verified dbNSFP pilot score extraction offline.', flush=True)
        return pd.read_csv(table_path, sep='\t', dtype=str, keep_default_na=False), manifest
    metadata_files(directory)
    chunks = tabix_chunks(directory / SOURCE['index']['name'], keys)
    store = BlockStore(directory / 'blocks')
    with bgzf.BgzfReader(fileobj=RangeReader(store), mode='rb') as stream:
        header = stream.readline().decode().rstrip('\r\n').split('\t')
    c.require(header[:4] == FIELDS[:4], 'Unexpected dbNSFP genomic columns')
    c.require(set(FIELDS).issubset(header), 'Missing requested dbNSFP score fields')
    indices, wanted = [header.index(name) for name in FIELDS], set(keys)

    def read_chunk(chunk):
        rows = []
        with bgzf.BgzfReader(fileobj=RangeReader(store), mode='rb') as stream:
            stream.seek(chunk[0])
            while stream.tell() < chunk[1]:
                line = stream.readline()
                c.require(bool(line), 'Truncated dbNSFP chunk')
                fields = line.decode().rstrip('\r\n').split('\t')
                c.require(len(fields) == len(header), 'Malformed dbNSFP row')
                key = f'GRCh38:{fields[0]}:{fields[1]}:{fields[2]}:{fields[3]}'
                if key in wanted:
                    rows.append([key, *[fields[i] for i in indices]])
        return rows

    print(f'Fetching {len(chunks):,} indexed regions for {len(keys):,} exact pilot alleles.', flush=True)
    rows = []
    with ThreadPoolExecutor(max_workers=4) as pool:
        for number, matches in enumerate(pool.map(read_chunk, chunks), 1):
            rows.extend(matches)
            if number % 100 == 0:
                print(f'  {number:,}/{len(chunks):,} regions read; {len(rows):,} matching annotations.', flush=True)
    frame = pd.DataFrame(rows, columns=['variant_key', *FIELDS])
    frame.to_csv(table_path, sep='\t', index=False, compression={'method': 'gzip', 'mtime': 0})
    manifest = {'identity': identity, 'table_sha256': c.digest(table_path), 'annotations': len(frame),
                'regions': len(chunks), 'cached_bytes': sum(p.stat().st_size for p in (directory / 'blocks').glob('*.bin')),
                'clinical_columns_retained': False}
    c.atomic_write(manifest_path, json.dumps(manifest, indent=2) + '\n')
    return frame, manifest
