"""CSR storage for policy targets, which are 99.84% zeros.

MEASURED on the generation-8 corpus (833,418 rows): a mean of 6.6 non-zero
entries per 4096-wide row, median 5, max 35. The dense array on disk is
**13.65 GB holding about 0.03 GB of information** -- 416x its own content.

That is not merely wasteful, it is the reason training slowed down. The corpus
is memory-mapped and read in shuffled order, so once the working set stops
fitting in the page cache every batch starts pulling random pages off disk.
Generation 7 trained on an 11.9 GB policy array at 1.91 min/epoch; generation
8, on 13.65 GB, took 3.58 min/epoch -- 14.5% more rows for 87% more time.
Under an accumulating corpus that cliff gets worse every generation.

DESIGN. The storage changes; the interface does not. `open_policies` returns
either a plain dense memmap (for corpora written before this existed) or a
`SparsePolicyTargets`, which answers `len`, `.shape`, `.ndim`, `.dtype` and
row indexing exactly as the dense array did and hands back a **dense
ndarray** for whatever rows are asked for. Every consumer in train.py keeps
working unchanged, and the values it sees are bit-identical -- zeros are
zeros, and the stored entries are copied, not recomputed.

Nothing already on disk is rewritten. Old corpora keep loading densely.
"""
import os

import numpy as np

SPARSE_NAME = "policies_sparse.npz"
DENSE_NAME = "policies.npy"

# 4096 (and the 4288 promotion-aware ABI) both fit a uint16, so indices cost
# 2 bytes rather than 8.
INDEX_DTYPE = np.uint16


class SparsePolicyTargets:
    """CSR-backed stand-in for the dense policies array.

    Supports exactly the operations train.py performs on it: len(), .shape,
    .ndim, .dtype, and indexing by an integer, a slice, or an array of row
    indices. Anything else raises rather than silently returning a shape the
    caller did not expect.
    """

    def __init__(self, indices, values, offsets, width):
        self.indices = indices
        self.values = values
        self.offsets = offsets
        self.width = int(width)
        self.dtype = np.dtype(np.float32)
        self.ndim = 2

    def __len__(self):
        return int(len(self.offsets) - 1)

    @property
    def shape(self):
        return (len(self), self.width)

    @property
    def nbytes(self):
        return int(self.indices.nbytes + self.values.nbytes
                   + self.offsets.nbytes)

    def __getitem__(self, key):
        if isinstance(key, (int, np.integer)):
            return self._densify(np.asarray([key], dtype=np.int64))[0]
        if isinstance(key, slice):
            rows = np.arange(*key.indices(len(self)), dtype=np.int64)
            return self._densify(rows)
        rows = np.asarray(key)
        if rows.dtype == bool:
            rows = np.flatnonzero(rows)
        if rows.ndim != 1:
            raise TypeError(
                f"SparsePolicyTargets supports int, slice, or a 1-D index "
                f"array; got {type(key).__name__} with ndim {rows.ndim}")
        return self._densify(rows.astype(np.int64, copy=False))

    def _densify(self, rows):
        """Scatter the stored entries of `rows` into a dense (len(rows), width)."""
        out = np.zeros((len(rows), self.width), dtype=np.float32)
        if len(rows) == 0:
            return out
        starts = self.offsets[rows]
        counts = self.offsets[rows + 1] - starts
        total = int(counts.sum())
        if total == 0:
            return out
        # Ragged gather: expand each row's [start, start+count) run without a
        # Python loop, which at batch 256 would otherwise dominate the load.
        out_rows = np.repeat(np.arange(len(rows), dtype=np.int64), counts)
        run_starts = np.repeat(starts - np.cumsum(counts) + counts, counts)
        flat = run_starts + np.arange(total, dtype=np.int64)
        out[out_rows, self.indices[flat].astype(np.int64)] = self.values[flat]
        return out

    def to_dense(self):
        """Full dense array. Only for tests and small corpora -- this is the
        13 GB the format exists to avoid."""
        return self[np.arange(len(self), dtype=np.int64)]


def encode_dense(block, width=None):
    """Return (indices, values, counts) for a dense 2-D block."""
    block = np.asarray(block)
    if block.ndim != 2:
        raise ValueError(f"expected a 2-D policy block, got {block.shape}")
    if width is not None and block.shape[1] != width:
        raise ValueError(
            f"policy width mismatch: block is {block.shape[1]}, expected "
            f"{width}")
    rows, cols = np.nonzero(block)
    counts = np.bincount(rows, minlength=block.shape[0]).astype(np.int64)
    return (cols.astype(INDEX_DTYPE), block[rows, cols].astype(np.float32),
            counts)


class Builder:
    """Accumulate dense blocks into CSR without ever holding the dense whole."""

    def __init__(self, width):
        self.width = int(width)
        self._indices, self._values, self._counts = [], [], []

    def add_dense(self, block):
        indices, values, counts = encode_dense(block, self.width)
        self._indices.append(indices)
        self._values.append(values)
        self._counts.append(counts)

    def add_sparse(self, source, rows=None):
        """Append rows of an existing SparsePolicyTargets, staying sparse."""
        if source.width != self.width:
            raise ValueError(
                f"policy width mismatch: source is {source.width}, expected "
                f"{self.width}")
        rows = (np.arange(len(source), dtype=np.int64) if rows is None
                else np.asarray(rows, dtype=np.int64))
        starts = source.offsets[rows]
        counts = source.offsets[rows + 1] - starts
        total = int(counts.sum())
        if total:
            run_starts = np.repeat(starts - np.cumsum(counts) + counts, counts)
            flat = run_starts + np.arange(total, dtype=np.int64)
            self._indices.append(np.asarray(source.indices[flat]))
            self._values.append(np.asarray(source.values[flat]))
        self._counts.append(counts.astype(np.int64))

    def build(self):
        counts = (np.concatenate(self._counts) if self._counts
                  else np.zeros((0,), dtype=np.int64))
        offsets = np.zeros((len(counts) + 1,), dtype=np.int64)
        np.cumsum(counts, out=offsets[1:])
        indices = (np.concatenate(self._indices) if self._indices
                   else np.zeros((0,), dtype=INDEX_DTYPE))
        values = (np.concatenate(self._values) if self._values
                  else np.zeros((0,), dtype=np.float32))
        return SparsePolicyTargets(indices, values, offsets, self.width)

    def save(self, directory):
        save(directory, self.build())


def save(directory, targets):
    """Write the CSR arrays. Atomic: a crash leaves no half-written corpus."""
    path = os.path.join(directory, SPARSE_NAME)
    tmp = path + ".tmp"
    with open(tmp, "wb") as handle:
        np.savez(handle, indices=targets.indices, values=targets.values,
                 offsets=targets.offsets,
                 width=np.asarray(targets.width, dtype=np.int64))
    os.replace(tmp, path)


def load(directory):
    """Load the sparse targets, or None when the corpus has none."""
    path = os.path.join(directory, SPARSE_NAME)
    if not os.path.exists(path):
        return None
    with np.load(path) as handle:
        return SparsePolicyTargets(
            handle["indices"], handle["values"], handle["offsets"],
            int(handle["width"]))


def has_sparse(directory):
    return os.path.exists(os.path.join(directory, SPARSE_NAME))


def open_policies(directory, mmap_mode=None):
    """Return the policy targets in whichever format the corpus has.

    Sparse wins when both are present: it is the format written going forward,
    and a stale dense file left beside it must never silently take precedence.
    """
    targets = load(directory)
    if targets is not None:
        return targets
    dense = os.path.join(directory, DENSE_NAME)
    if not os.path.exists(dense):
        raise FileNotFoundError(
            f"{directory} has neither {SPARSE_NAME} nor {DENSE_NAME}")
    return np.load(dense, mmap_mode=mmap_mode)


def iter_dense_blocks(policies, chunk_rows=2048):
    """Yield dense row-blocks from either representation."""
    for start in range(0, len(policies), chunk_rows):
        end = min(len(policies), start + chunk_rows)
        yield np.asarray(policies[start:end])
