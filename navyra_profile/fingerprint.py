"""
navyra_profile.fingerprint — semantic fingerprinting of LLM traffic.

Reference implementation (working). Converts each prompt into a compact
64-bit semantic fingerprint using sentence embeddings + sign random
projection. Prompts whose fingerprints are close in Hamming distance are
semantically similar — the raw signal behind "how templated is this
traffic?".

This module is deliberately self-contained and contains NO Navyra engine
technology: no gating, no thresholds used in the product, no activation
caching. It is the open, publishable measurement layer.

Design notes for the productionisation work (Ella):
- The embedding backend is pluggable. Default = sentence-transformers
  (all-MiniLM-L6-v2, 384-dim, runs fine on CPU). A hash-based fallback
  exists so the package imports and tests without the ML dependency.
- 64-bit fingerprints keep memory tiny: 1M prompts = 8 MB. All pairwise
  work happens in fingerprint space, never embedding space.
"""

from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass, field
import numpy as np

FINGERPRINT_BITS = 64
_RNG_SEED = 1337  # fixed so fingerprints are reproducible across runs
_WEIGHTS = np.uint64(1) << np.arange(FINGERPRINT_BITS, dtype=np.uint64)



# --------------------------------------------------------------------------
# Embedding backends
# --------------------------------------------------------------------------

class EmbeddingBackend:
    """Interface: embed(texts) -> np.ndarray [n, d] float32."""

    dim: int = 0

    def embed(self, texts: list[str]) -> np.ndarray:
        """Embed a list of input `texts` returning an array of shape
        `[n, d]` with dtype `float32`.

        Implementations must provide this method.
        """
        raise NotImplementedError


class SentenceTransformerBackend(EmbeddingBackend):
    """Default quality backend. pip install sentence-transformers"""

    def __init__(self, model_name: str = "sentence-transformers/all-MiniLM-L6-v2"):
        """Initialise the sentence-transformers backend using `model_name`.

        The selected model is loaded lazily at construction time.
        """
        from sentence_transformers import SentenceTransformer
        self.model = SentenceTransformer(model_name)
        self.dim = self.model.get_sentence_embedding_dimension()

    def embed(self, texts: list[str]) -> np.ndarray:
        """Encode `texts` to L2-normalised float32 embeddings.
        Returns an array of shape `[n, d]`.
        """
        return np.asarray(
            self.model.encode(texts, show_progress_bar=False,
                              normalize_embeddings=True),
            dtype=np.float32)


class HashingBackend(EmbeddingBackend):
    """Dependency-free fallback: character n-gram hashing into a dense
    vector. Much weaker semantically than a real embedding model, but keeps
    the package importable/testable anywhere. Do not use for real studies."""

    def __init__(self, dim: int = 256):
        """Initialise the hashing backend with an output `dim`.

        This backend is a deterministic, dependency-free fallback for
        testing and environments without ML libraries.
        """
        self.dim = dim

    def _vec(self, text: str) -> np.ndarray:
        """Compute a dense vector for `text` via character/word n-gram
        hashing and L2-normalise the result. Returns a float32 vector of
        length `self.dim`.
        """
        v = np.zeros(self.dim, dtype=np.float32)
        toks = re.findall(r"\w+", text.lower())
        grams = toks + [" ".join(p) for p in zip(toks, toks[1:])]
        for g in grams:
            h = int(hashlib.md5(g.encode()).hexdigest()[:8], 16)
            v[h % self.dim] += 1.0
        n = np.linalg.norm(v)
        return v / n if n > 0 else v

    def embed(self, texts: list[str]) -> np.ndarray:
        """Embed `texts` using the internal `_vec` hashing heuristic.

        Returns an array of shape `[n, dim]` and dtype `float32`.
        """
        return np.stack([self._vec(t) for t in texts])


def default_backend() -> EmbeddingBackend:
    """Return the preferred `EmbeddingBackend`.

    Attempts to construct a high-quality `SentenceTransformerBackend` and
    falls back to the lightweight `HashingBackend` with a warning when
    `sentence-transformers` is not installed.
    """
    try:
        return SentenceTransformerBackend()
    except ImportError:
        import warnings
        warnings.warn("sentence-transformers not installed - using weak "
                      "hashing backend. pip install sentence-transformers "
                      "for real measurements.")
        return HashingBackend()


# --------------------------------------------------------------------------
# Fingerprinting
# --------------------------------------------------------------------------

@dataclass
class Fingerprinter:
    """Embeds texts then projects to 64-bit sign-random-projection
    fingerprints. Hamming distance between fingerprints approximates
    angular distance between embeddings."""

    backend: EmbeddingBackend = field(default_factory=default_backend)
    _proj: np.ndarray | None = None

    def _projection(self, dim: int) -> np.ndarray:
        """Return a cached random projection matrix of shape
        `(dim, FINGERPRINT_BITS)`. The matrix is generated with a fixed
        RNG seed so fingerprints are reproducible across runs.
        """
        if self._proj is None or self._proj.shape[0] != dim:
            rng = np.random.default_rng(_RNG_SEED)
            self._proj = rng.standard_normal((dim, FINGERPRINT_BITS)) \
                            .astype(np.float32)
        return self._proj

    def fingerprints(self, texts: list[str],
                     batch_size: int = 256) -> np.ndarray:
        """Returns uint64 array [n] of fingerprints."""
        out = np.empty(len(texts), dtype=np.uint64)

        # ADDED EDIT - ELLA
        # caching the projection matrix in local var to avoid re-generating it for each batch

        proj_matrix = None

        for i in range(0, len(texts), batch_size):
            chunk = texts[i:i + batch_size]
            emb = self.backend.embed(chunk)                # [b, d] 
            
            #Added EDIT - ELLA
            # calling self._projection only once per batch 
            if proj_matrix is None:
                proj_matrix = self._projection(emb.shape[1])

            proj = emb @ proj_matrix    # [b, 64]
            bits = (proj > 0).astype(np.uint64)            # [b, 64]
            
        #Added EDIT - ELLA
        # using _WEIGHTS instead of generating it for each batch  

            out[i:i + len(chunk)] = (bits * _WEIGHTS).sum(axis=1)
        
        return out


def hamming(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Elementwise Hamming distance between uint64 fingerprint arrays.
    Uses the classic popcount-via-bit-tricks approach, vectorised."""
    x = a ^ b
    x = x - ((x >> np.uint64(1)) & np.uint64(0x5555555555555555))
    x = (x & np.uint64(0x3333333333333333)) + \
        ((x >> np.uint64(2)) & np.uint64(0x3333333333333333))
    x = (x + (x >> np.uint64(4))) & np.uint64(0x0F0F0F0F0F0F0F0F)
    return ((x * np.uint64(0x0101010101010101)) >> np.uint64(56)) \
        .astype(np.int64)


def pairwise_hamming(fps: np.ndarray, max_n: int = 5000) -> np.ndarray:
    """Full pairwise Hamming matrix (subsampled above max_n for memory)."""
    if len(fps) > max_n:
        idx = np.random.default_rng(0).choice(len(fps), max_n, replace=False)
        fps = fps[idx]
    return hamming(fps[:, None], fps[None, :])
