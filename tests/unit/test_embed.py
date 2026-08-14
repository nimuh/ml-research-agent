"""Embeddings: determinism, normalization, similarity signal, cache hits."""

from __future__ import annotations

import numpy as np
import pytest
from kb_helpers import kb_config

from ml_research_agent.errors import ConfigError
from ml_research_agent.knowledge.embed import CachedEmbedder, HashingEmbedder, get_embedder

TEXTS = [
    "sparse attention reduces the quadratic cost of transformers",
    "attention sparsity lowers transformer compute cost",
    "curriculum learning orders training examples by difficulty",
]


def test_hashing_embedder_is_deterministic_across_instances():
    a = HashingEmbedder(dim=128).embed(TEXTS)
    b = HashingEmbedder(dim=128).embed(TEXTS)
    assert np.array_equal(a, b)
    assert a.dtype == np.float32
    assert a.shape == (3, 128)


def test_vectors_are_l2_normalized():
    vectors = HashingEmbedder(dim=64).embed(TEXTS)
    assert np.allclose(np.linalg.norm(vectors, axis=1), 1.0, atol=1e-5)


def test_empty_text_is_a_zero_vector_not_a_nan():
    vector = HashingEmbedder(dim=32).embed([""])[0]
    assert not np.isnan(vector).any()
    assert float(np.linalg.norm(vector)) == 0.0


def test_related_texts_are_closer_than_unrelated_ones():
    vectors = HashingEmbedder(dim=512).embed(TEXTS)
    related = float(vectors[0] @ vectors[1])
    unrelated = float(vectors[0] @ vectors[2])
    assert related > unrelated


def test_a_different_seed_gives_a_different_projection():
    a = HashingEmbedder(dim=128, seed=0).embed(TEXTS)
    b = HashingEmbedder(dim=128, seed=7).embed(TEXTS)
    assert not np.array_equal(a, b)


def test_cache_serves_repeats_and_survives_a_new_instance(tmp_path):
    inner = HashingEmbedder(dim=64)
    cached = CachedEmbedder(inner, tmp_path / "cache", model_name="hashing-64")
    first = cached.embed(TEXTS)
    assert (cached.hits, cached.misses) == (0, 3)

    second = cached.embed(TEXTS)
    assert np.array_equal(first, second)
    assert cached.hits == 3

    reopened = CachedEmbedder(HashingEmbedder(dim=64), tmp_path / "cache", model_name="hashing-64")
    third = reopened.embed(TEXTS)
    assert np.array_equal(first, third)
    assert (reopened.hits, reopened.misses) == (3, 0)


def test_cache_only_recomputes_the_misses(tmp_path):
    cached = CachedEmbedder(HashingEmbedder(dim=64), tmp_path / "cache", model_name="m")
    cached.embed(TEXTS[:1])
    mixed = cached.embed(TEXTS)
    assert (cached.hits, cached.misses) == (1, 3)
    # Counting hits is not enough: a mixed hit/miss batch must still put each
    # vector on its own row, or the KB indexes passages under other passages'
    # embeddings and every retrieval is quietly wrong.
    assert np.array_equal(mixed, HashingEmbedder(dim=64).embed(TEXTS))


def test_get_embedder_returns_a_cached_hashing_embedder(tmp_path):
    config = kb_config(tmp_path, embedding_dim=256)
    embedder = get_embedder(config)
    assert isinstance(embedder, CachedEmbedder)
    assert embedder.dim == 256
    assert embedder.embed(["hello world"]).shape == (1, 256)


def test_api_backends_are_refused_rather_than_silently_faked(tmp_path):
    config = kb_config(tmp_path, embedding_backend="openai")
    with pytest.raises(ConfigError):
        get_embedder(config)
