"""Inference engines bundled with the unified voice service (Phase I).

Each engine is a thin wrapper around an upstream library — kept here so the
router code stays tiny and the lazy ModelRegistry can manage VRAM uniformly.
"""
