class _UnavailableTorchGeometricTemporalAttentionModel:
    """
    Placeholder class for local CPU/JAX debugging.
    These PyTorch-Geometric-Temporal attention models are not used by our JAX scripts.
    """
    def __init__(self, *args, **kwargs):
        raise ImportError(
            "This is a local stub. The original torch_geometric_temporal attention model "
            "is not available in this Mac CPU/JAX debugging environment."
        )

ASTGCN = _UnavailableTorchGeometricTemporalAttentionModel
GMAN = _UnavailableTorchGeometricTemporalAttentionModel
MSTGCN = _UnavailableTorchGeometricTemporalAttentionModel
