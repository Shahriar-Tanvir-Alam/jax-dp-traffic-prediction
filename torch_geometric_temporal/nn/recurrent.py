class _UnavailableTorchGeometricTemporalModel:
    """
    Placeholder class for local CPU/JAX debugging.
    These PyTorch-Geometric-Temporal models are not used by our JAX scripts.
    """
    def __init__(self, *args, **kwargs):
        raise ImportError(
            "This is a local stub. The original torch_geometric_temporal model "
            "is not available in this Mac CPU/JAX debugging environment."
        )

DCRNN = _UnavailableTorchGeometricTemporalModel
A3TGCN = _UnavailableTorchGeometricTemporalModel
AGCRN = _UnavailableTorchGeometricTemporalModel
A3TGCN2 = _UnavailableTorchGeometricTemporalModel
