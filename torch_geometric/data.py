class Data:
    """
    Minimal local replacement for torch_geometric.data.Data.
    Only used so original repo imports can pass during JAX CPU debugging.
    """
    def __init__(self, **kwargs):
        for k, v in kwargs.items():
            setattr(self, k, v)
