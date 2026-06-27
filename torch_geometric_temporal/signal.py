class StaticGraphTemporalSignal:
    """
    Minimal local replacement for torch_geometric_temporal.signal.StaticGraphTemporalSignal.
    This is only for Mac CPU/JAX debugging.
    """

    def __init__(self, edge_index, edge_weight, features, targets, **kwargs):
        self.edge_index = edge_index
        self.edge_weight = edge_weight
        self.features = features
        self.targets = targets
        self.additional_feature_keys = list(kwargs.keys())
        self.additional_features = kwargs

    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx):
        if isinstance(idx, slice):
            return StaticGraphTemporalSignal(
                self.edge_index,
                self.edge_weight,
                self.features[idx],
                self.targets[idx],
                **{k: v[idx] for k, v in self.additional_features.items()}
            )

        class Snapshot:
            pass

        snapshot = Snapshot()
        snapshot.edge_index = self.edge_index
        snapshot.edge_attr = self.edge_weight
        snapshot.x = self.features[idx]
        snapshot.y = self.targets[idx]

        for k, v in self.additional_features.items():
            setattr(snapshot, k, v[idx])

        return snapshot

    def __iter__(self):
        for i in range(len(self)):
            yield self[i]


class StaticGraphTemporalSignalBatch(StaticGraphTemporalSignal):
    pass


def temporal_signal_split(data_iterator, train_ratio=0.8):
    """
    Minimal replacement for torch_geometric_temporal.signal.temporal_signal_split.
    """
    n = len(data_iterator)
    split = int(n * train_ratio)
    return data_iterator[:split], data_iterator[split:]
