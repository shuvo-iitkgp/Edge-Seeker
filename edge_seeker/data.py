from torch_geometric.datasets import TUDataset

def load_tud_dataset(name: str, root: str = "data/TUD") -> TUDataset:
    """
    Load a TU dataset (MUTAG, PROTEINS, ENZYMES, etc).
    """
    dataset = TUDataset(root=root, name=name)
    return dataset
