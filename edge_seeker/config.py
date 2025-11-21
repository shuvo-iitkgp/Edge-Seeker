import torch
import random
import numpy as np

SEED = 42

def set_seed(seed: int = SEED) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Default hyperparams for LinkTeller experiments
DEFAULT_TRAIN_EPOCHS = 120
DEFAULT_DELTA = 1e-2
DEFAULT_AGG = "max"
DEFAULT_USE_PROB = True
DEFAULT_NODE2VEC_EPOCHS = 80
