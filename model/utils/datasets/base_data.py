from torch.utils.data import Dataset
from pathlib import Path


class Base_Dataset(Dataset):
    def __init__(self, **kargs):
        super().__init__()
        self.data_path = Path('data')

    def get_data(self, split: str):
        raise NotImplementedError
