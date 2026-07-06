from typing import Dict
from torch.utils.data import DataLoader

from .datasets.FakeTT_data import FakeTT_Dataset, FakeTT_Collator
from .datasets.FakeSV_data import FakeSV_Dataset, FakeSV_Collator
from .datasets.FVC_data import FVC_Dataset, FVC_Collator

def DataloaderFactory(dataset: str, **kwargs):

    dataset_obj = None
    collator = None
    if dataset == 'FakeTT':
        dataset_obj = FakeTT_Dataset(**kwargs)
        collator = FakeTT_Collator()
    elif dataset == 'FakeSV':
        dataset_obj = FakeSV_Dataset(**kwargs)
        collator = FakeSV_Collator()
    elif dataset == 'FVC':
        dataset_obj = FVC_Dataset(**kwargs)
        collator = FVC_Collator()
    else:
        raise NotImplementedError(f"Dataset {dataset} not supported")

    dataloader = DataLoader(dataset_obj, collate_fn=collator, **kwargs)
    return dataloader

def DataDfFactory(dataset: str, **kwargs):
    if dataset == 'FakeTT':
        dataset_obj = FakeTT_Dataset(**kwargs)
    elif dataset == 'FakeSV':
        dataset_obj = FakeSV_Dataset(**kwargs)
    elif dataset == 'FVC':
        dataset_obj = FVC_Dataset(**kwargs)
    else:
        raise NotImplementedError(f"Dataset {dataset} not supported")
    return dataset_obj.get_data(**kwargs)
