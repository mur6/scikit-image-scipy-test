from pathlib import Path
from dataset import RingFingerDataset

#base_data_dir = Path("data/outputs")
base_data_dir = Path("../../ring-finger-semseg/data/outputs/")

numbers_json = {
    "training": "data/contour_checked_numbers_training.json",
    "validation": "data/contour_checked_numbers_validation.json",
}

train_dataset = RingFingerDataset(base_data_dir / "training", numbers_json["training"])
valid_dataset = RingFingerDataset(base_data_dir / "validation", numbers_json["validation"])

from torch import nn
from torch.utils.data import DataLoader

train_dataloader = DataLoader(train_dataset, batch_size=8)
valid_dataloader = DataLoader(valid_dataset, batch_size=8)

from tqdm import tqdm
pbar = tqdm(train_dataloader)
for idx, batch in enumerate(pbar):
    #maskes = batch[1].to(device)
    print(idx)
