from pathlib import Path
from dataset import RingFingerDataset

#base_data_dir = Path("data/outputs")
base_data_dir = Path("./data/")

numbers_json_filepath = "data/contour_checked_numbers.json"

train_dataset = RingFingerDataset(base_data_dir / "training", numbers_json_filepath)
valid_dataset = RingFingerDataset(base_data_dir / "validation", numbers_json_filepath)

from torch import nn
from torch.utils.data import DataLoader

train_dataloader = DataLoader(train_dataset, batch_size=8, shuffle=True)
valid_dataloader = DataLoader(valid_dataset, batch_size=8)

from tqdm import tqdm
pbar = tqdm(train_dataloader)
for idx, batch in enumerate(pbar):
    #maskes = batch[1].to(device)
    print(idx)
