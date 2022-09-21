from pathlib import Path
from dataset import RingFingerDataset

#base_data_dir = Path("data/outputs")
base_data_dir = Path("../blender-for-finger-segmentation/data2/")

train_dataset = RingFingerDataset(base_data_dir / "training", "contour_checked_numbers.json", transform=None)
valid_dataset = RingFingerDataset(base_data_dir / "validation", "contour_checked_numbers.json", transform=None)

from torch import nn
from torch.utils.data import DataLoader

train_dataloader = DataLoader(train_dataset, batch_size=8, shuffle=True)
valid_dataloader = DataLoader(valid_dataset, batch_size=8)

from tqdm import tqdm
pbar = tqdm(train_dataloader)
for idx, batch in enumerate(pbar):
    #maskes = batch[1].to(device)
    print(idx)
