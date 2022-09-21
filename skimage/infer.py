import torch
import matplotlib.pyplot as plt
import matplotlib.patches as patches

from model import Net
model = Net()

model_path = 'models/ring_infer_model_02.pth'
model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))

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

def draw_dot(ax, point):
    x, y = tuple(point)
    c = patches.Circle(xy=(x, y), radius=4, color='red')
    ax.add_patch(c)

model.eval()
with torch.no_grad():
    data = valid_dataset[0]
    #print(data)
    masks = data[1]
    masks[masks==2] = -1
    points = data[2].numpy()
    #points = (points - (224 / 2.0)) / 112.0
    outputs = model(masks=masks.unsqueeze(0))
    outputs = outputs * 112+ 112
    #, points)
    print(masks.shape, points.shape)
    fig, ax = plt.subplots(1, 2)
    ax[0].imshow(data[0].numpy().transpose(1, 2, 0))
    ax[1].imshow(masks)
    # ax[1].imshow(img)
    draw_dot(ax[1], points[:2])
    draw_dot(ax[1], points[2:])
    plt.show()
