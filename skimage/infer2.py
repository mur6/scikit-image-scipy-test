import argparse
from pathlib import Path


import matplotlib.pyplot as plt
import matplotlib.patches as patches
import torch
from PIL import Image

from dataset import RingFingerDataset
from model import Net


softmax = torch.nn.Softmax()


def get_images(samples_dir):
    samples_dir = Path(samples_dir)

    def _iter_pil_images():
        sample_images = sorted(list(samples_dir.glob("*.jpg")))
        for p in sample_images:
            image = Image.open(p)
            yield image

    return tuple(_iter_pil_images())



def get_model():
    model = Net()
    model_path = 'models/ring_infer_model_02.pth'
    model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
    return model


# def draw_dot(ax, point):
#     x, y = tuple(point)
#     c = patches.Circle(xy=(x, y), radius=4, color='red')
#     ax.add_patch(c)



def draw_dot(ax, point):
    print(point)
    x, y = tuple(point)
    c = patches.Circle(xy=(x, y), radius=4, color='red')
    ax.add_patch(c)

from torch import nn
from torch.nn import functional as F
# t = torch.arange(32).reshape(2,1,4,4)
# t = t.type(torch.float32)

def main(args):
    logits = torch.load(args.input_pt_file)
    converted = (softmax(logits) > 0.95).type(torch.uint8)
    # converted = softmax(logits)
    orig_image = get_images("../../ring-finger-semseg//data/samples")[0]
    result_image = converted.detach().numpy()[0]
    model = get_model()
    model.eval()
    # plt.tight_layout()
    # plt.savefig(output_image_file, bbox_inches="tight")
    with torch.no_grad():
        masks = torch.from_numpy(result_image[1] + result_image[2] * -1).float()
        print(masks.shape)
        masks.unsqueeze_(0).unsqueeze_(0)
        print(masks.shape)
        masks = F.interpolate(masks, size=(224,224), mode='nearest').int().squeeze(0)
        print(masks.shape)
        outputs = model(masks=masks)
        outputs = outputs * 112 + 112
        print(outputs)
        points = outputs[0]
        # print(masks.shape, points.shape)
        fig, (orig_ax, ax) = plt.subplots(1, 2)
        orig_ax.imshow(orig_image)
        #ax[0].imshow(data[0].numpy().transpose(1, 2, 0))
        ax.imshow(masks.numpy().transpose(1, 2, 0))
        # ax[1].imshow(img)
        draw_dot(ax, points[:2])
        draw_dot(ax, points[2:])
        plt.show()


# def save(ax, orig_image, result_image):
#     ax[0].set_title("Original image")
#     ax[0].imshow(orig_image)
#     ax[1].set_title("hand / ring-finger")
#     ax[1].imshow(result_image[1] + result_image[2] * 2, interpolation="none")
#     # plt.savefig(f"output{idx}.png")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_pt_file", type=Path)
    #parser.add_argument("--output_file_path", type=Path, default="data/contour_checked_numbers.json")
    args = parser.parse_args()
    main(args)
