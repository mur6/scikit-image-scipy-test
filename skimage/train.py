import torch
from sklearn.metrics import accuracy_score
#from tqdm.notebook import tqdm
from transformers import AdamW

from model import Net

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

train_dataloader = DataLoader(train_dataset, batch_size=8, shuffle=True)
valid_dataloader = DataLoader(valid_dataset, batch_size=8)

from tqdm import tqdm


model = Net()

criterion = nn.MSELoss()
optimizer = AdamW(model.parameters(), lr=0.00006)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
print("Model Initialized!")

# from torch.utils.tensorboard import SummaryWriter
# writer = SummaryWriter()

for epoch in range(1, 3 + 1):  # loop over the dataset multiple times
    print("Epoch:", epoch)
    pbar = tqdm(train_dataloader)
    accuracies = []
    losses = []
    val_accuracies = []
    val_losses = []
    model.train()
    train_loss = 0.0
    for idx, batch in enumerate(pbar):
        # get the inputs;
        masks = batch[1]
        # print(masks.unique())
        masks[masks==2] = -1
        masks = masks.to(device)
        points = batch[2].to(device)
        #print(masks.shape, masks.dtype, points.shape)

        # zero the parameter gradients
        optimizer.zero_grad()

        # forward
        outputs = model(masks=masks)
        # print()
        # print(outputs.dtype)

        # evaluate
        points = ((points - (224 / 2.0)) / 112.0).float()

        # pred_labels = predicted[mask].detach().cpu().numpy()
        # true_labels = labels[mask].detach().cpu().numpy()
        # accuracy = accuracy_score(pred_labels, true_labels)
        # accuracies.append(accuracy)
        # pbar.set_postfix(
        #     {"Batch": idx, "Pixel-wise accuracy": sum(accuracies) / len(accuracies), "Loss": sum(losses) / len(losses)}
        # )

        # backward + optimize
        loss = criterion(outputs, points)
        train_loss += loss.item()
        loss.backward()
        optimizer.step()
    else:
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for idx, batch in enumerate(valid_dataloader):
                masks = batch[1]
                masks[masks==2] = -1
                masks = masks.to(device)
                points = batch[2].to(device)
                points = (points - (224 / 2.0)) / 112.0
                outputs = model(masks=masks)
                # upsampled_logits = nn.functional.interpolate(
                #     outputs.logits, size=labels.shape[-2:], mode="bilinear", align_corners=False
                # )
                # predicted = upsampled_logits.argmax(dim=1)

                # mask = labels != 0  # we don't include the background class in the accuracy calculation
                # pred_labels = predicted[mask].detach().cpu().numpy()
                # true_labels = labels[mask].detach().cpu().numpy()
                # accuracy = accuracy_score(pred_labels, true_labels)
                # val_loss = outputs.loss
                # val_accuracies.append(accuracy)
                # val_losses.append(val_loss.item())
                loss = criterion(outputs, points)
                val_loss += loss.item()
    # writer.add_scalar('Loss/train', sum(losses)/len(losses), epoch)
    # writer.add_scalar('Loss/val', sum(val_losses)/len(val_losses), epoch)
    # writer.add_scalar('Accuracy/train', sum(accuracies)/len(accuracies), epoch)
    # writer.add_scalar('Accuracy/val', sum(val_accuracies)/len(val_accuracies), epoch)
    # f"Train Pixel-wise accuracy: {sum(accuracies)/len(accuracies)}"
    # f"Train Loss: {sum(losses)/len(losses)}"
    # f"Val Pixel-wise accuracy: {sum(val_accuracies)/len(val_accuracies)}"
    # f"Val Loss: {sum(val_losses)/len(val_losses)}"
    train_count = len(train_dataloader)
    val_count = len(valid_dataloader)
    s1 = f"Training: Mean Squared Error: {train_loss/train_count}"
    s2 = f"Validation: Mean Squared Error: {val_loss/val_count}"
    print(s1 + " " + s2)
#writer.flush()
