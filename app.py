import torch
import torchvision
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torch.utils.data import DataLoader, Dataset, random_split
import torchvision.transforms as T
import os
from PIL import Image
import pickle
import matplotlib.pyplot as plt
import numpy as np


class NightOwlsDataset(Dataset):
    def __init__(self, annotations, root, transforms=None):
        self.root = root
        self.transforms = transforms
        self.annotations = annotations

    def __getitem__(self, idx):
        video_name = list(self.annotations.keys())[idx]
        frame_annotations = self.annotations[video_name]['frame annotations']
        frame = list(frame_annotations.keys())[0]  # Just picking the first frame for simplicity
        img_relative_path = self.annotations[video_name]['images']['path'][frame]
        img_path = os.path.join(self.root, img_relative_path)

        if not os.path.exists(img_path):
            raise FileNotFoundError(f"Image file not found: {img_path}")

        img = Image.open(img_path).convert("RGB")

        annots = frame_annotations[frame]
        boxes = [annot['bounding box'] for annot in annots]
        boxes = torch.as_tensor([[b['x'], b['y'], b['x'] + b['width'], b['y'] + b['height']] for b in boxes],
                                dtype=torch.float32)

        labels = torch.tensor([1] * len(boxes), dtype=torch.int64)  # Assuming 1 is the label for person
        image_id = torch.tensor([idx])
        area = (boxes[:, 3] - boxes[:, 1]) * (boxes[:, 2] - boxes[:, 0])
        iscrowd = torch.zeros((len(boxes),), dtype=torch.int64)

        target = {}
        target["boxes"] = boxes
        target["labels"] = labels
        target["image_id"] = image_id
        target["area"] = area
        target["iscrowd"] = iscrowd

        if self.transforms is not None:
            img = self.transforms(img)

        return img, target

    def __len__(self):
        return len(self.annotations)


def collate_fn(batch):
    return tuple(zip(*batch))


def get_transform(train):
    transforms = []
    transforms.append(T.ToTensor())
    return T.Compose(transforms)


def get_model_instance_segmentation(num_classes):
    model = torchvision.models.detection.fasterrcnn_resnet50_fpn(weights='DEFAULT')
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
    return model


def train_one_epoch(model, optimizer, data_loader, device, epoch, print_freq):
    model.train()
    total_loss = 0.0
    for images, targets in data_loader:
        images = list(image.to(device) for image in images)
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

        loss_dict = model(images, targets)
        losses = sum(loss for loss in loss_dict.values())
        total_loss += losses.item()

        optimizer.zero_grad()
        losses.backward()
        optimizer.step()

        if print_freq and (epoch + 1) % print_freq == 0:
            print(f"Epoch: {epoch}, Loss: {losses.item()}")

    average_loss = total_loss / len(data_loader)
    print(f"Epoch {epoch} - Average Loss: {average_loss}")


def visualize_results(images, targets, outputs, idx):
    # Convert tensor to numpy array and display images
    img = images[idx].cpu().numpy().transpose((1, 2, 0))
    img = img * 255
    img = img.astype(np.uint8)

    plt.figure(figsize=(12, 8))
    plt.imshow(img)

    # Draw bounding boxes
    boxes = outputs[idx]['boxes'].cpu().numpy()
    labels = outputs[idx]['labels'].cpu().numpy()
    scores = outputs[idx]['scores'].cpu().numpy()

    for box, label, score in zip(boxes, labels, scores):
        if score > 0.5:  # Only show boxes with score > 0.5
            x0, y0, x1, y1 = box
            plt.gca().add_patch(plt.Rectangle((x0, y0), x1 - x0, y1 - y0, fill=False, color='red', linewidth=3))
            plt.text(x0, y0, f'Label: {label}, Score: {score:.2f}', color='red', fontsize=12,
                     bbox=dict(facecolor='yellow', alpha=0.5))

    plt.axis('off')
    plt.show()


def main():
    annotations_file = 'annotations.pkl'

    if not os.path.exists(annotations_file):
        print(f"Error: {annotations_file} not found.")
        return

    with open(annotations_file, 'rb') as f:
        annotations = pickle.load(f, encoding='latin1')  # Add encoding='latin1' to handle the encoding issue

    root_dir = 'images_annotated'
    full_dataset = NightOwlsDataset(annotations, root_dir, get_transform(train=True))

    # Split dataset into training and testing sets
    total_size = len(full_dataset)
    test_size = int(0.2 * total_size)  # 20% for testing
    train_size = total_size - test_size
    train_dataset, test_dataset = random_split(full_dataset, [train_size, test_size])

    # Create DataLoader instances
    data_loader = DataLoader(train_dataset, batch_size=2, shuffle=True, num_workers=4, collate_fn=collate_fn)
    data_loader_test = DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=4, collate_fn=collate_fn)

    # Prepare for GPU
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    num_classes = 2  # 1 class (person) + background
    model = get_model_instance_segmentation(num_classes)
    model.to(device)

    # Define the optimizer
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.SGD(params, lr=0.005, momentum=0.9, weight_decay=0.0005)

    # Training loop
    num_epochs = 25
    for epoch in range(num_epochs):
        train_one_epoch(model, optimizer, data_loader, device, epoch, print_freq=1)

    # Save the trained model
    torch.save(model.state_dict(), "nightowls_model.pth")

    # Evaluate the model on the test set
    model.eval()
    with torch.no_grad():
        for images, targets in data_loader_test:
            images = list(img.to(device) for img in images)
            outputs = model(images)
            for idx in range(len(images)):
                visualize_results(images, targets, outputs, idx)
                break


if __name__ == "__main__":
    main()
