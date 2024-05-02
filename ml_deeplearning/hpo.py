#TODO: Import your dependencies.
#For instance, below are some dependencies you might need if you are using Pytorch
# import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.models as models
import torchvision.transforms as transforms
from torch.utils.data import DataLoader

import argparse

import os
from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True


def test(model, test_loader, criterion, device):
    print("Testing Model on Whole Testing Dataset")
    model.eval()
    running_loss = 0
    running_corrects = 0

    for inputs, labels in test_loader:
        inputs = inputs.to(device)
        labels = labels.to(device)
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        _, preds = torch.max(outputs, 1)
        running_loss += loss.item() * inputs.size(0)
        running_corrects += torch.sum(preds == labels.data).item()

    test_loss = running_loss / len(test_loader.dataset)
    test_acc = running_corrects / len(test_loader.dataset)
    print("\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n".format(
            test_loss, running_corrects, len(test_loader.dataset), 100.0 * test_acc))


def train(model, train_loader, criterion, optimizer, device, epochs):
    for epoch in range(epochs):
        print(f"Epoch {epoch+1}")
        model.train()
        running_loss = 0.0
        running_corrects = 0
        running_samples = 0

        for step, (inputs, labels) in enumerate(train_loader):
            inputs = inputs.to(device)
            labels = labels.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            _, preds = torch.max(outputs, 1)

            running_loss += loss.item() * inputs.size(0)
            running_corrects += torch.sum(preds == labels.data).item()
            running_samples += len(inputs)

            if running_samples % 128 == 0:
                accuracy = running_corrects/running_samples
                print("Images [{}/{} ({:.0f}%)] Loss: {:.2f} Accuracy: {}/{} ({:.2f}%)".format(
                        running_samples,
                        len(train_loader.dataset),
                        100.0 * (running_samples / len(train_loader.dataset)),
                        loss.item(),
                        running_corrects,
                        running_samples,
                        100.0*accuracy,
                    )
                )

            # NOTE: Training on portion (50%) of dataset
            if running_samples > (0.5*len(train_loader.dataset)):
                break

        epoch_loss = running_loss / running_samples
        epoch_acc = running_corrects / running_samples
        print(f"Epoch Train Accuracy: {100*epoch_acc}, Epoch Train Loss: {epoch_loss}")

    return model


def net():
    model = models.resnet18(pretrained=True)

    for param in model.parameters():
        param.requires_grad = False

    num_features = model.fc.in_features
    model.fc = nn.Sequential(
                   nn.Linear(num_features, 133))
    return model


def create_data_loaders(data, batch_size, shuffle):
    '''
    This is an optional function that you may or may not need to implement
    depending on whether you need to use data loaders or not
    '''
    return DataLoader(data, batch_size, shuffle=shuffle)


def main(args):
    model = net()
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Running on Device {device}")
    model = model.to(device)

    loss_criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.fc.parameters(), lr=args.lr)

    train_transform = transforms.Compose([
        transforms.Resize(256),  # Resize the image to a larger size
        transforms.CenterCrop(224),  # Crop the center of the resized image to 224 by 224
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    test_transform = transforms.Compose([
        transforms.Resize(256),  # Resize the image to a larger size
        transforms.CenterCrop(224),  # Crop the center of the resized image to 224 by 224
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    # Create ImageFolder dataset
    train_dataset = torchvision.datasets.ImageFolder(root=os.environ['SM_CHANNEL_TRAIN'], transform=train_transform)
    test_dataset = torchvision.datasets.ImageFolder(root=os.environ['SM_CHANNEL_TEST'], transform=test_transform)

    train_kwargs = {"batch_size": args.batch_size, "shuffle": True}
    test_kwargs = {"batch_size": args.test_batch_size, "shuffle": False}

    train_loader = create_data_loaders(train_dataset, **train_kwargs)
    test_loader = create_data_loaders(test_dataset, **test_kwargs)

    model = train(model, train_loader, loss_criterion, optimizer, device, args.epochs)

    test(model, test_loader, loss_criterion, device)

    model_path = os.path.join(os.environ['SM_MODEL_DIR'], 'dog_resnet.pt')
    torch.save(model.state_dict(), model_path)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--batch-size",
        type=int,
        default=64,
        metavar="N",
        help="input batch size for training (default: 64)",
    )
    parser.add_argument(
        "--test-batch-size",
        type=int,
        default=100,
        metavar="N",
        help="input batch size for testing (default: 100)",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=3,
        metavar="N",
        help="number of epochs to train (default: 3)",
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=0.01,
        metavar="LR",
        help="learning rate (default: 0.01)"
    )
    parser.add_argument(
        "--model-dir",
        type=str,
        default=os.environ.get('SM_MODEL_DIR')
    )
    parser.add_argument(
        "--train",
        type=str,
        default=os.environ.get('SM_CHANNEL_TRAIN')
    )
    parser.add_argument(
        "--test",
        type=str,
        default=os.environ.get('SM_CHANNEL_TEST')
    )

    args = parser.parse_args()

    main(args)
