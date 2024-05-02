import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.models as models
import torchvision.transforms as transforms
from torch.utils.data import DataLoader

import argparse
import smdebug.pytorch as smd
import os
from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True


def test(model, test_loader, criterion, device):
    print("Testing Model on Whole Testing Dataset")
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


def train(model, train_loader, val_loader, criterion, optimizer, device, epochs, hook):
    for epoch in range(epochs):
        print(f"Epoch {epoch+1}/{epochs}")
        hook.set_mode(smd.modes.TRAIN)
        model.train()

        running_train_loss = 0.0
        running_train_corrects = 0
        running_train_samples = 0

        # Training phase
        for step, (inputs, labels) in enumerate(train_loader):
            inputs = inputs.to(device)
            labels = labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            _, preds = torch.max(outputs, 1)
            running_train_loss += loss.item() * inputs.size(0)
            running_train_corrects += torch.sum(preds == labels.data).item()
            running_train_samples += len(inputs)

            if running_train_samples % 128 == 0:  # Adjust this value as needed
                train_accuracy = running_train_corrects / running_train_samples
                print(f"Train Step {step}: Loss: {loss.item():.4f}, Accuracy: {100*train_accuracy:.2f}%")

            # NOTE: Training on portion (50%) of dataset
            if running_train_samples > (0.5*len(train_loader.dataset)):
                break

        epoch_train_loss = running_train_loss / running_train_samples
        epoch_train_acc = running_train_corrects / running_train_samples
        print(f"Epoch Train Loss: {epoch_train_loss:.4f}, Epoch Train Accuracy: {100*epoch_train_acc:.2f}%")

        # Validation phase
        hook.set_mode(smd.modes.EVAL)
        model.eval()  # Set model to evaluation mode
        running_val_loss = 0.0
        running_val_corrects = 0
        running_val_samples = 0

        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs = inputs.to(device)
                labels = labels.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, labels)

                _, preds = torch.max(outputs, 1)
                running_val_loss += loss.item() * inputs.size(0)
                running_val_corrects += torch.sum(preds == labels.data).item()
                running_val_samples += len(inputs)

        epoch_val_loss = running_val_loss / running_val_samples
        epoch_val_acc = running_val_corrects / running_val_samples
        print(f"Epoch Validation Loss: {epoch_val_loss:.4f}, Epoch Validation Accuracy: {100*epoch_val_acc:.2f}%")

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


# Define the model loading function
def model_fn(model_dir):
    model = net()
    model.load_state_dict(torch.load(os.path.join(model_dir, "model.pth")))
    return model


def main(args):
    model = net()
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Running on Device {device}")
    model = model.to(device)
    hook = smd.Hook.create_from_json_file()
    hook.register_hook(model)

    loss_criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.fc.parameters(), lr=args.lr)

    train_transform = transforms.Compose([
        transforms.Resize(256),  # Resize the image to a larger size
        transforms.CenterCrop(224),  # Crop the center of the resized image to 224 by 224
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    val_transform = transforms.Compose([
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
    val_dataset = torchvision.datasets.ImageFolder(root=os.environ['SM_CHANNEL_VALIDATION'], transform=val_transform)
    test_dataset = torchvision.datasets.ImageFolder(root=os.environ['SM_CHANNEL_TEST'], transform=test_transform)

    train_kwargs = {"batch_size": args.batch_size, "shuffle": True}
    val_kwargs = {"batch_size": args.val_batch_size, "shuffle": False}
    test_kwargs = {"batch_size": args.test_batch_size, "shuffle": False}

    train_loader = create_data_loaders(train_dataset, **train_kwargs)
    val_loader = create_data_loaders(val_dataset, **val_kwargs)
    test_loader = create_data_loaders(test_dataset, **test_kwargs)

    model = train(model, train_loader, val_loader, loss_criterion, optimizer, device, args.epochs, hook)

    test(model, test_loader, loss_criterion, device)

    model_path = os.path.join(os.environ['SM_MODEL_DIR'], 'model.pth')
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
        "--val-batch-size",
        type=int,
        default=100,
        metavar="N",
        help="input batch size for validation (default: 100)",
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
        "--lr", type=float, default=0.01, metavar="LR", help="learning rate (default: 0.01)"
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
        "--validation",
        type=str,
        default=os.environ.get('SM_CHANNEL_VALIDATION')
    )
    parser.add_argument(
        "--test",
        type=str,
        default=os.environ.get('SM_CHANNEL_TEST')
    )
    args = parser.parse_args()

    main(args)
