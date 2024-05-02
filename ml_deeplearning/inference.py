import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as transforms
import os
import io
import json
from PIL import Image


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def net():
    model = models.resnet18(pretrained=True)

    for param in model.parameters():
        param.requires_grad = False

    num_features = model.fc.in_features
    model.fc = nn.Sequential(
                   nn.Linear(num_features, 133))
    return model


def model_fn(model_dir):
    """Load saved model from file"""
    file_path = os.path.join(model_dir, "model.pth")
    model = net().to(device)
    model.load_state_dict(torch.load(file_path))
    model.eval()
    return model


# Define the transformation pipeline
img_transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])


# Define the input function
def input_fn(request_body, content_type):
    # Check if the content type is 'image/jpeg'
    if content_type == 'image/jpeg':
        # Open the image using PIL
        try:
            image = Image.open(io.BytesIO(request_body))
            # Apply transformations
            image = img_transform(image)
            return image
        except Exception as e:
            raise ValueError(f"Error opening JPEG image: {e}")
    else:
        # Raise an error if content type is not 'image/jpeg'
        raise ValueError(f"Unsupported content type: {content_type}")


def output_fn(predictions, content_type):
    if content_type == 'application/json':
        try:
            # Convert predictions to JSON-serializable format
            res = predictions.cpu().numpy().tolist()
            return json.dumps(res), content_type
        except Exception as e:
            raise e
    else:
        raise ValueError(f"Unsupported content type: {content_type}")


def predict_fn(input_object, model):
    try:
        with torch.no_grad():
            prediction = model(input_object.unsqueeze(0))
        return prediction
    except Exception as e:
        raise e

