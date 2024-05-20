# import torch
# import torchvision.models as models
# import torchvision.transforms as transforms
# from PIL import Image
#
#
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# print(f"Using device: {device}")
#
# from torchvision.models import resnet152, ResNet152_Weights
# model = resnet152(weights=ResNet152_Weights.DEFAULT).to(device)
#
# # # Load a pre-trained ResNet model and move it to the selected device
# # model = models.resnet152(pretrained=True).to(device)
#
# # Define the image transformation
# transform = transforms.Compose([
#     transforms.Resize(256),
#     transforms.CenterCrop(224),
#     transforms.ToTensor(),
#     transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
# ])
#
#
# def load_and_transform_images(image_paths):
#     images = []
#     for image_path in image_paths:
#         image = Image.open(image_path)
#         image = image.convert('RGB')
#         image = transform(image)
#         images.append(image)
#     # Stack all images into a single tensor
#     images_tensor = torch.stack(images)
#     return images_tensor
#
#
# def embed_images(image_paths):
#     # Load and transform images
#     images_tensor = load_and_transform_images(image_paths)
#
#     # Get the embeddings (features from the last layer)
#     with torch.no_grad():
#         embeddings = model(images_tensor)
#
#     return embeddings
#
# folder_path = '/home/zhilin.wang/paper/selfea/data/DBP100k/fr_en/feature/ent1'
# # Example usage
# #get all .png files
# import os
# image_paths = [os.path.join(folder_path, f) for f in os.listdir(folder_path) if f.endswith('.png')]
# image_paths = image_paths[:1000]
# import time
# start = time.time()
# embeddings = embed_images(image_paths)
# print("Time taken to embed {} images: {:.2f} seconds".format(len(image_paths), time.time() - start))
# print(embeddings)
import torch
import torchvision.models as models
import torchvision.transforms as transforms
from PIL import Image
import os
import time

# Set the device to GPU if available, otherwise use CPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Load a pre-trained ResNet model and move it to the selected device
from torchvision.models import resnet152, ResNet152_Weights
model = resnet152(weights=ResNet152_Weights.DEFAULT).to(device)

# Define the image transformation
transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

def load_and_transform_images(image_paths):
    images = []
    for image_path in image_paths:
        image = Image.open(image_path)
        image = image.convert('RGB')
        image = transform(image)
        images.append(image)
    # Stack all images into a single tensor and move it to the GPU
    images_tensor = torch.stack(images).to(device)
    return images_tensor

def embed_images(image_paths):
    # Load and transform images
    images_tensor = load_and_transform_images(image_paths)
    # Get the embeddings (features from the last layer)
    with torch.no_grad():
        embeddings = model(images_tensor)
    return embeddings

folder_path = '/home/zhilin.wang/paper/selfea/data/DBP100k/fr_en/feature/ent1'
# Get all .png files
image_paths = [os.path.join(folder_path, f) for f in os.listdir(folder_path) if f.endswith('.png')]
image_paths = image_paths[:1000]  # Limit to first 1000 images
start = time.time()
embeddings = embed_images(image_paths)
print("Time taken to embed {} images: {:.2f} seconds".format(len(image_paths), time.time() - start))
print(embeddings)
