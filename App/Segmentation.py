import torch
from PIL import Image
from torchvision import transforms
from .Model import BiSeNet
import numpy as np
import cv2
import os

# Function that change PIL image format or numpy array to tensor and normalize
ToTensor = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(
        (.485, .456, .406),
        (.229, .224, .225)
    )
])

# Function to evaluate image
def Evaluate(image):
    # Init model
    NET = BiSeNet(19)

    # Load state with pre-trained model and available device (CPU)
    NET.load_state_dict(
        torch.load(
            os.path.abspath("App/Model/79999_iter.pth"), 
            map_location=torch.device("cpu")
        )
    )

    # Set model to evaluation mode
    NET.eval()
    
    # Disable gradient calculation
    with torch.no_grad():
        # Change format image to tensor
        img = ToTensor(image)
        
        # Change image dimension 
        img = torch.unsqueeze(img, 0)

        # Test model
        out = NET(img)[0]

        # Change format and image dimension with device cpu
        parsing = out.squeeze(0).cpu().numpy().argmax(0)

        # Return result with numpy format
        return parsing
    
# Function to segmentation with the given image, parts, and colors
def Segmentation(image, parts, colors):
    # Convert image to numpy array
    im = np.array(image)
    imask = im.astype(np.uint8)

    # Evaluate image
    parsing = Evaluate(image)

    # Parsing annotation with evaluated image
    parsing_anno = parsing.astype(np.uint8)

    # Get all annotation shape
    parsing_anno_color = np.zeros((parsing.shape[0], parsing.shape[1], 3))

    # Get annotation shape based on part and draw with color
    for part, color in zip(parts, colors):
        index = np.where(parsing_anno == part)
        parsing_anno_color[index[0], index[1], :] = color

    # Parse data type
    parsing_anno_color = parsing_anno_color.astype(np.uint8)

    # Blend mask with annotation color
    imask = cv2.addWeighted(cv2.cvtColor(imask, cv2.COLOR_RGB2BGR), 0.4, parsing_anno_color, 0.6, 0)

    # Return blend mask image and annotation shape
    return imask, parsing_anno_color

    
    
