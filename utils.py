import numpy as np
import torch
import cv2
import matplotlib.pyplot as plt
from skimage import color
from PIL import Image
import os
from math import ceil

def load_img(path):
    """Load an image and convert to a numpy array"""
    img = Image.open(path)
    return np.array(img)

def resize_img(img, target_size=256):
    """Resize image to target size"""
    if img.shape[0] != target_size or img.shape[1] != target_size:
        return cv2.resize(img, (target_size, target_size))
    return img

def rgb_to_lab(img_rgb):
    """Convert RGB image to LAB color space"""
    # Convert to float32
    img_rgb = img_rgb.astype(np.float32) / 255.0
    # Convert to LAB
    img_lab = color.rgb2lab(img_rgb)
    # Return L channel and ab channels
    L = img_lab[:, :, 0]
    ab = img_lab[:, :, 1:]
    return L, ab

def lab_to_rgb(L, ab):
    """Convert L and ab channels to RGB color space"""
    # Reshape L to match ab shape
    L = L[:, :, np.newaxis]
    # Concatenate L and ab
    img_lab = np.concatenate((L, ab), axis=2)
    # Convert to RGB
    img_rgb = color.lab2rgb(img_lab)
    # Clip values to [0, 1]
    img_rgb = np.clip(img_rgb, 0, 1)
    return img_rgb

def preprocess_img(img_path, img_size=256):
    """Load, resize, and convert an image to Lab colorspace"""
    # Load image
    img = load_img(img_path)
    
    # Convert grayscale to RGB if needed
    if len(img.shape) < 3:
        img = np.stack([img, img, img], axis=2)
    elif img.shape[2] == 4:  # Handle RGBA
        img = img[:, :, :3]
    
    # Resize
    img = resize_img(img, img_size)
    
    # Convert to LAB
    L, ab = rgb_to_lab(img)
    
    # Normalize L to range [-1, 1]
    L = L / 50.0 - 1.0
    
    # Normalize ab to range [-1, 1]
    ab = ab / 110.0
    
    return L, ab

def lab_tensors_to_rgb(L_tensor, ab_tensor):
    """Convert L and ab tensors to RGB image
    
    Args:
        L_tensor (torch.Tensor): Lightness tensor with shape (B, 1, H, W)  
        ab_tensor (torch.Tensor): ab color tensor with shape (B, 2, H, W)
        
    Returns:
        numpy.ndarray: RGB image with shape (H, W, 3) and values in range [0, 1]
    """
    # Convert tensors to numpy arrays
    L = tensor_to_numpy(L_tensor.squeeze(0))  # (H, W)
    ab = tensor_to_numpy(ab_tensor.squeeze(0).transpose(1, 2, 0))  # (H, W, 2)
    
    # Denormalize L from [-1, 1] to [0, 100]
    L = (L + 1.0) * 50.0
    
    # Denormalize ab from [-1, 1] to [-128, 127]
    ab = ab * 110.0
    
    # Convert LAB to RGB
    rgb_img = lab_to_rgb(L, ab)
    
    return rgb_img

def postprocess_output(L, ab_pred):
    """Convert model output back to RGB image"""
    # Denormalize L
    L = (L + 1.0) * 50.0
    
    # Denormalize ab
    ab_pred = ab_pred * 110.0
    
    # Convert back to RGB
    rgb_image = lab_to_rgb(L, ab_pred)
    
    # Convert to uint8
    rgb_image = (rgb_image * 255).astype(np.uint8)
    
    return rgb_image

def tensor_to_numpy(tensor):
    """Convert a PyTorch tensor to numpy array"""
    return tensor.detach().cpu().numpy()

def numpy_to_tensor(array, device='cuda' if torch.cuda.is_available() else 'cpu'):
    """Convert a numpy array to PyTorch tensor"""
    return torch.from_numpy(array).to(device)

def prepare_input(img_path, img_size=256, device='cuda' if torch.cuda.is_available() else 'cpu'):
    """Prepare an image for model input"""
    # Preprocess image
    L, _ = preprocess_img(img_path, img_size)
    
    # Convert to tensor
    L_tensor = torch.from_numpy(L).unsqueeze(0).unsqueeze(0).float().to(device)
    
    return L_tensor

def save_result(original_img_path, colorized_img, output_path):
    """Save original grayscale and colorized images side by side"""
    # Load original image
    original = cv2.imread(original_img_path, 0)  # Read as grayscale
    original = cv2.resize(original, (colorized_img.shape[1], colorized_img.shape[0]))
    
    # Convert grayscale to RGB for side-by-side display
    original_rgb = cv2.cvtColor(original, cv2.COLOR_GRAY2RGB)
    
    # Create side-by-side comparison
    comparison = np.hstack([original_rgb, colorized_img])
    
    # Save the comparison
    cv2.imwrite(output_path, cv2.cvtColor(comparison, cv2.COLOR_RGB2BGR))

def create_directories(*dirs):
    """Create directories if they don't exist"""
    for dir_path in dirs:
        if not os.path.exists(dir_path):
            os.makedirs(dir_path, exist_ok=True)
            print(f"Created directory: {dir_path}")

def calculate_metrics(pred, target):
    """Calculate metrics for colorization results
    
    Args:
        pred (torch.Tensor): Predicted ab channels
        target (torch.Tensor): Ground truth ab channels
        
    Returns:
        dict: Dictionary of metrics
    """
    # Calculate L1 loss
    l1_loss = torch.mean(torch.abs(pred - target))
    
    # Calculate PSNR (Peak Signal-to-Noise Ratio)
    mse = torch.mean((pred - target) ** 2)
    psnr = 20 * torch.log10(2.0 / torch.sqrt(mse))  # 2.0 because ab values are in [-1, 1]
    
    return {
        'l1_loss': l1_loss.item(),
        'psnr': psnr.item()
    }

def pad_tensor(x, div_by=64):
    """Pad tensor to be divisible by div_by"""
    h, w = x.shape[-2:]
    new_h = ceil(h / div_by) * div_by
    new_w = ceil(w / div_by) * div_by
    
    padding_h = new_h - h
    padding_w = new_w - w
    padding = [padding_w // 2, padding_w - padding_w // 2, padding_h // 2, padding_h - padding_h // 2]
    
    return torch.nn.functional.pad(x, padding, mode='reflect'), padding

def unpad_tensor(x, padding):
    """Remove padding from tensor"""
    h, w = x.shape[-2:]
    padding_left, padding_right, padding_top, padding_bottom = padding
    return x[..., padding_top:h-padding_bottom, padding_left:w-padding_right]

def visualize_colorization(L, ab_pred, ab_true=None, save_path=None):
    """Visualize colorization results and optionally save to file"""
    fig = visualize_result(L, ab_pred, ab_true)
    
    if save_path is not None:
        fig.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close(fig)
    else:
        plt.show()
    
    return fig

def visualize_result(L, ab_pred, ab_true=None):
    """Visualize colorization results"""
    # Denormalize L
    L_np = tensor_to_numpy(L.squeeze(0))
    L_np = (L_np + 1.0) * 50.0
    
    # Denormalize ab predictions
    ab_pred_np = tensor_to_numpy(ab_pred.squeeze(0).transpose(1, 2, 0))
    ab_pred_np = ab_pred_np * 110.0
    
    # Convert predicted color to RGB
    pred_rgb = lab_to_rgb(L_np, ab_pred_np)
    
    # Set up the figure
    fig, axes = plt.subplots(1, 3 if ab_true is not None else 2, figsize=(12, 4))
    
    # Display grayscale image
    axes[0].imshow(L_np, cmap='gray')
    axes[0].set_title('Grayscale (L)')
    axes[0].axis('off')
    
    # Display colorized image
    axes[1].imshow(pred_rgb)
    axes[1].set_title('Colorized (Prediction)')
    axes[1].axis('off')
    
    # Display ground truth if available
    if ab_true is not None:
        # Denormalize ground truth ab
        ab_true_np = tensor_to_numpy(ab_true.squeeze(0).transpose(1, 2, 0))
        ab_true_np = ab_true_np * 110.0
        
        # Convert ground truth to RGB
        true_rgb = lab_to_rgb(L_np, ab_true_np)
        
        # Display ground truth
        axes[2].imshow(true_rgb)
        axes[2].set_title('Ground Truth')
        axes[2].axis('off')
    
    plt.tight_layout()
    return fig