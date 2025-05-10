import os
import argparse
import torch
import numpy as np
import cv2
from PIL import Image
import matplotlib.pyplot as plt

from model import ECCVColorizer, SIGGRAPHColorizer
from utils import prepare_input, postprocess_output

def parse_args():
    parser = argparse.ArgumentParser(description='Colorize a black and white image')
    parser.add_argument('--input', type=str, required=True, help='Path to input black and white image')
    parser.add_argument('--output', type=str, default='colorized.jpg', help='Path to save colorized output')
    parser.add_argument('--model_path', type=str, required=True, help='Path to model checkpoint')
    parser.add_argument('--model_type', type=str, default='eccv', choices=['eccv', 'siggraph'], 
                        help='Model type: eccv or siggraph')
    parser.add_argument('--img_size', type=int, default=256, help='Size to process image')
    parser.add_argument('--no_display', action='store_true', help='Do not display the result')
    
    return parser.parse_args()

def colorize_image():
    args = parse_args()
    
    # Check if input file exists
    if not os.path.isfile(args.input):
        print(f"Error: Input file {args.input} does not exist")
        return
    
    # Create output directory if it doesn't exist
    output_dir = os.path.dirname(args.output)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)
    
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Load model
    if args.model_type == 'eccv':
        model = ECCVColorizer().to(device)
        print("Using ECCV model")
    else:
        model = SIGGRAPHColorizer().to(device)
        print("Using SIGGRAPH model")
    
    # Load checkpoint
    if os.path.isfile(args.model_path):
        print(f"Loading checkpoint from {args.model_path}")
        checkpoint = torch.load(args.model_path, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
    else:
        print(f"Error: Model checkpoint {args.model_path} does not exist")
        return
    
    # Set model to evaluation mode
    model.eval()
    
    # Prepare input image
    try:
        L_tensor = prepare_input(args.input, args.img_size, device)
    except Exception as e:
        print(f"Error loading or processing input image: {e}")
        return
    
    # Colorize image
    with torch.no_grad():
        ab_pred = model(L_tensor)
    
    # Convert output to RGB image
    try:
        # Convert tensors to numpy arrays
        L_np = L_tensor.squeeze().cpu().numpy()
        ab_pred_np = ab_pred.squeeze().cpu().numpy()
        
        # Convert L from [-1, 1] to [0, 100]
        L_np = (L_np + 1.0) * 50.0
        
        # Convert ab from [-1, 1] to [-110, 110]
        ab_pred_np = ab_pred_np.transpose(1, 2, 0) * 110.0
        
        # Convert to RGB
        from utils import lab_to_rgb
        colorized_img = lab_to_rgb(L_np, ab_pred_np)
        
        # Convert to uint8
        colorized_img = (colorized_img * 255).astype(np.uint8)
    except Exception as e:
        print(f"Error converting colorized image: {e}")
        return
    
    # Save the result
    try:
        cv2.imwrite(args.output, cv2.cvtColor(colorized_img, cv2.COLOR_RGB2BGR))
        print(f"Colorized image saved to {args.output}")
    except Exception as e:
        print(f"Error saving output image: {e}")
        return
    
    # Display the result if requested
    if not args.no_display:
        # Load original grayscale image for comparison
        orig_img = np.array(Image.open(args.input).convert('L'))
        orig_img_rgb = np.stack([orig_img, orig_img, orig_img], axis=2)
        
        # Create side-by-side comparison
        plt.figure(figsize=(12, 6))
        
        plt.subplot(1, 2, 1)
        plt.imshow(orig_img, cmap='gray')
        plt.title('Original B&W Image')
        plt.axis('off')
        
        plt.subplot(1, 2, 2)
        plt.imshow(colorized_img)
        plt.title('Colorized Image')
        plt.axis('off')
        
        plt.tight_layout()
        plt.show()

if __name__ == '__main__':
    colorize_image() 