# Colorize: Black and White Image Colorization

This project implements deep neural networks for automatic colorization of black and white images. The implementation is based on research from Zhang et al. with two models:

1. **ECCV Colorizer**: Based on the "Colorful Image Colorization" paper from ECCV 2016.
2. **SIGGRAPH Colorizer**: Based on the "Real-Time User-Guided Image Colorization with Learned Deep Priors" paper from SIGGRAPH 2017.

## Features

- PyTorch implementation of state-of-the-art colorization models
- Training pipeline with validation and visualization
- Easy inference script for colorizing individual images
- Support for different model architectures

## Installation

Clone the repository and install the required dependencies:

```bash
git clone https://github.com/eclipse1605/colorizer.git
cd colorize
pip install -r requirements.txt
```

## Project Structure

```
colorize/
├── model.py          # Model architectures
├── utils.py          # Utility functions
├── dataset.py        # Dataset class for training
├── train.py          # Training script
├── colorize.py       # Inference script
└── requirements.txt  # Dependencies
```

## Training

To train a model, use the `train.py` script with appropriate arguments:

```bash
python train.py \
  --data_dir path/to/training/images \
  --output_dir outputs \
  --batch_size 16 \
  --img_size 256 \
  --epochs 100 \
  --model_type eccv  # or 'siggraph'
```

### Arguments

- `--data_dir`: Directory containing RGB training images
- `--output_dir`: Directory to save model checkpoints and outputs
- `--batch_size`: Batch size for training
- `--img_size`: Size to resize images during training
- `--epochs`: Number of training epochs
- `--lr`: Learning rate
- `--model_type`: Type of model architecture ('eccv' or 'siggraph')
- `--save_freq`: Frequency to save model checkpoints

## Inference

To colorize a black and white image, use the `colorize.py` script:

```bash
python colorize.py \
  --input path/to/bw/image.jpg \
  --output colorized.jpg \
  --model_path path/to/model/checkpoint.pth \
  --model_type eccv  # or 'siggraph' 
```

### Arguments

- `--input`: Path to input black and white image
- `--output`: Path to save colorized output
- `--model_path`: Path to the trained model checkpoint
- `--model_type`: Type of model architecture ('eccv' or 'siggraph')
- `--img_size`: Size to process image (default: 256)
- `--no_display`: Flag to disable displaying the result

## Model Architectures

### ECCV Colorizer

Based on the "Colorful Image Colorization" paper, this model uses a CNN architecture to predict the a* and b* color channels in the LAB color space from the L* (lightness) channel. The architecture involves:

- Encoder: Progressively downsamples the image while extracting features
- Middle layers: Deep processing of extracted features
- Decoder: Upsampling to generate the final color channels

### SIGGRAPH Colorizer

Based on the "Real-Time User-Guided Image Colorization with Learned Deep Priors" paper, this model enhances the ECCV model with global feature extraction capabilities:

- Low-level features network: Processes the grayscale image to extract local features
- Global features network: Extracts global context information
- Fusion: Combines global and local features for color prediction
- Upsampling: Generates the final color channels

## Technical Details

- Input: L channel of the LAB color space (normalized to [-1, 1])
- Output: a* and b* channels of the LAB color space (normalized to [-1, 1])
- Loss function: L1 loss between predicted and ground truth a*b* channels
- Optimizer: Adam with learning rate scheduling