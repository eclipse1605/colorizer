import os
import urllib.request
import zipfile
import shutil
from tqdm import tqdm
from dataset import create_dataloaders

def get_dataloaders(data_dir, batch_size, img_size, num_workers=4, split_ratio=0.9):
    """
    Wrapper function to maintain backward compatibility with older code.
    Creates train and validation dataloaders from a directory of images.
    
    Args:
        data_dir (str): Directory containing images
        batch_size (int): Batch size
        img_size (int): Size to resize images to
        num_workers (int): Number of workers for DataLoader
        split_ratio (float): Ratio of train to validation split
        
    Returns:
        tuple: (train_loader, val_loader)
    """
    return create_dataloaders(data_dir, batch_size, img_size, num_workers, split_ratio)


class DownloadProgressBar(tqdm):
    def update_to(self, b=1, bsize=1, tsize=None):
        if tsize is not None:
            self.total = tsize
        self.update(b * bsize - self.n)


def download_url(url, output_path):
    """Download a file from a URL to the specified output path with a progress bar."""
    with DownloadProgressBar(unit='B', unit_scale=True, miniters=1, desc=url.split('/')[-1]) as t:
        urllib.request.urlretrieve(url, filename=output_path, reporthook=t.update_to)


def prepare_dataset_from_download(dataset_name, output_dir):
    """
    Download and prepare a dataset for training.
    Currently supports: 'div2k'
    
    Args:
        dataset_name (str): Name of the dataset to download ('div2k')
        output_dir (str): Directory to save the downloaded dataset
        
    Returns:
        str: Path to the directory containing the extracted dataset
    """
    os.makedirs(output_dir, exist_ok=True)
    
    if dataset_name.lower() == 'div2k':
        # URLs for DIV2K dataset
        train_url = 'https://data.vision.ee.ethz.ch/cvl/DIV2K/DIV2K_train_HR.zip'
        val_url = 'https://data.vision.ee.ethz.ch/cvl/DIV2K/DIV2K_valid_HR.zip'
        
        # Download paths
        train_zip = os.path.join(output_dir, 'DIV2K_train_HR.zip')
        val_zip = os.path.join(output_dir, 'DIV2K_valid_HR.zip')
        
        # Extract paths
        train_dir = os.path.join(output_dir, 'train')
        val_dir = os.path.join(output_dir, 'val')
        
        # Create directories
        os.makedirs(train_dir, exist_ok=True)
        os.makedirs(val_dir, exist_ok=True)
        
        # Download training data if it doesn't exist
        if not os.path.exists(train_zip):
            print(f"Downloading DIV2K training data to {train_zip}...")
            download_url(train_url, train_zip)
        
        # Download validation data if it doesn't exist
        if not os.path.exists(val_zip):
            print(f"Downloading DIV2K validation data to {val_zip}...")
            download_url(val_url, val_zip)
        
        # Extract training data
        if not os.path.exists(os.path.join(train_dir, 'DIV2K_train_HR')):
            print(f"Extracting DIV2K training data to {train_dir}...")
            with zipfile.ZipFile(train_zip, 'r') as zip_ref:
                zip_ref.extractall(train_dir)
        
        # Extract validation data
        if not os.path.exists(os.path.join(val_dir, 'DIV2K_valid_HR')):
            print(f"Extracting DIV2K validation data to {val_dir}...")
            with zipfile.ZipFile(val_zip, 'r') as zip_ref:
                zip_ref.extractall(val_dir)
        
        # Return the path to the training data
        return os.path.join(train_dir, 'DIV2K_train_HR')
    
    else:
        raise ValueError(f"Unknown dataset: {dataset_name}")
