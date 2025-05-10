import os 
import argparse 
import torch 
import numpy as np 
import cv2 
from PIL import Image 
import matplotlib .pyplot as plt 
from tqdm import tqdm 

from model import ECCVColorizer ,SIGGRAPHColorizer 
from utils import prepare_input ,postprocess_output ,lab_to_rgb 

def colorize_image (input_path ,output_path ,model_path ,model_type ='eccv',img_size =256 ,display =False ):
    """Colorize a grayscale image using the trained model"""

    device =torch .device ('cuda'if torch .cuda .is_available ()else 'cpu')


    if model_type =='eccv':
        model =ECCVColorizer ().to (device )
    else :
        model =SIGGRAPHColorizer ().to (device )


    checkpoint =torch .load (model_path ,map_location =device )
    model .load_state_dict (checkpoint ['model_state_dict'])


    model .eval ()


    L_tensor =prepare_input (input_path ,img_size ,device )


    with torch .no_grad ():
        ab_pred =model (L_tensor )


    L_np =L_tensor .squeeze ().cpu ().numpy ()
    ab_pred_np =ab_pred .squeeze ().cpu ().numpy ()


    L_np =(L_np +1.0 )*50.0 


    ab_pred_np =ab_pred_np .transpose (1 ,2 ,0 )*110.0 


    colorized_img =lab_to_rgb (L_np ,ab_pred_np )


    colorized_img =(colorized_img *255 ).astype (np .uint8 )


    cv2 .imwrite (output_path ,cv2 .cvtColor (colorized_img ,cv2 .COLOR_RGB2BGR ))

    return colorized_img 

def batch_test (test_dir ,output_dir ,checkpoints_dir ,model_type ='eccv'):
    """Test colorization on multiple images using different checkpoints"""
    os .makedirs (output_dir ,exist_ok =True )


    test_images =[f for f in os .listdir (test_dir )if f .endswith ('_gray.png')]


    checkpoints =[
    ('best_model.pth','best'),
    ('final_model.pth','final')
    ]

    print (f"Testing {len (test_images )} images with {len (checkpoints )} model checkpoints...")


    for img_name in test_images :
        base_name =img_name .replace ('_gray.png','')
        input_path =os .path .join (test_dir ,img_name )


        original_path =os .path .join (test_dir ,f"{base_name }_color.png")
        original_img =np .array (Image .open (original_path ).convert ('RGB'))


        gray_img =np .array (Image .open (input_path ).convert ('L'))


        plt .figure (figsize =(15 ,5 ))


        plt .subplot (1 ,len (checkpoints )+2 ,1 )
        plt .imshow (gray_img ,cmap ='gray')
        plt .title ('Grayscale Input')
        plt .axis ('off')


        plt .subplot (1 ,len (checkpoints )+2 ,2 )
        plt .imshow (original_img )
        plt .title ('Original Color')
        plt .axis ('off')


        for i ,(checkpoint_name ,checkpoint_label )in enumerate (checkpoints ):
            model_path =os .path .join (checkpoints_dir ,checkpoint_name )
            output_name =f"{base_name }_{checkpoint_label }.png"
            output_path =os .path .join (output_dir ,output_name )

            print (f"Colorizing {img_name } with {checkpoint_name }...")
            colorized =colorize_image (input_path ,output_path ,model_path ,model_type )


            plt .subplot (1 ,len (checkpoints )+2 ,i +3 )
            plt .imshow (colorized )
            plt .title (f'Model: {checkpoint_label }')
            plt .axis ('off')


        comparison_path =os .path .join (output_dir ,f"{base_name }_comparison.png")
        plt .tight_layout ()
        plt .savefig (comparison_path ,dpi =150 )
        plt .close ()

        print (f"Saved comparison to {comparison_path }")

    print ("Testing complete!")

if __name__ =='__main__':
    parser =argparse .ArgumentParser (description ='Test colorization models')
    parser .add_argument ('--test_dir',type =str ,default ='test_images',help ='Directory with test images')
    parser .add_argument ('--output_dir',type =str ,default ='test_results',help ='Directory to save results')
    parser .add_argument ('--checkpoints_dir',type =str ,default ='outputs/div2k_eccv/checkpoints',help ='Directory with model checkpoints')
    parser .add_argument ('--model_type',type =str ,default ='eccv',choices =['eccv','siggraph'],help ='Model type')

    args =parser .parse_args ()

    batch_test (args .test_dir ,args .output_dir ,args .checkpoints_dir ,args .model_type )
