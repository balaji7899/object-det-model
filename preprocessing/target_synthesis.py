#!/usr/bin/env python3
"""
target_synthesis.py

This script synthesizes three target domain effects (foggy, low-light, and artistic)
from source training and validation images. It saves the processed images into 
designated output directories.

Usage:
    python target_synthesis.py [options]

Example:
    python target_synthesis.py \
      --source_train data/source/train \
      --source_val data/source/val \
      --output_foggy_train data/target/foggy/train \
      --output_foggy_val data/target/foggy/val \
      --output_lowlight_train data/target/lowlight/train \
      --output_lowlight_val data/target/lowlight/val \
      --output_artistic_train data/target/artistic/train \
      --output_artistic_val data/target/artistic/val \
      --fog_intensity 0.5 \
      --lowlight_intensity 0.5 \
      --artistic_intensity 0.5
      ----split_ratio 1.0
"""

import os
import shutil
import random
import argparse
from PIL import Image, ImageEnhance, ImageFilter

def parse_args():
    parser = argparse.ArgumentParser(
        description="Synthesize target domain images from source images by applying domain shift effects."
    )
    # Source directories for train and validation images
    parser.add_argument("--source_train", type=str, default="data/source/train",
                        help="Directory for source training images (default: data/source/train)")
    parser.add_argument("--source_val", type=str, default="data/source/val",
                        help="Directory for source validation images (default: data/source/val)")
    # Output directories for each target effect (train and val)
    parser.add_argument("--output_foggy_train", type=str, default="data/target/foggy/train",
                        help="Output directory for foggy training images (default: data/target/foggy/train)")
    parser.add_argument("--output_foggy_val", type=str, default="data/target/foggy/val",
                        help="Output directory for foggy validation images (default: data/target/foggy/val)")
    parser.add_argument("--output_lowlight_train", type=str, default="data/target/lowlight/train",
                        help="Output directory for low-light training images (default: data/target/lowlight/train)")
    parser.add_argument("--output_lowlight_val", type=str, default="data/target/lowlight/val",
                        help="Output directory for low-light validation images (default: data/target/lowlight/val)")
    parser.add_argument("--output_artistic_train", type=str, default="data/target/artistic/train",
                        help="Output directory for artistic training images (default: data/target/artistic/train)")
    parser.add_argument("--output_artistic_val", type=str, default="data/target/artistic/val",
                        help="Output directory for artistic validation images (default: data/target/artistic/val)")
    # Intensity parameters for effects
    parser.add_argument("--fog_intensity", type=float, default=0.5,
                        help="Intensity of the fog effect (default: 0.5)")
    parser.add_argument("--lowlight_intensity", type=float, default=0.5,
                        help="Intensity of the low-light effect (default: 0.5)")
    parser.add_argument("--artistic_intensity", type=float, default=0.5,
                        help="Intensity of the artistic effect (default: 0.5)")
    # Split ratio for train/validation
    parser.add_argument("--split_ratio", type=float, default=1.0,
                        help="Proportion of images to use for training (default: 1.0)")
    args = parser.parse_args()
    return args

# Define effect functions
def apply_fog_effect(image, intensity=0.5):
    """
    Apply a foggy effect by blurring the image and reducing its contrast.
    """
    # Apply Gaussian blur to simulate fog
    foggy_image = image.filter(ImageFilter.GaussianBlur(radius=5))
    # Reduce contrast: factor less than 1 reduces contrast
    enhancer = ImageEnhance.Contrast(foggy_image)
    foggy_image = enhancer.enhance(1 - intensity)
    return foggy_image

def apply_lowlight_effect(image, intensity=0.5):
    """
    Apply a low-light effect by reducing the brightness of the image.
    """
    enhancer = ImageEnhance.Brightness(image)
    lowlight_image = enhancer.enhance(1 - intensity)
    return lowlight_image

def apply_artistic_effect(image, intensity=0.5):
    """
    Apply an artistic effect by boosting color saturation.
    """
    enhancer = ImageEnhance.Color(image)
    artistic_image = enhancer.enhance(1 + intensity)
    return artistic_image

def process_and_save_images(source_dir, output_dir, effect_func, intensity, split_ratio):
    """
    Process images from the source directory using the specified effect function,
    then split and save them into output training and validation directories.
    """
    os.makedirs(output_dir, exist_ok=True)
    # List all image files (assume jpg and png)
    images = [f for f in os.listdir(source_dir) if f.lower().endswith(('.jpg', '.png'))]
    random.shuffle(images)
    # split_point = int(len(images) * split_ratio)
    # train_images = images[:split_point]
    # val_images = images[split_point:]
    
    # Create subdirectories for train and val
    # train_dir = os.path.join(output_dir, "train")
    # val_dir = os.path.join(output_dir, "val")
    # os.makedirs(train_dir, exist_ok=True)
    # os.makedirs(val_dir, exist_ok=True)
    
    # Process training images
    for img_name in images:
        try:
            img_path = os.path.join(source_dir, img_name)
            image = Image.open(img_path).convert("RGB")
            transformed_image = effect_func(image, intensity)
            transformed_image.save(os.path.join(output_dir, img_name))
        except Exception as e:
            print(f"Error processing {img_name} in train set: {e}")
    
    # Process validation images
    # for img_name in val_images:
    #     try:
    #         img_path = os.path.join(source_dir, img_name)
    #         image = Image.open(img_path).convert("RGB")
    #         transformed_image = effect_func(image, intensity)
    #         transformed_image.save(os.path.join(val_dir, img_name))
    #     except Exception as e:
    #         print(f"Error processing {img_name} in val set: {e}")
    
    print(f"Processed {len(images)} images from {source_dir} into {output_dir}.")

def main():
    args = parse_args()
    
    # Process each target domain separately
    print("Processing foggy target domain...")
    process_and_save_images(
        source_dir=args.source_train, 
        output_dir=args.output_foggy_train,
        # output_dir=os.path.dirname(args.output_foggy_train), 
        effect_func=apply_fog_effect, 
        intensity=args.fog_intensity, 
        split_ratio=args.split_ratio
    )
    process_and_save_images(
        source_dir=args.source_val, 
        output_dir=args.output_foggy_val,
        # output_dir=os.path.dirname(args.output_foggy_val), 
        effect_func=apply_fog_effect, 
        intensity=args.fog_intensity, 
        split_ratio=args.split_ratio
    )
    
    print("Processing low-light target domain...")
    process_and_save_images(
        source_dir=args.source_train, 
        output_dir=args.output_lowlight_train,
        # output_dir=os.path.dirname(args.output_lowlight_train), 
        effect_func=apply_lowlight_effect, 
        intensity=args.lowlight_intensity, 
        split_ratio=args.split_ratio
    )
    process_and_save_images(
        source_dir=args.source_val, 
        output_dir=args.output_lowlight_val,
        # output_dir=os.path.dirname(args.output_lowlight_val), 
        effect_func=apply_lowlight_effect, 
        intensity=args.lowlight_intensity, 
        split_ratio=args.split_ratio
    )
    
    print("Processing artistic target domain...")
    process_and_save_images(
        source_dir=args.source_train, 
        output_dir=args.output_artistic_train,
        # output_dir=os.path.dirname(args.output_artistic_train), 
        effect_func=apply_artistic_effect, 
        intensity=args.artistic_intensity, 
        split_ratio=args.split_ratio
    )
    process_and_save_images(
        source_dir=args.source_val, 
        output_dir=args.output_artistic_val,
        # output_dir=os.path.dirname(args.output_artistic_val), 
        effect_func=apply_artistic_effect, 
        intensity=args.artistic_intensity, 
        split_ratio=args.split_ratio
    )
    
    print("Target domain synthesis complete.")

if __name__ == "__main__":
    main()
