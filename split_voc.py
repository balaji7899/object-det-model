import os
import shutil

def read_image_ids(file_path):
    with open(file_path, 'r') as f:
        ids = [line.strip() for line in f.readlines()]
    return ids

def copy_images(image_ids, src_dir, dest_dir, extension=".jpg"):
    os.makedirs(dest_dir, exist_ok=True)
    for img_id in image_ids:
        src_path = os.path.join(src_dir, img_id + extension)
        dest_path = os.path.join(dest_dir, img_id + extension)
        if os.path.exists(src_path):
            shutil.copy(src_path, dest_path)
        else:
            print(f"Warning: {src_path} not found.")

def main():
    voc_root = "VOCdevkit/VOC2012"  # Change this if your VOC dataset is elsewhere
    image_sets_dir = os.path.join(voc_root, "ImageSets", "Main")
    jpeg_images_dir = os.path.join(voc_root, "JPEGImages")
    
    # Read train and val image IDs
    train_ids = read_image_ids(os.path.join(image_sets_dir, "train.txt"))
    val_ids   = read_image_ids(os.path.join(image_sets_dir, "val.txt"))
    
    # Define destination directories
    dest_train_dir = "data/source/train"
    dest_val_dir   = "data/source/val"
    
    # Copy images
    print("Copying training images...")
    copy_images(train_ids, jpeg_images_dir, dest_train_dir)
    
    print("Copying validation images...")
    copy_images(val_ids, jpeg_images_dir, dest_val_dir)
    
    print("Dataset splitting completed.")

if __name__ == "__main__":
    main()
