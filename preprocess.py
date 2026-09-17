import os
import sys
import numpy as np
from PIL import Image

BASE_DIR = "./archive"
OUTPUT_DIR = "./processed"
TARGET_SIZE = (224, 224)
GRAYSCALE = True
NORMALIZE = True

def preprocess_image(img_path):
    """Prétraite une seule image."""
    img = Image.open(img_path)
    
    if GRAYSCALE:
        img = img.convert("L")
    else:
        img = img.convert("RGB")
    
    img = img.resize(TARGET_SIZE, Image.BILINEAR)
    img_array = np.array(img, dtype=np.float32)
    
    if NORMALIZE:
        img_array = img_array / 255.0
    
    return img_array

def process_split(split_name):
    """Traite Training ou Testing en préservant les classes."""
    input_split = os.path.join(BASE_DIR, split_name)
    output_split = os.path.join(OUTPUT_DIR, split_name)
    
    if not os.path.exists(input_split):
        print(f"Warning: {input_split} not found.", file=sys.stderr)
        return 0, 0
    
    processed = 0
    errors = 0
    
    for class_name in os.listdir(input_split):
        class_input = os.path.join(input_split, class_name)
        class_output = os.path.join(output_split, class_name)
        
        if not os.path.isdir(class_input):
            continue
        
        os.makedirs(class_output, exist_ok=True)
        
        for filename in os.listdir(class_input):
            if not filename.lower().endswith(".jpg"):
                continue
            
            img_path = os.path.join(class_input, filename)
            try:
                img_array = preprocess_image(img_path)
                out_name = os.path.splitext(filename)[0] + ".npy"
                np.save(os.path.join(class_output, out_name), img_array)
                
                processed += 1
                if processed % 100 == 0:
                    print(f"[{split_name}/{class_name}] {processed} images...")
            except Exception as e:
                print(f"Error: {img_path}: {e}", file=sys.stderr)
                errors += 1
    
    return processed, errors

def main():
    if not os.path.exists(BASE_DIR):
        print(f"Error: '{BASE_DIR}' not found.", file=sys.stderr)
        return 84
    
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    total_processed = 0
    total_errors = 0
    
    for split in ["Training", "Testing"]:
        print(f"\n=== {split} ===")
        processed, errors = process_split(split)
        total_processed += processed
        total_errors += errors
        print(f"→ {processed} processed, {errors} errors")
    
    print(f"\n=== TOTAL: {total_processed} processed, {total_errors} errors ===")
    return 0

if __name__ == "__main__":
    sys.exit(main())
