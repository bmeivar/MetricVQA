'''
Prepare the Objectron dataset for training according to our format
of query+image -> answer.

This tool treats Google Objectron dataset - It extracts the object
dimensions and one frame per video, and creates queries about the dimensions.

The dataset will be stored as csv file + images directory
where the csv file contains the relevant queries for each image.

Environment installation:
pip install tensorflow opencv-python numpy tensorflow-io pandas
'''


import argparse
import os
import pandas as pd
import tensorflow as tf
import cv2
import numpy as np


CLASSES = ['bike', 'book', 'bottle', 'camera', 'cereal_box', 'chair', 'cup', 'laptop', 'shoe']
SHARDS_TO_PROCESS = 3
IMAGES_PER_SHARD = 200
BLUR_THRESHOLD = 200


def is_blurry(image_np) :
    """
    Returns True if the image is considered blurry.
    Uses the Variance of Laplacian method.
    """
    gray = cv2.cvtColor(image_np, cv2.COLOR_BGR2GRAY)
    score = cv2.Laplacian(gray, cv2.CV_64F).var()
    return score < BLUR_THRESHOLD


def parse_record(proto):
    features = {
        'image/encoded': tf.io.FixedLenFeature([], tf.string),
        'image/filename': tf.io.FixedLenFeature([], tf.string, default_value=""),
        'object/scale': tf.io.VarLenFeature(tf.float32),
        'object/category': tf.io.FixedLenFeature([], tf.string, default_value="unknown"),
        'point_2d': tf.io.VarLenFeature(tf.float32)
    }

    try:
        parsed = tf.io.parse_single_example(proto, features)
        
        points = tf.sparse.to_dense(parsed['point_2d']).numpy()
        
        # If we don't have enough points (should be 27 values for 9 points), skip
        if len(points) < 27 :
            return None, None, "missing_points"

        # Calculate Bounding Box manually
        # Indices (0, 3, 6) are X coordinates
        # Indices (1, 4, 7) are Y coordinates
        x_coords = points[0::3]
        y_coords = points[1::3]
        xmin = np.min(x_coords)
        xmax = np.max(x_coords)
        ymin = np.min(y_coords)
        ymax = np.max(y_coords)

        # Check Visibility (Cut-off check)
        # 0.0 to 1.0 is the frame. If min < 0 or max > 1, it's outside.
        if xmin < 0.01 or ymin < 0.01 or xmax > 0.99 or ymax > 0.99 :
            return None, None, "cut_off"

        # Extract dimensions and image
        dims = tf.sparse.to_dense(parsed['object/scale']).numpy()
        image_bytes = parsed['image/encoded'].numpy()
        nparr = np.frombuffer(image_bytes, np.uint8)
        image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        filename_str = parsed['image/filename'].numpy().decode('utf-8')

        if len(dims) != 3 :
            return None, None, "invalid_dims"

        # Check Blurriness
        if is_blurry(image):
            return None, None, "blurry"

        return dims, image, filename_str

    except Exception:
        return None, None, None


def main(out_dir_imgs : str, out_file_qa : str) :

    if not os.path.exists(out_dir_imgs) :
        os.makedirs(out_dir_imgs)
    
    dataset_qa = []
    global_counter = 0
    gcs_bucket = "gs://objectron/v1/records_shuffled"

    for category in CLASSES :
        print(f"Processing Class: {category}")
        
        seen_video_ids = set()

        # search for files using GCS glob
        search_pattern = f"{gcs_bucket}/{category}/{category}_train*"
        
        try :
            files = tf.io.gfile.glob(search_pattern)
        except tf.errors.UnimplementedError :
            print("[ERROR] TensorFlow GCS support missing. Install tensorflow-io.")
            continue
            
        if not files :
            print(f"[WARNING] No train files. Trying 'test' split...")
            files = tf.io.gfile.glob(f"{gcs_bucket}/{category}/{category}_test*")
        
        if not files :
            print(f"[FAIL] No files found for {category}. Skipping.")
            continue

        files = sorted(files)
        files_to_process = files[:SHARDS_TO_PROCESS]
        print(f"Found {len(files)} shards. Processing first {len(files_to_process)}...")

        for file_path in files_to_process :
            filename = file_path.split("/")[-1]
            print(f"Parsing {filename}...")
            
            try :
                dataset = tf.data.TFRecordDataset(file_path, compression_type="")
                
                file_image_count = 0
                
                # Take images from this shard
                for raw_record in dataset :
                    
                    # STOP condition: Have we collected enough?
                    if file_image_count >= IMAGES_PER_SHARD :
                        break 
                    
                    dims, image, filename_str = parse_record(raw_record)
                    
                    if dims is None or image is None or filename_str is None :
                        print(f"Skipping record due to parsing error or validation "
                              f"failure: {filename_str}")
                        continue
                    
                    # Validate dimensions (filter out bad data/zeros)
                    if np.sum(dims) <= 0.01 :
                        continue

                    # Validate video_id uniqueness (to avoid duplicates across shards)
                    # Filename format is usually: "category/batch/video_id"
                    # Example: "bike/batch-4/55"
                    if filename_str in seen_video_ids:
                        print(f"Skipping record due to duplicate video_id: {filename_str}")
                        continue # We already have a frame from this video/object! SKIP.
            
                    seen_video_ids.add(filename_str)

                    img_filename = f"{category}__{str(global_counter).zfill(4)}.png"
                    img_path = os.path.join(out_dir_imgs, img_filename)
                    
                    cv2.imwrite(img_path, image)
                    
                    w, h, d = float(dims[0] * 100), float(dims[1] * 100), float(dims[2] * 100)
                    
                    dataset_qa.append({"filename": img_filename,
                                       "query": f"What is the width of this {category}?",
                                       "answer": w})
                    dataset_qa.append({"filename": img_filename,
                                       "query": f"What is the height of this {category}?",
                                       "answer": h})
                    dataset_qa.append({"filename": img_filename,
                                       "query": f"What is the depth of this {category}?",
                                       "answer": d})
                    
                    global_counter += 1
                    file_image_count += 1
                
                print(f"Extracted {file_image_count} images.")
                
            except Exception as e :
                print(f"Error reading file: {e}")
                continue

    df = pd.DataFrame(dataset_qa)
    df.to_csv(out_file_qa, index=False)

    print(f"\nDone! Total Images: {global_counter}")


if __name__ == "__main__" :

    parser = argparse.ArgumentParser(description="Prepare Objectron dataset for MetricVQA.")
    parser.add_argument("--out_dir", type=str, default="../data/processed_data/my_objectron_dataset", help="Directory to save the extracted dataset.")
    args = parser.parse_args()
    
    main(out_dir_imgs=os.path.join(args.out_dir, "images"), out_file_qa=os.path.join(args.out_dir, "questions.csv"))