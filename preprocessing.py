import cv2
import os
import pandas as pd
from mtcnn import MTCNN
from PIL import Image
from augment import save_augmented_faces

def preprocess_image(image_path, detector):
    image = cv2.imread(image_path)
    if image is None:
        print(f"[WARNING] Gambar tidak ditemukan: {image_path}")
        return None, None

    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    faces = detector.detect_faces(image_rgb)
    return image_rgb, faces

def save_faces(image, faces, output_dir, file_name):
    for idx, face in enumerate(faces):
        x, y, w, h = face['box']
        conf = face['confidence']
        if conf < 0.7:
            continue

        # Ensure no negative coordinates
        x, y = max(0, x), max(0, y)
        face_crop = image[y:y+h, x:x+w]
        face_crop = cv2.resize(face_crop, (160, 160))

        save_path = os.path.join(output_dir, f"{file_name}.jpg")
        image_bgr = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        save_augmented_faces(image_bgr, faces, save_path)

def process_faces(df, detector):
    for _, row in df.iterrows():
        img_path = row["Path"]
        print(img_path)

        image_rgb, faces = preprocess_image(img_path, detector)
        if image_rgb is None:
            continue

        print(f"  - Wajah terdeteksi: {len(faces)}")

        # Extract person and ethnicity from path
        parts = img_path.replace("\\", "/").split("/")
        if len(parts) < 3:
            print(f"[WARNING] Path tidak valid: {img_path}")
            continue

        person_name = parts[-3]
        ethnicity = parts[-2]
        file_name = os.path.basename(img_path).split(".")[0]

        output_dir = os.path.join("dataset", "preprocessed", person_name, ethnicity)
        os.makedirs(output_dir, exist_ok=True)

        save_faces(image_rgb, faces, output_dir, file_name)

if __name__ == "__main__":
    detector = MTCNN()
    df = pd.read_csv("dataset.csv")

    # Clean column names
    df.columns = df.columns.str.strip()
    print("[DEBUG] Kolom CSV ditemukan:", df.columns)

    # Process images and save faces
    process_faces(df, detector)
