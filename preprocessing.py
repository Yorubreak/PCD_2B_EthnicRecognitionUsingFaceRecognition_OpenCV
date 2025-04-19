import cv2
import os
import csv
import pandas as pd
from mtcnn import MTCNN
from augment import save_augmented_faces

def preprocess_image(image_path, detector):
    image = cv2.imread(image_path)
    if image is None:
        print(f"[WARNING] Gambar tidak ditemukan: {image_path}")
        return None, None

    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    faces = detector.detect_faces(image_rgb)
    return image_rgb, faces

def extract_labels_from_path(image_path):
    filename = os.path.basename(image_path).split('.')[0]
    
    filename = filename.replace("_", ",")
    
    parts = filename.split(',')
    
    # Pastikan ada setidaknya 5 bagian dalam nama file
    if len(parts) >= 5:
        return parts[0], parts[1], parts[2], parts[3], parts[4]
    else:
        return "Unknown", "Unknown", "Unknown", "Unknown", "Unknown"


def save_to_csv(csv_path, image_path, nama, suku, ekspresi, sudut, pencahayaan):
    image_path = image_path.replace('/', '\\')  # format path Windows
    row = [image_path, nama, suku, ekspresi, sudut, pencahayaan]

    file_exists = os.path.isfile(csv_path)
    with open(csv_path, mode='a', newline='', encoding='utf-8') as csvfile:
        writer = csv.writer(csvfile)
        if not file_exists:
            writer.writerow(['Path', 'Nama', 'Suku', 'Ekspresi', 'Sudut', 'Pencahayaan'])
        writer.writerow(row)

def save_to_training_csv(csv_path, image_path, nama, suku, ekspresi, sudut, pencahayaan):
    image_path = image_path.replace('/', '\\')  # format path Windows
    row = [image_path, nama, suku, ekspresi, sudut, pencahayaan]

    file_exists = os.path.isfile(csv_path)
    with open(csv_path, mode='a', newline='', encoding='utf-8') as csvfile:
        writer = csv.writer(csvfile)
        if not file_exists:
            writer.writerow(['Path', 'Nama', 'Suku', 'Ekspresi', 'Sudut', 'Pencahayaan'])
        writer.writerow(row)



def save_faces(image, faces, output_dir, file_name, label_info, csv_path, training_csv_path=None):
    for idx, face in enumerate(faces):
        x, y, w, h = face['box']
        conf = face['confidence']
        if conf < 0.7:
            continue

        x, y = max(0, x), max(0, y)
        face_crop = image[y:y+h, x:x+w]
        face_crop = cv2.resize(face_crop, (160, 160))

        save_path = os.path.join(output_dir, f"{file_name}.jpg")
        image_bgr = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

        augmented_paths = save_augmented_faces(image_bgr, [face], save_path)

        for aug_path in augmented_paths:
            save_to_csv(csv_path, aug_path, *label_info)

            aug_name = os.path.basename(aug_path)
            if 'sharpened' in aug_name or 'contrast' in aug_name or 'saturation' in aug_name or 'noise' in aug_name or 'flip' in aug_name:
                save_to_training_csv(training_csv_path, aug_path, *label_info)



def process_faces(df, detector, csv_path, training_csv_path):
    for _, row in df.iterrows():
        img_path = row["Path"]
        print(f"[INFO] Memproses: {img_path}")

        image_rgb, faces = preprocess_image(img_path, detector)
        if image_rgb is None:
            continue

        print(f"  - Wajah terdeteksi: {len(faces)}")

        parts = img_path.replace("\\", "/").split("/")
        if len(parts) < 3:
            print(f"[WARNING] Path tidak valid: {img_path}")
            continue

        person_name = parts[-3]
        ethnicity = parts[-2]
        file_name = os.path.basename(img_path).split(".")[0]

        output_dir = os.path.join("dataset", "preprocessed", person_name, ethnicity)
        os.makedirs(output_dir, exist_ok=True)

        # Ekstrak label dari nama file
        label_info = extract_labels_from_path(img_path)

        # Simpan wajah dan catat ke CSV
        save_faces(image_rgb, faces, output_dir, file_name, label_info, csv_path, training_csv_path)

if __name__ == "__main__":
    detector = MTCNN()
    csv_path = "dataset.csv"
    training_csv_path = "training.csv"

    # Load dataset CSV yang berisi path gambar awal
    df = pd.read_csv(csv_path)
    df.columns = df.columns.str.strip()

    # Hapus CSV lama dan buat baru
    if os.path.exists(csv_path):
        os.remove(csv_path)
    
    if os.path.exists(training_csv_path):
        os.remove(training_csv_path)

    print("[DEBUG] Kolom CSV ditemukan:", df.columns)
    process_faces(df, detector, csv_path, training_csv_path)
    print("Proses selesai. Semua hasil tercatat dalam dataset.csv dan training.csv")


