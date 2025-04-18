import cv2
import os
from PIL import ImageEnhance, Image
import numpy as np

def crop_face(image, box):
    x, y, w, h = box
    return image[y:y+h, x:x+w]

def adjust_brightness_pil(image_array, factor=1.0):
    image = Image.fromarray(cv2.cvtColor(image_array, cv2.COLOR_BGR2RGB))
    enhancer = ImageEnhance.Brightness(image)
    image_enhanced = enhancer.enhance(factor)
    return cv2.cvtColor(np.array(image_enhanced), cv2.COLOR_RGB2BGR)

def save_augmented_faces(original_image, faces, save_path):
    name = os.path.splitext(os.path.basename(save_path))[0]
    output_dir = os.path.dirname(save_path)

    for idx, face in enumerate(faces):
        x, y, w, h = face['box']
        face_crop = original_image[y:y+h, x:x+w]
        face_crop = cv2.resize(face_crop, (160, 160))

        face_crop = crop_face(original_image, face['box'])
        cv2.imwrite(os.path.join(output_dir, f"{name}_cropped.jpg"), face_crop)


        # Brightness menggunakan PIL agar warna tidak rusak
        bright = adjust_brightness_pil(face_crop, factor=1.5)
        dark = adjust_brightness_pil(face_crop, factor=0.5)

        cv2.imwrite(os.path.join(output_dir, f"{name}_bright.jpg"), bright)
        cv2.imwrite(os.path.join(output_dir, f"{name}_dark.jpg"), dark)
