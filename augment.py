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

def sharpen_image(image_array):
    kernel = np.array([[0, -1, 0],
                       [-1, 5,-1],
                       [0, -1, 0]])
    return cv2.filter2D(image_array, -1, kernel)

def adjust_contrast(image_array, factor=1.5):
    image = Image.fromarray(cv2.cvtColor(image_array, cv2.COLOR_BGR2RGB))
    enhancer = ImageEnhance.Contrast(image)
    image_enhanced = enhancer.enhance(factor)
    return cv2.cvtColor(np.array(image_enhanced), cv2.COLOR_RGB2BGR)

def apply_clahe(image_array):
    lab = cv2.cvtColor(image_array, cv2.COLOR_BGR2Lab)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    cl = clahe.apply(l)
    limg = cv2.merge((cl, a, b))
    return cv2.cvtColor(limg, cv2.COLOR_Lab2BGR)

def rotate_image(image, angle):
    h, w = image.shape[:2]
    center = (w // 2, h // 2)
    rotation_matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
    rotated_image = cv2.warpAffine(image, rotation_matrix, (w, h))
    return rotated_image

def horizontal_flip(image):
    return cv2.flip(image, 1)

def adjust_saturation(image, factor=1.5):
    image = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    enhancer = ImageEnhance.Color(image)
    image_enhanced = enhancer.enhance(factor)
    return cv2.cvtColor(np.array(image_enhanced), cv2.COLOR_RGB2BGR)

def add_gaussian_noise(image):
    mean = 0
    var = 10
    sigma = var ** 0.5
    gaussian = np.random.normal(mean, sigma, image.shape).astype(np.float32)
    noisy = cv2.add(image.astype(np.float32), gaussian)
    noisy = np.clip(noisy, 0, 255).astype(np.uint8)
    return noisy

def save_augmented_faces(original_image, faces, save_path):
    name = os.path.splitext(os.path.basename(save_path))[0]
    output_dir = os.path.dirname(save_path)
    saved_paths = []

    for idx, face in enumerate(faces):
        cropped = crop_face(original_image, face['box'])
        if cropped.size == 0:
            continue

        face_resized = cv2.resize(cropped, (160, 160))

        original_face_path = os.path.join(output_dir, f"{name}.jpg")
        cv2.imwrite(original_face_path, face_resized)
        saved_paths.append(original_face_path)

        variations = {
            "bright": adjust_brightness_pil(face_resized, factor=1.5),
            "dark": adjust_brightness_pil(face_resized, factor=0.5),
            "sharpened": sharpen_image(face_resized),
            "contrast": adjust_contrast(face_resized, factor=1.5),
            "clahe": apply_clahe(face_resized),
            "rotate_kanan": rotate_image(face_resized, 15),
            "rotate_kiri": rotate_image(face_resized, -15),
            "flip": horizontal_flip(face_resized),
            "saturation": adjust_saturation(face_resized, factor=1.5),
            "noise": add_gaussian_noise(face_resized)
        }

        for aug_name, aug_img in variations.items():
            filename = f"{name}_{aug_name}.jpg"
            save_to = os.path.join(output_dir, filename)
            cv2.imwrite(save_to, aug_img)
            saved_paths.append(save_to)

    return saved_paths

