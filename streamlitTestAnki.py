import streamlit as st
from PIL import Image, ImageDraw
import os
import numpy as np
import math
from collections import deque
import onnxruntime as ort

# --- Initialisation ---

det_model_path = "models/en_PP-OCRv3_det_infer.onnx"
rec_model_path = "models/latin_PP-OCRv3_rec_infer.onnx"
dict_path = "models/latin_dict.txt"

det_session = ort.InferenceSession(det_model_path)
rec_session = ort.InferenceSession(rec_model_path)

def load_character_dict(dict_path):
    with open(dict_path, "rb") as f:
        char_list = [line.decode('utf-8').strip() for line in f.readlines()]
    if ' ' not in char_list:
        char_list.insert(0, ' ')
    char_list.append('</s>')
    return char_list

character_dict = load_character_dict(dict_path)
blank_idx = len(character_dict) - 1

# --- Fonctions utilitaires (copiées de ton code) ---

def preprocess_det(image_pil, model_input_size=(960, 960)):
    image_resized = image_pil.resize(model_input_size, Image.BILINEAR)
    image_np = np.array(image_resized).astype(np.float32) / 255.0
    image_np -= np.array([0.485, 0.456, 0.406])
    image_np /= np.array([0.229, 0.224, 0.225])
    image_np = image_np.transpose(2, 0, 1)
    image_np = np.expand_dims(image_np, axis=0)
    return image_np

def resize_norm_img(img, rec_image_shape=(3, 48, 320)):
    img_c, img_h, img_w = rec_image_shape
    w, h = img.size
    ratio = w / float(h)
    resized_w = min(img_w, int(math.ceil(img_h * ratio)))
    resized_image = img.resize((resized_w, img_h), Image.BILINEAR)
    resized_image = np.array(resized_image).astype(np.float32) / 255.0
    resized_image -= 0.5
    resized_image /= 0.5
    resized_image = resized_image.transpose((2, 0, 1))
    padding_im = np.zeros((img_c, img_h, img_w), dtype=np.float32)
    padding_im[:, :, 0:resized_w] = resized_image
    return np.expand_dims(padding_im, axis=0)

def boxes_are_touching_or_close(box1, box2, tolerance=5):
    ax1, ay1, ax2, ay2 = box1
    bx1, by1, bx2, by2 = box2
    ax1 -= tolerance
    ay1 -= tolerance
    ax2 += tolerance
    ay2 += tolerance
    overlap_x = not (bx2 < ax1 or bx1 > ax2)
    overlap_y = not (by2 < ay1 or by1 > ay2)
    return overlap_x and overlap_y

def merge_touching_rectangles(rects, tolerance=5):
    merged, used = [], [False]*len(rects)
    rects_xyxy = [(r[0][0], r[0][1], r[1][0], r[1][1]) for r in rects]
    for i in range(len(rects_xyxy)):
        if used[i]: continue
        x1, y1, x2, y2 = rects_xyxy[i]
        for j in range(i + 1, len(rects_xyxy)):
            if used[j]: continue
            if boxes_are_touching_or_close((x1, y1, x2, y2), rects_xyxy[j], tolerance):
                x1 = min(x1, rects_xyxy[j][0])
                y1 = min(y1, rects_xyxy[j][1])
                x2 = max(x2, rects_xyxy[j][2])
                y2 = max(y2, rects_xyxy[j][3])
                used[j] = True
        used[i] = True
        merged.append([[x1, y1], [x2, y2]])
    return merged

def detect_boxes_numpy(prob_map, thresh=0.3, min_area=10):
    heatmap = prob_map[0, 0]
    binary_map = heatmap > thresh
    height, width = binary_map.shape
    labeled_array = np.zeros_like(binary_map, dtype=int)
    current_label = 1
    for y_start in range(height):
        for x_start in range(width):
            if binary_map[y_start, x_start] and labeled_array[y_start, x_start] == 0:
                q = deque([(y_start, x_start)])
                labeled_array[y_start, x_start] = current_label
                while q:
                    y, x = q.popleft()
                    for dy in [-1, 0, 1]:
                        for dx in [-1, 0, 1]:
                            ny, nx = y + dy, x + dx
                            if (0 <= ny < height and 0 <= nx < width and
                                binary_map[ny, nx] and labeled_array[ny, nx] == 0):
                                labeled_array[ny, nx] = current_label
                                q.append((ny, nx))
                current_label += 1
    boxes = []
    for label_id in range(1, current_label):
        coords = np.argwhere(labeled_array == label_id)
        if len(coords) < min_area: continue
        y_coords, x_coords = coords[:, 0], coords[:, 1]
        y_min, y_max = np.min(y_coords), np.max(y_coords)
        x_min, x_max = np.min(x_coords), np.max(x_coords)
        boxes.append([[x_min, y_min], [x_max + 1, y_max + 1]])
    return boxes

def decode_rec(preds, character_dict):
    preds_idx = preds.argmax(axis=2)
    preds_prob = preds.max(axis=2)
    result, confidences, last_idx = [], [], -1
    for i, idx in enumerate(preds_idx[0]):
        if idx != last_idx and idx != blank_idx:
            result.append(character_dict[idx])
            confidences.append(preds_prob[0, i])
        last_idx = idx
    final_confidence = np.mean([c for c in confidences if c > 0]) if confidences else 0
    return ''.join(result), float(final_confidence)

# --- Fonction OCR principale ---

def run_ocr(image_pil):
    original_width, original_height = image_pil.size
    input_blob = preprocess_det(image_pil)
    output = det_session.run(None, {'x': input_blob})
    boxes = merge_touching_rectangles(detect_boxes_numpy(output[0], 0.3, 20), 20)

    model_input_size = 960.0
    scale_x = original_width / model_input_size
    scale_y = original_height / model_input_size

    results = []
    for box in boxes:
        (x1, y1), (x2, y2) = box
        pt1 = (int(x1 * scale_x), int(y1 * scale_y))
        pt2 = (int(x2 * scale_x), int(y2 * scale_y))
        cropped = image_pil.crop((pt1[0], pt1[1], pt2[0], pt2[1]))
        if cropped.width < 1 or cropped.height < 1:
            continue
        rec_input = resize_norm_img(cropped)
        rec_output = rec_session.run(None, {'x': rec_input})[0]
        text, confidence = decode_rec(rec_output, character_dict)
        if len(text) > 2 and (text.count("0") + text.count("o")) < len(text) / 2:
            results.append({
                "box": [pt1, pt2],
                "text": text,
                "confidence": confidence
            })
    return results

# --- Streamlit UI ---

st.title("OCR Auto-Occlusion avec Streamlit")

uploaded_file = st.file_uploader("Upload une image", type=["jpg", "jpeg", "png"])

if uploaded_file:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Image uploadée", use_column_width=True)

    if st.button("Lancer OCR"):
        with st.spinner("Analyse en cours..."):
            results = run_ocr(image)
        if not results:
            st.write("Aucun texte détecté.")
        else:
            # Dessiner les boîtes sur l'image
            draw = ImageDraw.Draw(image)
            for res in results:
                pt1, pt2 = res["box"]
                draw.rectangle([pt1, pt2], outline="red", width=2)
                draw.text(pt1, f'{res["text"]} ({res["confidence"]:.2f})', fill="red")

            st.image(image, caption="Résultats OCR", use_column_width=True)

            # Afficher les résultats sous forme de tableau
            st.write("Texte détecté :")
            for res in results:
                st.write(f'- "{res["text"]}" avec confiance {res["confidence"]:.2f}')

