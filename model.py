import streamlit as st
import os
import cv2
import numpy as np
from keras.models import load_model
from PIL import Image

# Đường dẫn đến thư mục chứa ảnh
BASE_PATH = 'dataset'


def load_and_cache_model():
    model = load_model('model_pre.h5')
    return model


def preprocess_image(image):
    # Kích thước mong muốn
    target_size = (224, 224)

    # Resize ảnh về kích thước mong muốn
    resized_image = cv2.resize(image, target_size)

    # Chuẩn hóa ảnh bằng cách chia cho 255.0
    normalized_image = resized_image / 255.

    # Thêm một chiều để phù hợp với định dạng đầu vào của mô hình
    processed_image = np.expand_dims(normalized_image, axis=0)
    return processed_image


def predict(image, model):
    # Tiền xử lý ảnh
    image_cop = image.copy()
    preprocessed_image = preprocess_image(image)
    # Dự đoán nhãn và bounding box
    (label_preds, box_preds) = model.predict(preprocessed_image)
    # Lấy nhãn dự đoán và các bounding box
    predicted_label_idx = np.argmax(label_preds)
    class_labels = sorted(os.listdir(os.path.join(BASE_PATH, 'data')))
    predicted_label = class_labels[predicted_label_idx]
    # Lấy tọa độ của bounding box
    (x1, y1, x2, y2) = box_preds[0]
    h, w, _ = image.shape
    x1 *= w
    y1 *= h
    x2 *= w
    y2 *= h
    img_with_boxes = cv2.rectangle(image, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 3)
    cv2.putText(img_with_boxes, predicted_label, (int(x1), int(y1) - 10), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2,cv2.LINE_AA)
    return image_cop, img_with_boxes, predicted_label


def main():
    st.title("Phát hiện đối tượng trong ảnh")

    # Tải ảnh lên
    uploaded_file = st.file_uploader("Tải ảnh lên", type=['jpg', 'jpeg'])

    if uploaded_file is not None:
        # Đọc ảnh từ file
        image = np.array(Image.open(uploaded_file))
        # Tải và cache mô hình
        model = load_and_cache_model()

        # Dự đoán của mô hình
        img_before, img_with_boxes, predicted_label = predict(image, model)

        # Hiển thị ảnh gốc
        st.image(img_before, caption='Ảnh Gốc', use_column_width=True)

        # Hiển thị ảnh với hộp giới hạn và nhãn
        st.image(img_with_boxes, caption='Đối tượng Phát hiện', use_column_width=True)

        # Chuyển nhãn dự đoán thành chuỗi thể hiện đối tượng
        label_pre = ''
        if predicted_label == 'person':
            label_pre = ' Người đi bộ'
        elif predicted_label == 'motorbike':
            label_pre = ' Xe máy'
        elif predicted_label == 'bike':
            label_pre = ' Xe đạp'
        else:
            label_pre = ' Xe ô tô'

        # Hiển thị nhãn dự đoán
        st.write("Đối tượng phát hiện trong ảnh: ", label_pre)


if __name__ == "__main__":
    main()
