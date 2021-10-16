import streamlit as st
import numpy as np
import cv2
import matplotlib.pyplot as plt
import requests
from PIL import Image

st.set_page_config(
    page_title='Konrad-Suze Team - Elang.AI',
    page_icon='https://telkomuniversity.ac.id/wp-content/uploads/2019/07/cropped-favicon-2-32x32.png',
    layout='wide'
)


def draw_bounding_box(img, boxes, pred_cls, rect_th=3):
    img = np.asarray(img)
    class_color_dict = {}

    for cat in pred_cls:
        class_color_dict[cat] = [255]

    for i in range(len(boxes)):
        cv2.rectangle(
            img,
            (int(boxes[i][0]['x1']), int(boxes[i][0]['y1'])),
            (int(boxes[i][1]['x2']), int(boxes[i][1]['y2'])),
            color=class_color_dict[pred_cls[i]], thickness=rect_th
        )

    plt.figure(figsize=(20, 30))
    plt.imshow(img)
    plt.xticks([])
    plt.yticks([])
    st.pyplot(plt)


def predict_image(image):
    response = requests.post("https://elang.kurniadiwijaya.my.id/predict_image", files={'file': image.getbuffer()})
    dict_response = response.json()

    return dict_response


def main():
    st.title('People Detection Demo')
    st.write("")
    col1, col2, col3 = st.columns([3, 0.5, 3])

    with col1:
        image = st.file_uploader("Pilih Gambar", type=['png', 'jpeg', 'jpg'])

        if image is not None:
            image_1 = Image.open(image)
            st.image(image_1, caption='Uploaded Image.')

    if image is not None:
        try:
            data_dict = predict_image(image)
            with col3:
                st.write("## **People Detection Result**")
                draw_bounding_box(Image.open(image), data_dict['data']['boxes'], data_dict['data']['classes'])
                total = len(data_dict['data']['classes'])
                st.write(f'Jumlah Orang Terdeteksi : {total}')

            if image is not None:
                st.write("")
                st.write("## **JSON Fetch Endpoint**")
                st.code("""
            response = requests.post("https://elang.kurniadiwijaya.my.id/predict_image", files={'file': image.getbuffer()})
            dict_response = response.json()
            """, language='python')
                st.json(data_dict)
        except():
            st.write("Error Access The Server API")


main()
