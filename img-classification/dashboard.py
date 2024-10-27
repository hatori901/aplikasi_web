import streamlit as st
from PIL import Image
from transformers import pipeline

# Inisialisasi pipeline untuk klasifikasi gambar
classifier = pipeline("image-classification", model="dima806/hair_type_image_detection") 

# Judul aplikasi
st.title("Klasifikasi Tipe Rambut")

# Upload gambar
uploaded_file = st.file_uploader("Pilih gambar model", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Menampilkan gambar yang diunggah
    image = Image.open(uploaded_file)
    st.image(image, caption='Gambar yang diunggah', use_column_width=True)

    # Klasifikasi gambar
    st.write("Menganalisis gambar...")
    predictions = classifier(image)

    # Menampilkan hasil klasifikasi
    st.write("Hasil klasifikasi:")
    for prediction in predictions:
        st.write(f"Tipe rambut: {prediction['label']}, Akurasi: {prediction['score']:.2f}")