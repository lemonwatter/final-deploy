import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
import io
import os
import time

# --- Konfigurasi Global ---
# Ganti nama file model Anda di sini
MODEL_PATH = "pix2pix_tryon_G.h5"
INPUT_SHAPE = (256, 192) # Contoh ukuran input yang umum (Tinggi x Lebar) untuk model Try-On
IMAGE_CHANNELS = 3

# --- Fungsi Pemuatan Model (Menggunakan Cache Streamlit) ---
@st.cache_resource
def load_tryon_model():
    """Memuat model Keras dari file. Menggunakan st.cache_resource agar hanya dimuat sekali."""
    st.info(f"Memuat model dari: {MODEL_PATH}...")
    try:
        # PENTING: Pastikan file 'pix2pix_tryon_G.h5' berada di direktori yang sama.
        model = tf.keras.models.load_model(MODEL_PATH)
        st.success("Model berhasil dimuat!")
        return model
    except Exception as e:
        st.error(f"Gagal memuat model: {e}")
        st.warning("Pastikan Anda sudah menginstal TensorFlow dan file model berada di direktori yang benar.")
        # Mengembalikan None atau model dummy untuk memungkinkan UI berjalan
        return None

# --- Fungsi Utilitas Gambar ---

def preprocess_image(img, target_shape):
    """Mengubah ukuran dan menormalisasi gambar untuk input model."""
    img = img.resize((target_shape[1], target_shape[0])) # Resize ke (Lebar, Tinggi)
    img_array = np.array(img).astype('float32')
    
    # Model Pix2Pix umumnya menggunakan normalisasi -1 ke 1
    if img_array.ndim == 3 and img_array.shape[2] == 3:
        img_array = (img_array / 127.5) - 1.0
    else:
        # Jika gambar grayscale, konversi ke RGB
        st.error("Format gambar tidak valid. Harap gunakan gambar RGB.")
        return None
        
    # Tambahkan dimensi batch (1, H, W, C)
    return np.expand_dims(img_array, axis=0)

def postprocess_image(output_array):
    """Menormalisasi output model kembali ke 0-255 dan mengkonversi ke objek PIL Image."""
    # Hapus dimensi batch
    output_array = output_array[0]
    
    # Denormalisasi dari -1 ke 1 menjadi 0 ke 255
    output_array = (output_array + 1.0) * 127.5
    output_array = np.clip(output_array, 0, 255).astype('uint8')
    
    return Image.fromarray(output_array)

# --- Mock Data untuk Sepatu dan Kaki ---

# Gunakan placeholder untuk sepatu
MOCK_SHOES = {
    "Sneakers Merah": "https://placehold.co/192x256/EF4444/FFFFFF/png?text=Sepatu+Merah",
    "Sepatu Boot Coklat": "https://placehold.co/192x256/A3E635/000000/png?text=Sepatu+Boot",
    "Sepatu Kasual Biru": "https://placehold.co/192x256/3B82F6/FFFFFF/png?text=Sepatu+Biru",
}

# Gunakan placeholder untuk kaki
MOCK_FEET = {
    "Kaki Standar (Depan)": "https://placehold.co/192x256/FBBF24/000000/png?text=Kaki+Depan+Contoh",
    "Kaki Standar (Samping)": "https://placehold.co/192x256/22C55E/FFFFFF/png?text=Kaki+Samping+Contoh",
}

# --- Fungsi Utama Streamlit ---
def main():
    st.set_page_config(
        page_title="Virtual Try-On Sepatu",
        layout="wide",
        initial_sidebar_state="expanded"
    )

    st.title("👟 Aplikasi Virtual Try-On Sepatu")
    st.markdown("Aplikasi ini memungkinkan Anda mencoba sepatu secara virtual pada gambar kaki Anda menggunakan model *Image-to-Image Generation* (Pix2Pix/sejenisnya).")

    # Muat model
    generator_model = load_tryon_model()

    # Inisialisasi state sesi untuk menyimpan gambar yang dipilih
    if 'selected_shoe_url' not in st.session_state:
        st.session_state.selected_shoe_url = list(MOCK_SHOES.values())[0]

    if 'selected_foot_img' not in st.session_state:
        st.session_state.selected_foot_img = None
    
    # ----------------------------------------------------
    # Bagian Kiri: Pemilihan Sepatu & Kaki
    # ----------------------------------------------------
    
    col_shoe, col_foot, col_result = st.columns([1, 1, 2])

    with col_shoe:
        st.header("1. Pilih Sepatu")
        shoe_options = list(MOCK_SHOES.keys())
        selected_shoe_name = st.selectbox("Pilih model sepatu:", shoe_options)
        
        selected_shoe_url = MOCK_SHOES[selected_shoe_name]
        st.session_state.selected_shoe_url = selected_shoe_url

        st.image(selected_shoe_url, caption=f"Sepatu Terpilih: {selected_shoe_name}", width=192)

        # Muat gambar sepatu untuk diproses
        try:
            # Menggunakan tf.keras.utils.get_file untuk mendapatkan gambar dari URL (hanya untuk mock data)
            # Dalam aplikasi nyata, Anda mungkin memuatnya dari direktori lokal.
            shoe_path = tf.keras.utils.get_file(selected_shoe_url.split('/')[-1], selected_shoe_url)
            st.session_state.shoe_img = Image.open(shoe_path).convert("RGB")
        except Exception as e:
             st.session_state.shoe_img = None
             st.error(f"Gagal memuat gambar sepatu: {e}")


    with col_foot:
        st.header("2. Sumber Citra Kaki")
        foot_source = st.radio("Pilih sumber citra kaki:", ("Contoh Kaki Disediakan", "Unggah Citra Kaki Pengguna"))
        
        foot_img = None
        
        if foot_source == "Contoh Kaki Disediakan":
            foot_options = list(MOCK_FEET.keys())
            selected_foot_name = st.selectbox("Pilih contoh citra kaki:", foot_options)
            selected_foot_url = MOCK_FEET[selected_foot_name]
            
            st.image(selected_foot_url, caption=f"Kaki Contoh Terpilih: {selected_foot_name}", width=192)
            
            try:
                foot_path = tf.keras.utils.get_file(selected_foot_url.split('/')[-1], selected_foot_url)
                foot_img = Image.open(foot_path).convert("RGB")
            except Exception as e:
                st.error(f"Gagal memuat gambar kaki: {e}")
                foot_img = None

        else:
            uploaded_file = st.file_uploader("Unggah gambar kaki Anda (JPG/PNG)", type=["jpg", "png", "jpeg"])
            if uploaded_file is not None:
                try:
                    foot_img = Image.open(uploaded_file).convert("RGB")
                    st.image(foot_img, caption="Citra Kaki yang Diunggah", width=192)
                except Exception as e:
                    st.error("Terjadi kesalahan saat memuat gambar yang diunggah.")
                    foot_img = None
            else:
                 st.info("Silakan unggah gambar kaki.")

        st.session_state.selected_foot_img = foot_img

    # ----------------------------------------------------
    # Bagian Tengah: Tombol Try-On
    # ----------------------------------------------------
    
    st.markdown("---")
    
    # Pastikan tombol berada di area yang mudah dilihat
    if st.button("👟 LAKUKAN VIRTUAL TRY ON", type="primary", use_container_width=True):
        if generator_model is None:
            st.error("Model tidak tersedia. Silakan periksa pesan kesalahan di atas.")
        elif st.session_state.get('shoe_img') is None or st.session_state.get('selected_foot_img') is None:
            st.warning("Mohon pilih gambar sepatu dan sediakan citra kaki terlebih dahulu.")
        else:
            with st.spinner("⏳ Sedang menggenerate hasil Try-On... Ini mungkin membutuhkan waktu beberapa detik."):
                
                # Mendapatkan gambar dari state
                shoe_img = st.session_state.shoe_img
                foot_img = st.session_state.selected_foot_img

                # 1. Pra-proses gambar
                # Catatan: Model Pix2Pix menggabungkan kedua gambar di channel input (misalnya 6 channel) 
                # atau menumpuknya dalam dimensi batch. Kita akan menggunakan cara menumpuk di dimensi channel (6 channel).
                try:
                    shoe_input = preprocess_image(shoe_img, INPUT_SHAPE)
                    foot_input = preprocess_image(foot_img, INPUT_SHAPE)

                    if shoe_input is None or foot_input is None:
                         st.error("Prapemrosesan gambar gagal.")
                         return

                    # Gabungkan input: Stack di sumbu terakhir (C), sehingga menjadi (1, H, W, 6)
                    combined_input = np.concatenate([shoe_input, foot_input], axis=-1)
                    
                    # 2. Inferensi Model
                    output_array = generator_model.predict(combined_input)
                    
                    # 3. Pasca-proses dan Tampilkan Hasil
                    result_img = postprocess_image(output_array)
                    
                    with col_result:
                        st.header("3. Hasil Virtual Try-On")
                        st.image(result_img, caption="Hasil Try-On Virtual", use_column_width=True)
                        st.balloons()
                        st.success("Virtual Try-On Selesai!")
                        
                except Exception as e:
                    st.error(f"Terjadi kesalahan saat menjalankan inferensi model: {e}")
                    st.warning("Periksa apakah dimensi input model Anda adalah (None, 256, 192, 6).")

    # ----------------------------------------------------
    # Bagian Kanan: Tampilan Hasil
    # ----------------------------------------------------
    with col_result:
        if not st.session_state.get('result_img'):
            st.header("3. Hasil Virtual Try-On")
            st.info("Tekan tombol 'LAKUKAN VIRTUAL TRY ON' untuk melihat hasilnya.")
            
if __name__ == "__main__":
    # Pastikan TensorFlow tidak menggunakan memori GPU secara berlebihan
    gpus = tf.config.experimental.list_physical_devices('GPU')
    if gpus:
        try:
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
        except RuntimeError as e:
            st.warning(f"Error setting GPU memory growth: {e}")
            
    main()