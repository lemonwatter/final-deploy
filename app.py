import streamlit as st
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Conv2D, LeakyReLU, Activation, BatchNormalization, Dropout, Concatenate, Conv2DTranspose
import numpy as np
from PIL import Image
import os
import glob
import io 
import logging
from functools import lru_cache

# ==============================================================================
# KONFIGURASI DAN UTILITAS
# ==============================================================================
IMG_SIZE = 256
MODEL_G_PATH = 'models/pix2pix_tryon_G.h5' 

# Konfigurasi logging untuk debugging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

st.set_page_config(
    page_title="Virtual Shoe Try-On App",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# --- 1. Re-Definisi Arsitektur Model (Generator UNet) ---
# Disesuaikan agar SAMA PERSIS dengan kode training Anda: Input 7 Channel, Filter 32, LeakyReLU(0.2)

def downsample(filters, size, apply_batchnorm=True):
    initializer = tf.random_normal_initializer(0., 0.02)
    result = tf.keras.Sequential()
    result.add(Conv2D(filters, size, strides=2, padding='same',
                      kernel_initializer=initializer, use_bias=False))
    if apply_batchnorm:
        result.add(BatchNormalization())
    result.add(LeakyReLU(0.2)) 
    return result

def upsample(filters, size, apply_dropout=False):
    initializer = tf.random_normal_initializer(0., 0.02)
    result = tf.keras.Sequential()
    result.add(Conv2DTranspose(filters, size, strides=2, padding='same',
                               kernel_initializer=initializer, use_bias=False))
    result.add(BatchNormalization())
    if apply_dropout:
        result.add(Dropout(0.5))
    result.add(Activation('relu'))
    return result

def GeneratorUNet(input_shape=(IMG_SIZE, IMG_SIZE, 7), output_channels=3):
    inputs = Input(shape=input_shape) 

    # Downsampling (5 Lapisan)
    down_stack = [
        downsample(32, 4, apply_batchnorm=False), # L1
        downsample(64, 4),                       # L2
        downsample(128, 4),                      # L3
        downsample(256, 4),                      # L4
        downsample(512, 4, apply_batchnorm=False), # L5: Bottleneck
    ]

    # Upsampling (4 Lapisan)
    up_stack = [
        upsample(256, 4, apply_dropout=True), # U1 (Koneksi ke L4)
        upsample(128, 4),                     # U2 (Koneksi ke L3)
        upsample(64, 4),                      # U3 (Koneksi ke L2)
        upsample(32, 4),                      # U4 (Koneksi ke L1)
    ]
    
    initializer = tf.random_normal_initializer(0., 0.02)
    last = Conv2DTranspose(output_channels, 4, strides=2, padding='same',
                           kernel_initializer=initializer, activation='tanh')

    x = inputs
    skips = []
    
    for down in down_stack:
        x = down(x)
        skips.append(x)
    
    x = up_stack[0](skips[-1]) 
    x = Concatenate()([x, skips[3]]) 

    for up, skip_idx in zip(up_stack[1:], [2, 1, 0]):
        x = up(x)
        x = Concatenate()([x, skips[skip_idx]])

    x = last(x)

    return Model(inputs=inputs, outputs=x, name='Generator')

# --- 2. Fungsi Pemuatan Model dan Aset ---

@st.cache_resource
def load_generator_model(model_path):
    """Memuat model generator (netG) dari path lokal."""
    logger.info(f"Attempting to load model from: {model_path}")

    if not os.path.exists(model_path):
        logger.error(f"File model not found at: {model_path}")
        st.error(f"❌ File model tidak ditemukan di: {model_path}. Pastikan path dan nama filenya benar di GitHub.")
        st.stop()
    
    try:
        netG = GeneratorUNet()
        logger.info("GeneratorUNet architecture defined successfully.")
        
        # Coba muat weights
        netG.load_weights(model_path)
        logger.info("Model weights loaded successfully.")
        
        st.success("✅ Model Generator berhasil dimuat secara lokal.")
        return netG
    except Exception as e:
        logger.error(f"FAILED TO LOAD MODEL: {e}")
        st.error(f"❌ Gagal memuat model. Error: {e}")
        st.stop()

def get_asset_paths(folder_name):
    """Mendapatkan daftar path file gambar dari folder assets."""
    files = glob.glob(os.path.join('assets', folder_name, '*.[jp][pn]g'), recursive=True)
    files.extend(glob.glob(os.path.join('assets', folder_name, '*.jpeg'), recursive=True))
    return files

# --- 3. Pre-pemrosesan dan Inferensi ---

def normalize(image):
    return (image / 127.5) - 1

def load_image(image_data):
    """Mengubah data gambar menjadi tensor yang diproses."""
    if isinstance(image_data, str):
        img = Image.open(image_data).convert('RGB')
    elif isinstance(image_data, io.BytesIO) or hasattr(image_data, 'read'):
        img = Image.open(image_data).convert('RGB')
    else:
        st.error("Input data gambar tidak valid.")
        return None

    img = img.resize((IMG_SIZE, IMG_SIZE))
    img = np.array(img, dtype=np.float32)
    return img

def process_inference(shoe_path, feet_data, netG, result_container):
    """Memproses gambar sepatu dan kaki, melakukan inferensi, dan menampilkan hasil."""
    
    shoe_img = load_image(shoe_path)
    feet_img = load_image(feet_data)

    if shoe_img is None or feet_img is None:
        return 

    # Normalisasi
    shoe_norm = normalize(shoe_img) # Rentang [-1, 1]
    feet_norm = normalize(feet_img) # Rentang [-1, 1]
    
    # === PENGATURAN 7 CHANNEL INPUT: MASKING KE -1.0 ===
    # Karena output putih (bias ke 1.0), kita coba nilai ekstrem sebaliknya.
    # Mengisi channel ke-7 (mask) dengan nilai -1.0
    mask_channel_float = np.full((IMG_SIZE, IMG_SIZE, 1), -1.0, dtype=np.float32) 
    
    input_tensor = np.concatenate([shoe_norm, feet_norm, mask_channel_float], axis=-1) # Total 7 channels!
    
    input_tensor = np.expand_dims(input_tensor, axis=0) # Tambah dimensi batch

    with result_container:
        with st.spinner('⏳ Sedang Menerapkan Try-On Virtual (Mask Value: -1.0)...'):
            try:
                # Inferensi
                prediction = netG(input_tensor, training=False)[0].numpy()
                
                # Tambahkan Debugging Output Mentah
                min_raw = np.min(prediction)
                max_raw = np.max(prediction)
                mean_raw = np.mean(prediction)
                
                logger.info(f"Raw Output Range: Min={min_raw:.6f}, Max={max_raw:.6f}, Mean={mean_raw:.6f}")
                
                if mean_raw > 0.99:
                    st.warning("⚠️ Hasilnya mungkin putih. Coba ganti model atau periksa skema masking saat training.")

                # Denormalisasi (Sudah benar: [-1, 1] -> [0, 255])
                prediction = (prediction * 0.5 + 0.5) * 255.0
                prediction = prediction.clip(0, 255).astype(np.uint8)
                
                # Tampilkan hasil
                st.subheader("🎉 Hasil Virtual Try-On")
                st.image(prediction, caption="Hasil Try-On (Mask Value: -1.0)", use_column_width=True) 
                
            except Exception as e:
                st.error(f"❌ Terjadi kesalahan saat inferensi: {e}")

# ==============================================================================
# APLIKASI STREAMLIT UTAMA
# ==============================================================================

# Inisialisasi State
if 'selected_shoe_path' not in st.session_state:
    st.session_state['selected_shoe_path'] = None
if 'feet_input_data' not in st.session_state:
    st.session_state['feet_input_data'] = None
    
# Muat Model
netG = load_generator_model(MODEL_G_PATH)

st.title("👟 Aplikasi Virtual Try-On Sepatu")
st.markdown("---")

def shoe_catalog(shoe_assets):
    st.header("1. Pilih Sepatu dari Katalog")
    st.markdown("*(Klik tombol 'Pilih' di bawah gambar untuk mengaktifkan Try-On)*")
    
    cols = st.columns(4)
    
    for i, shoe_path in enumerate(shoe_assets):
        with cols[i % 4]:
            shoe_name = os.path.basename(shoe_path)
            
            is_selected = (shoe_path == st.session_state['selected_shoe_path'])
            
            # Rendering gambar yang stabil
            st.image(shoe_path, caption="", use_column_width=True)
            
            button_label = "✅ Dipilih" if is_selected else "Pilih"
            button_type = "secondary" if is_selected else "primary"
            
            # Tombol untuk memilih sepatu
            if st.button(button_label, key=f'select_{shoe_name}', type=button_type, use_container_width=True):
                st.session_state['selected_shoe_path'] = shoe_path
                st.session_state['feet_input_data'] = None 
                st.rerun() 

col_input, col_result = st.columns([1, 1], gap="large")

with col_input:
    shoe_assets = get_asset_paths('shoes')
    shoe_catalog(shoe_assets)

st.markdown("---") 

# --- Bagian Try-On (Hanya muncul jika sepatu sudah dipilih) ---
if st.session_state['selected_shoe_path']:
    with col_input:
        st.header("2. Sediakan Citra Kaki")
        
        st.subheader("Sepatu yang Dipilih:")
        st.image(st.session_state['selected_shoe_path'], use_column_width=True)
        st.markdown("---")
        
        # OPSI INPUT KAKI (Radio Button)
        input_method = st.radio(
            "Pilih Metode Input Kaki:",
            ("Pilih dari Galeri", "Unggah Citra Kaki Sendiri"),
            key='input_method_radio'
        )
        
        input_feet_data = None
        
        # LOGIKA PILIH DARI GALERI
        if input_method == "Pilih dari Galeri":
            feet_assets = get_asset_paths('feet')
            feet_options = [os.path.basename(p) for p in feet_assets]
            
            selected_feet_name = st.selectbox(
                "Pilih Bentuk Kaki Galeri:", 
                feet_options, 
                index=0, 
                key='select_feet_gallery'
            )
            input_feet_data = os.path.join('assets', 'feet', selected_feet_name)

        # LOGIKA OPSI UNGGAH
        else:
            uploaded_file = st.file_uploader(
                "Unggah Citra Kaki (JPG/PNG)", 
                type=["jpg", "png", "jpeg"], 
                key='feet_uploader'
            )
            if uploaded_file is not None:
                input_feet_data = uploaded_file
            
        
        # Menampilkan citra kaki yang dipilih di kolom input
        if input_feet_data is not None:
            st.markdown("---")
            st.subheader("Pratinjau Citra Kaki:")
            try:
                if input_method == "Pilih dari Galeri":
                    st.image(input_feet_data, caption=os.path.basename(input_feet_data), use_column_width=True)
                else:
                    st.image(input_feet_data, caption="Citra Kaki Unggahan Anda", use_column_width=True)
                st.session_state['feet_input_data'] = input_feet_data 
            except Exception as e:
                st.warning(f"Tidak dapat menampilkan pratinjau gambar: {e}")
            st.markdown("---")
        else:
            st.session_state['feet_input_data'] = None

        st.markdown("<br>", unsafe_allow_html=True)
        # TOMBOL TRY-ON
        if st.button("✨ Terapkan Virtual Try-On", key='tryon_button', type="primary", use_container_width=True):
            if st.session_state['selected_shoe_path'] and st.session_state['feet_input_data']:
                # Panggil inference dengan mask 7th channel set ke -1.0
                process_inference(
                    st.session_state['selected_shoe_path'], 
                    st.session_state['feet_input_data'], 
                    netG, col_result
                )
            else:
                with col_result: 
                    st.warning("Mohon pilih sepatu dan sediakan citra kaki terlebih dahulu.")
else:
    # Tampilkan instruksi jika belum ada sepatu yang dipilih
    with col_result:
        st.header("Selamat Datang!")
        st.info("👈 Silakan klik salah satu tombol 'Pilih' di katalog (kolom kiri) untuk memulai Virtual Try-On.")
        st.markdown("""
        **Langkah Selanjutnya:**
        1. Pilih Sepatu dengan mengklik tombol 'Pilih'.
        2. Pilih sumber gambar kaki (Galeri atau Unggah).
        3. Klik tombol Try-On untuk melihat hasilnya di kolom ini.
        """)

# Tambahkan sedikit CSS custom untuk tampilan Streamlit yang lebih baik
st.markdown("""
<style>
.stButton>button {
    font-weight: bold;
    border-radius: 0.5rem;
    box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1), 0 2px 4px -2px rgba(0, 0, 0, 0.06);
}
</style>
""", unsafe_allow_html=True)