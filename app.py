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

# ==============================================================================
# KONFIGURASI DAN UTILITAS
# ==============================================================================
IMG_SIZE = 256
MODEL_G_PATH = 'models/pix2pix_tryon_G.h5' # Dipastikan model dimuat secara lokal

# Konfigurasi logging untuk debugging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

st.set_page_config(
    page_title="Virtual Shoe Try-On App",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# --- 1. Re-Definisi Arsitektur Model (Generator UNet) ---
# Disesuaikan agar SAMA PERSIS dengan kode training Anda (tensor.py):
# - Input 7 Channel
# - LeakyReLU(0.2)
# - Filter awal 32

def downsample(filters, size, apply_batchnorm=True):
    initializer = tf.random_normal_initializer(0., 0.02)
    result = tf.keras.Sequential()
    result.add(Conv2D(filters, size, strides=2, padding='same',
                      kernel_initializer=initializer, use_bias=False))
    if apply_batchnorm:
        result.add(BatchNormalization())
    # PENTING: Gunakan alpha 0.2 sesuai kode training Anda
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
    # PENTING: Input 7 channel sesuai training Anda (Sepatu 3ch + Kaki 3ch + Mask 1ch)
    inputs = Input(shape=input_shape) 

    # Downsampling (5 Lapisan)
    down_stack = [
        # PENTING: Filter awal 32 sesuai training Anda
        downsample(32, 4, apply_batchnorm=False), # L1
        downsample(64, 4), # L2
        downsample(128, 4), # L3
        downsample(256, 4), # L4
        downsample(512, 4, apply_batchnorm=False), # L5: Bottleneck
    ]

    # Upsampling (4 Lapisan)
    up_stack = [
        # U1: 
        upsample(256, 4, apply_dropout=True), 
        # U2: 
        upsample(128, 4), 
        # U3: 
        upsample(64, 4), 
        # U4: 
        upsample(32, 4), 
    ]
    
    initializer = tf.random_normal_initializer(0., 0.02)
    last = Conv2DTranspose(output_channels, 4, strides=2, padding='same',
                           kernel_initializer=initializer, activation='tanh')

    x = inputs
    skips = []
    
    # Downward Pass (Menghasilkan 5 skips: skips[0] - skips[4])
    for down in down_stack:
        x = down(x)
        skips.append(x)
    
    # Upward Pass (sesuai logika tensor.py Anda)

    # U1 menyambung ke skips[3] (256 filters)
    # skips[-1] adalah bottleneck (skips[4])
    x = up_stack[0](skips[-1]) 
    x = Concatenate()([x, skips[3]]) # Tambahkan koneksi skip dari L4 (skips[3])

    # Melakukan upsampling U2, U3, U4
    # reversed(skips[1:-1]) = skips[2], skips[1] (Kita mulai dari U2, jadi skips[2] dulu)
    # Sisa lapisan upstack adalah [U2, U3, U4]
    
    for up, skip_idx in zip(up_stack[1:], [2, 1, 0]):
         x = up(x)
         x = Concatenate()([x, skips[skip_idx]])

    # Final Layer
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
        st.error(f"❌ Gagal memuat model. Error Incompatibility. Detail: {e}")
        st.stop()

def get_asset_paths(folder_name):
    """Mendapatkan daftar path file gambar dari folder assets."""
    return glob.glob(os.path.join('assets', folder_name, '*.[jp][pn]g'), recursive=True)

# --- 3. Pre-pemrosesan dan Inferensi ---
# Di sini kita perlu meniru apa yang dilakukan oleh 7 channel input.
# Asumsi: Channel ke-7 adalah Mask Biner (semua 1)
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
    shoe_norm = normalize(shoe_img)
    feet_norm = normalize(feet_img)
    
    # === PENTING: MENGATUR 7 CHANNEL INPUT ===
    # Gabungkan input (Sepatu 3ch + Kaki 3ch). Tambahkan 1 Channel Mask biner (semua 1)
    # Ini meniru mask yang mungkin Anda gunakan di training (sebagai placeholder)
    mask_channel = np.ones((IMG_SIZE, IMG_SIZE, 1), dtype=np.float32)
    input_tensor = np.concatenate([shoe_norm, feet_norm, mask_channel], axis=-1) # Total 7 channels!
    
    input_tensor = np.expand_dims(input_tensor, axis=0) # Tambah dimensi batch

    with result_container:
        with st.spinner('⏳ Sedang Menerapkan Try-On Virtual...'):
            try:
                # Inferensi
                prediction = netG(input_tensor, training=False)[0].numpy()
                
                # Denormalisasi
                prediction = (prediction * 0.5 + 0.5) * 255.0
                prediction = prediction.clip(0, 255).astype(np.uint8)
                
                # Tampilkan hasil
                st.subheader("🎉 Hasil Virtual Try-On")
                st.image(prediction, caption="Hasil Penggabungan Sepatu dan Kaki", use_column_width=True)
                
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
    
# Muat Model (netG akan mencoba memuat model dengan arsitektur 7-channel yang baru)
netG = load_generator_model(MODEL_G_PATH)

st.title("👟 Aplikasi Virtual Try-On Sepatu")
st.markdown("---")

# [Sisa UI logic tetap sama]

def shoe_catalog(shoe_assets):
    st.header("1. Pilih Sepatu dari Katalog")
    st.markdown("*(Klik pada gambar sepatu untuk mengaktifkan Try-On)*")
    
    cols = st.columns(4)
    
    for i, shoe_path in enumerate(shoe_assets):
        with cols[i % 4]:
            shoe_name = os.path.basename(shoe_path)
            
            is_selected = (shoe_path == st.session_state['selected_shoe_path'])
            
            st.image(shoe_path, caption="", use_column_width=True)
            
            button_label = "✅ Dipilih" if is_selected else "Pilih"
            button_type = "secondary" if is_selected else "primary"
            
            if st.button(button_label, key=f'select_{shoe_name}', type=button_type, use_container_width=True):
                st.session_state['selected_shoe_path'] = shoe_path
                st.session_state['feet_input_data'] = None 
                st.rerun() 

col_input, col_result = st.columns([1, 1], gap="large")

with col_input:
    shoe_assets = get_asset_paths('shoes')
    shoe_catalog(shoe_assets)

st.markdown("---") 

if st.session_state['selected_shoe_path']:
    with col_input:
        st.header("2. Sediakan Citra Kaki")
        
        st.subheader("Sepatu yang Dipilih:")
        st.image(st.session_state['selected_shoe_path'], use_column_width=True)
        st.markdown("---")
        
        input_method = st.radio(
            "Pilih Metode Input Kaki:",
            ("Pilih dari Galeri", "Unggah Citra Kaki Sendiri"),
            key='input_method_radio'
        )
        
        input_feet_data = None
        
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

        else:
            uploaded_file = st.file_uploader(
                "Unggah Citra Kaki (JPG/PNG)", 
                type=["jpg", "png", "jpeg"], 
                key='feet_uploader'
            )
            if uploaded_file is not None:
                input_feet_data = uploaded_file
            
        
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
        if st.button("✨ Terapkan Virtual Try-On", key='tryon_button', type="primary", use_container_width=True):
            if st.session_state['selected_shoe_path'] and st.session_state['feet_input_data']:
                process_inference(st.session_state['selected_shoe_path'], st.session_state['feet_input_data'], netG, col_result)
            else:
                with col_result: 
                    st.warning("Mohon pilih sepatu dan sediakan citra kaki terlebih dahulu.")
else:
    with col_result:
        st.header("Selamat Datang!")
        st.info("👈 Silakan klik salah satu gambar sepatu di katalog (kolom kiri) untuk memulai Virtual Try-On.")
        st.markdown("""
        **Langkah Selanjutnya:**
        1. Pilih Sepatu.
        2. Pilih sumber gambar kaki (Galeri atau Unggah).
        3. Klik tombol Try-On untuk melihat hasilnya di kolom ini.
        """)

st.markdown("""
<style>
.stButton>button {
    font-weight: bold;
    border-radius: 0.5rem;
    box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1), 0 2px 4px -2px rgba(0, 0, 0, 0.06);
}
</style>
""", unsafe_allow_html=True)