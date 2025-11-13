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
# PASTIKAN PATH MODEL INI BENAR
MODEL_G_PATH = 'pix2pix_tryon_G.h5' 

# Konfigurasi logging untuk debugging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

st.set_page_config(
    page_title="Virtual Shoe Try-On App",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- 1. Re-Definisi Arsitektur Model (Generator UNet) ---
# Dibiarkan sama persis dengan kode Anda untuk memastikan kompatibilitas weights
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
        upsample(128, 4),                      # U2 (Koneksi ke L3)
        upsample(64, 4),                       # U3 (Koneksi ke L2)
        upsample(32, 4),                       # U4 (Koneksi ke L1)
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
    x = Concatenate()([x, skips[3]]) # Connection L4 to U1

    for up, skip_idx in zip(up_stack[1:], [2, 1, 0]):
        x = up(x)
        x = Concatenate()([x, skips[skip_idx]])

    x = last(x)

    return Model(inputs=inputs, outputs=x, name='Generator')

# --- 2. Fungsi Pemuatan Model dan Aset ---

# Ubah agar mencoba menggunakan load_model penuh terlebih dahulu, lebih stabil
@st.cache_resource
def load_generator_model(model_path):
    """Memuat model generator (netG) dari path lokal."""
    logger.info(f"Attempting to load model from: {model_path}")

    if not os.path.exists(model_path):
        logger.error(f"File model not found at: {model_path}")
        st.error(f"❌ File model tidak ditemukan di: {model_path}. Pastikan path dan nama filenya benar!")
        return None
    
    try:
        # Paling baik jika model disimpan sebagai file .h5 lengkap
        netG = tf.keras.models.load_model(model_path, compile=False)
        st.success("✅ Model Generator berhasil dimuat menggunakan tf.keras.models.load_model.")
        return netG
    except Exception as e:
        logger.warning(f"Gagal memuat model penuh: {e}. Mencoba memuat weights ke arsitektur kustom.")
        try:
            # Fallback: jika hanya weights yang disimpan
            netG = GeneratorUNet()
            netG.load_weights(model_path)
            st.warning("⚠️ Model dimuat dengan memuat weights ke arsitektur GeneratorUNet kustom.")
            return netG
        except Exception as e:
            logger.error(f"FAILED TO LOAD MODEL (Weights/Architecture): {e}")
            st.error(f"❌ Gagal memuat model. Periksa apakah arsitektur UNet di kode sama persis dengan model training Anda.")
            return None

@lru_cache(maxsize=32)
def get_asset_paths(folder_name):
    """Mendapatkan daftar path file gambar dari folder assets."""
    # Ini memerlukan struktur folder: 'assets/shoes/' dan 'assets/feet/'
    base_dir = os.path.join(os.getcwd(), 'assets', folder_name)
    if not os.path.isdir(base_dir):
        # Buat direktori mock jika tidak ada (untuk Streamlit Cloud)
        os.makedirs(base_dir, exist_ok=True)
        # Tambahkan mock files agar UI tetap berjalan saat deployment pertama kali
        if folder_name == 'shoes':
            mock_files = [f"{base_dir}/shoe_{i}.png" for i in range(1, 4)]
            # Buat placeholder jika file tidak ada
            for path in mock_files:
                if not os.path.exists(path):
                    Image.new('RGB', (IMG_SIZE, IMG_SIZE), color = 'red' if '1' in path else 'blue').save(path)
            return mock_files
        if folder_name == 'feet':
            mock_files = [f"{base_dir}/feet_{i}.png" for i in range(1, 3)]
            for path in mock_files:
                if not os.path.exists(path):
                    Image.new('RGB', (IMG_SIZE, IMG_SIZE), color = 'lightgray').save(path)
            return mock_files
            
    # Mencari file nyata
    files = glob.glob(os.path.join(base_dir, '*.[jp][pn]g'), recursive=True)
    files.extend(glob.glob(os.path.join(base_dir, '*.jpeg'), recursive=True))
    if not files:
         st.warning(f"Tidak ada file yang ditemukan di folder assets/{folder_name}/. Menggunakan mock data.")
         return get_asset_paths(folder_name)
         
    return files

# --- 3. Pre-pemrosesan dan Inferensi ---

def normalize(image):
    # Asumsi Tanh activation di output layer, normalisasi ke [-1, 1]
    return (image / 127.5) - 1

def denormalize(prediction):
    # Denormalisasi dari [-1, 1] ke [0, 255]
    prediction = (prediction * 0.5 + 0.5) * 255.0
    return prediction.clip(0, 255).astype(np.uint8)

def load_image(image_data):
    """Mengubah data gambar menjadi tensor yang diproses."""
    if isinstance(image_data, str) and os.path.exists(image_data):
        img = Image.open(image_data).convert('RGB')
    elif hasattr(image_data, 'read'): # Untuk Streamlit uploaded_file
        img = Image.open(image_data).convert('RGB')
    else:
        # Ini akan terjadi jika input_data adalah path string tapi tidak ada file (misal mock data URL)
        logger.error("Input data gambar tidak valid atau file tidak ada.")
        return None

    img = img.resize((IMG_SIZE, IMG_SIZE))
    img = np.array(img, dtype=np.float32)
    return img

def process_inference(shoe_path, feet_data, netG, result_container, mask_value):
    """Memproses gambar, inferensi, dan menampilkan hasil."""
    
    shoe_img = load_image(shoe_path)
    feet_img = load_image(feet_data)

    if shoe_img is None or feet_img is None:
        result_container.error("Gagal memuat atau memproses salah satu gambar. Pastikan gambar ada dan valid.")
        return 

    # Normalisasi
    shoe_norm = normalize(shoe_img) 
    feet_norm = normalize(feet_img) 
    
    # === KRITIKAL: PENGATURAN MASK 7th CHANNEL ===
    # Mengisi channel ke-7 (mask) dengan nilai tunggal yang dipilih pengguna
    mask_channel_float = np.full((IMG_SIZE, IMG_SIZE, 1), mask_value, dtype=np.float32) 
    
    # Concatenation: (H, W, 3) + (H, W, 3) + (H, W, 1) = (H, W, 7)
    input_tensor = np.concatenate([shoe_norm, feet_norm, mask_channel_float], axis=-1) 
    
    input_tensor = np.expand_dims(input_tensor, axis=0) # Tambah dimensi batch (1, H, W, 7)

    with result_container:
        with st.spinner(f'⏳ Sedang Menerapkan Try-On Virtual (Mask Value: {mask_value})...'):
            try:
                # Inferensi
                prediction = netG(input_tensor, training=False)[0].numpy()
                
                # Cek hasil mentah untuk debugging output putih
                min_raw = np.min(prediction)
                max_raw = np.max(prediction)
                mean_raw = np.mean(prediction)
                logger.info(f"Raw Output Range: Min={min_raw:.4f}, Max={max_raw:.4f}, Mean={mean_raw:.4f}")
                
                if abs(max_raw - min_raw) < 0.01:
                    st.error("❌ Hasilnya sangat flat/hampir putih. Coba ganti '7th Channel Mask Value' di sidebar.")

                # Denormalisasi dan klip
                final_output = denormalize(prediction)
                
                # Tampilkan hasil
                st.subheader("🎉 Hasil Virtual Try-On")
                st.image(final_output, caption=f"Hasil Try-On (Mask Value: {mask_value})", use_column_width=True) 
                st.balloons()
                
            except Exception as e:
                st.error(f"❌ Terjadi kesalahan saat inferensi: {e}")
                logger.error(f"Inferensi error: {e}")

# ==============================================================================
# APLIKASI STREAMLIT UTAMA
# ==============================================================================

# Inisialisasi State
if 'selected_shoe_path' not in st.session_state:
    st.session_state['selected_shoe_path'] = None
if 'feet_input_data' not in st.session_state:
    st.session_state['feet_input_data'] = None

# Sidebar untuk Model Debugging
with st.sidebar:
    st.title("⚙️ Pengaturan Model")
    netG = load_generator_model(MODEL_G_PATH)
    st.markdown("---")
    
    # DEBUG CONTROL KRITIS
    st.subheader("Debug Masker (7th Channel)")
    mask_option = st.selectbox(
        "7th Channel Mask Value:", 
        [-1.0, 0.0, 1.0],
        help="Pix2Pix Try-On memerlukan nilai spesifik di channel ke-7 (mask) agar bekerja. Coba ganti nilainya."
    )
    st.markdown("---")
    st.caption("Jika Anda melihat error 'File model tidak ditemukan', pastikan `pix2pix_tryon_G.h5` ada di folder yang sama.")

st.title("👟 Aplikasi Virtual Try-On Sepatu")
st.markdown("---")

def shoe_catalog(shoe_assets):
    st.header("1. Pilih Sepatu dari Katalog")
    st.markdown("*(Klik tombol 'Pilih' di bawah gambar)*")
    
    cols = st.columns(4)
    
    for i, shoe_path in enumerate(shoe_assets):
        with cols[i % 4]:
            shoe_name = os.path.basename(shoe_path)
            is_selected = (shoe_path == st.session_state.get('selected_shoe_path'))
            
            # Tampilkan gambar
            try:
                 st.image(shoe_path, caption="", use_column_width=True)
            except Exception:
                 st.image(Image.new('RGB', (IMG_SIZE, IMG_SIZE), color = 'black'), caption="File Hilang", use_column_width=True)
            
            button_label = "✅ Dipilih" if is_selected else "Pilih"
            button_type = "secondary" if is_selected else "primary"
            
            if st.button(button_label, key=f'select_{shoe_name}', type=button_type, use_container_width=True):
                st.session_state['selected_shoe_path'] = shoe_path
                st.session_state['feet_input_data'] = None # Reset input kaki saat ganti sepatu
                st.rerun() 

col_input, col_result = st.columns([1, 1], gap="large")

with col_input:
    shoe_assets = get_asset_paths('shoes')
    shoe_catalog(shoe_assets)

st.markdown("---") 

# --- Bagian Input Kaki dan Try-On ---
if st.session_state['selected_shoe_path']:
    
    with col_input:
        st.header("2. Sediakan Citra Kaki")
        
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
            # Pastikan path lengkap
            input_feet_data = os.path.join(os.getcwd(), 'assets', 'feet', selected_feet_name)
            
        # LOGIKA OPSI UNGGAH
        else:
            uploaded_file = st.file_uploader(
                "Unggah Citra Kaki (JPG/PNG)", 
                type=["jpg", "png", "jpeg"], 
                key='feet_uploader'
            )
            if uploaded_file is not None:
                input_feet_data = uploaded_file
            
        
        # Menampilkan citra kaki yang dipilih
        if input_feet_data is not None:
            st.markdown("---")
            st.subheader("Pratinjau Citra Kaki:")
            try:
                 # Menggunakan load_image untuk memastikan gambar terbaca
                loaded_img = load_image(input_feet_data)
                if loaded_img is not None:
                    st.image(loaded_img.astype(np.uint8), caption="Citra Kaki", use_column_width=True)
                    st.session_state['feet_input_data'] = input_feet_data 
            except Exception as e:
                st.warning(f"Tidak dapat menampilkan pratinjau gambar: {e}")
            st.markdown("---")
        else:
            st.session_state['feet_input_data'] = None

        st.markdown("<br>", unsafe_allow_html=True)
        # TOMBOL TRY-ON
        if st.session_state['feet_input_data']:
             if st.button("✨ Terapkan Virtual Try-On", key='tryon_button', type="primary", use_container_width=True):
                if netG is None:
                    col_result.error("Model Generator gagal dimuat. Tidak dapat melakukan inferensi.")
                else:
                    process_inference(
                        st.session_state['selected_shoe_path'], 
                        st.session_state['feet_input_data'], 
                        netG, col_result, 
                        mask_value # Gunakan nilai mask dari sidebar
                    )
        else:
             st.warning("Silakan pilih atau unggah citra kaki.")
else:
    # Tampilkan instruksi jika belum ada sepatu yang dipilih
    with col_result:
        st.header("Selamat Datang!")
        st.info("👈 Silakan klik salah satu tombol 'Pilih' di katalog (kolom kiri) untuk memulai Virtual Try-On.")
        st.markdown("""
        **Langkah Selanjutnya:**
        1. Pilih **Sepatu** dengan mengklik tombol 'Pilih'.
        2. Pilih sumber gambar **kaki** (Galeri atau Unggah).
        3. Coba ganti **'7th Channel Mask Value'** di **Sidebar** jika hasilnya putih.
        4. Klik tombol Try-On untuk melihat hasilnya di kolom ini.
        """)

# Tambahkan sedikit CSS custom untuk tampilan Streamlit yang lebih baik
st.markdown("""
<style>
.stButton>button {
    font-weight: bold;
    border-radius: 0.5rem;
    box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1), 0 2px 4px -2px rgba(0, 0, 0, 0.06);
}
.stImage img {
    border-radius: 8px;
    border: 1px solid #e0e0e0;
}
</style>
""", unsafe_allow_html=True)