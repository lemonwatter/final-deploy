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
# GANTI PATH INI DENGAN LOKASI FILE .h5 ANDA YANG BENAR
MODEL_G_PATH = 'models/pix2pix_tryon_G.h5' 

# Konfigurasi logging untuk debugging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# --- Konfigurasi Halaman Streamlit ---
st.set_page_config(
    page_title="Virtual Shoe Try-On App",
    layout="wide",
    initial_sidebar_state="expanded" 
)

# --- 1. Re-Definisi Arsitektur Model (Generator UNet) ---

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
    down_stack = [
        downsample(32, 4, apply_batchnorm=False), # L1 
        downsample(64, 4),                      # L2 
        downsample(128, 4),                     # L3 
        downsample(256, 4),                     # L4 
        downsample(512, 4, apply_batchnorm=False), # L5: Bottleneck 
    ]
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
    # Kode dummy/error handling tetap sama untuk stabilitas

    if not os.path.exists(model_path):
        st.warning(f"⚠️ File model tidak ditemukan di: `{model_path}`. Aplikasi akan menggunakan model dummy.")
        netG = GeneratorUNet()
        return netG
    
    try:
        netG = GeneratorUNet()
        netG.load_weights(model_path)
        logger.info("Model weights loaded successfully.")
        st.success("✅ Model Generator berhasil dimuat secara lokal.")
        return netG
    except Exception as e:
        st.error(f"❌ Gagal memuat model. Error: {e}")
        netG = GeneratorUNet()
        return netG

def get_asset_paths(folder_name):
    """Mendapatkan daftar path file gambar dari folder assets."""
    # Kode dummy/path handling tetap sama
    base_dir = os.path.join('.', 'assets', folder_name)
    if not os.path.isdir(base_dir):
        return [f"assets/{folder_name}/image_{i}.png" for i in range(4)]
        
    files = glob.glob(os.path.join(base_dir, '*.[jp][pn]g'), recursive=True)
    files.extend(glob.glob(os.path.join(base_dir, '*.jpeg'), recursive=True))
    return files

# --- 3. Pre-pemrosesan dan Inferensi ---

def normalize(image):
    return (image / 127.5) - 1

def load_image(image_data):
    try:
        if isinstance(image_data, str):
            if not os.path.exists(image_data):
                 logger.warning(f"Path asset tidak ditemukan: {image_data}")
                 img = Image.new('RGB', (IMG_SIZE, IMG_SIZE), color = 'red') 
            else:
                 img = Image.open(image_data).convert('RGB')
        elif isinstance(image_data, io.BytesIO) or hasattr(image_data, 'read'):
            img = Image.open(image_data).convert('RGB')
        else:
            raise ValueError("Input data gambar tidak valid.")

        img = img.resize((IMG_SIZE, IMG_SIZE))
        img = np.array(img, dtype=np.float32)
        return img
    except Exception as e:
        st.error(f"Gagal memuat atau memproses gambar: {e}")
        return None

def create_simple_mask(image_norm, mask_strategy):
    """
    Membuat mask biner berdasarkan strategi yang dipilih.
    image_norm dalam rentang [-1, 1].
    """
    if mask_strategy == 'Kontras Hitam/Putih (Asli)':
        # Strategi asli: Mengukur jarak dari abu-abu tengah (0)
        mean_abs_rgb = np.mean(np.abs(image_norm), axis=-1, keepdims=True)
        # Ambang batas 0.15 (sangat sensitif)
        mask = tf.cast(mean_abs_rgb > 0.15, tf.float32).numpy()
    
    elif mask_strategy == 'Putih vs Warna (Direkomendasikan)':
        # Strategi untuk gambar sepatu putih di latar belakang putih.
        # Fokus pada nilai channel B (biru) atau nilai yang TIDAK putih (kurang dari ambang batas tinggi).
        
        # Ambang batas 0.8 pada nilai norm [-1, 1] setara dengan ~229 pada [0, 255]
        # Kami mencari piksel yang BUKAN putih cerah.
        # Ambil nilai max dari R, G, B
        max_rgb = np.max(image_norm, axis=-1, keepdims=True)
        # Mask = area yang nilainya kurang dari 0.8 (bukan putih murni)
        mask = tf.cast(max_rgb < 0.80, tf.float32).numpy()
        
    # Normalisasi mask biner (0 atau 1) ke [-1, 1]
    mask = mask * 2 - 1 
    
    logger.info(f"Mask Min: {np.min(mask):.2f}, Mask Max: {np.max(mask):.2f}, Mask Mean: {np.mean(mask):.2f}")
    return mask

def process_inference(shoe_path, feet_data, netG, result_container, debug_mode, mask_strategy):
    """Memproses gambar sepatu dan kaki, melakukan inferensi, dan menampilkan hasil."""
    
    shoe_img = load_image(shoe_path)
    feet_img = load_image(feet_data)

    if shoe_img is None or feet_img is None:
        return 

    # Normalisasi
    shoe_norm = normalize(shoe_img) # Rentang [-1, 1]
    feet_norm = normalize(feet_img) # Rentang [-1, 1]
    
    # === PENGATURAN 7 CHANNEL INPUT: AUTO-MASKING DENGAN STRATEGI ===
    shoe_mask = create_simple_mask(shoe_norm, mask_strategy)
    
    # Gabungkan input (Sepatu 3ch + Kaki 3ch + Mask Sepatu 1ch)
    input_tensor_data = np.concatenate([shoe_norm, feet_norm, shoe_mask], axis=-1) 
    input_tensor = np.expand_dims(input_tensor_data, axis=0) # Tambah dimensi batch

    with result_container:
        if debug_mode:
            st.subheader("🛠️ Informasi Debugging Input & Output Mentah")
            
            # DEBUG 1: Tampilkan Mask Sepatu
            mask_display = (shoe_mask[:,:,0] * 0.5 + 0.5) * 255.0
            st.image(mask_display.astype(np.uint8), caption=f"DEBUG: Mask Sepatu ({mask_strategy})", clamp=True)
            st.warning("Pastikan mask mengisolasi area sepatu. Jika salah, coba strategi masking lain di sidebar.")

        with st.spinner('⏳ Sedang Menerapkan Try-On Virtual...'):
            try:
                # Inferensi
                prediction_raw = netG(input_tensor, training=False)[0].numpy()
                
                if debug_mode:
                    st.markdown("---")
                    st.subheader("DEBUG: Output Mentah Model (Rentang [-1, 1])")
                    st.code(f"""
    Min Output Mentah: {np.min(prediction_raw):.6f}
    Max Output Mentah: {np.max(prediction_raw):.6f}
    Mean Output Mentah: {np.mean(prediction_raw):.6f}
    
    Jika Min & Max mendekati 1, hasil akan putih total.
                    """)
                    st.markdown("---")
                
                # Denormalisasi ([-1, 1] -> [0, 255])
                prediction = (prediction_raw * 0.5 + 0.5) * 255.0
                prediction = prediction.clip(0, 255).astype(np.uint8)
                
                # Tampilkan hasil
                st.subheader("🎉 Hasil Virtual Try-On")
                st.image(prediction, caption=f"Hasil Try-On (Strategi: {mask_strategy})", use_column_width=True) 
                
            except Exception as e:
                st.error(f"❌ Terjadi kesalahan saat inferensi: {e}")

# ==============================================================================
# APLIKASI STREAMLIT UTAMA
# ==============================================================================

if 'selected_shoe_path' not in st.session_state:
    st.session_state['selected_shoe_path'] = None
if 'feet_input_data' not in st.session_state:
    st.session_state['feet_input_data'] = None
    
netG = load_generator_model(MODEL_G_PATH)

st.title("👟 Aplikasi Virtual Try-On Sepatu")
st.markdown("---")

# --- Bilah Sisi (Sidebar) untuk Debugging dan Strategi Masking ---
with st.sidebar:
    st.header("Opsi Debugging & Masking")
    
    mask_strategy = st.radio(
        "Pilih Strategi Auto-Masking:",
        ('Putih vs Warna (Direkomendasikan)', 'Kontras Hitam/Putih (Asli)'),
        key='mask_strategy_radio',
        help="Pilih 'Putih vs Warna' jika gambar sepatu Anda berlatar belakang putih. Pilih 'Kontras' jika berlatar belakang hitam."
    )
    
    debug_mode = st.checkbox("Aktifkan Mode Debugging", value=False)
    st.info("Gunakan ini untuk melihat hasil mask sepatu dan output mentah model.")


def shoe_catalog(shoe_assets):
    st.header("1. Pilih Sepatu dari Katalog")
    st.markdown("*(Klik tombol 'Pilih' di bawah gambar untuk mengaktifkan Try-On)*")
    
    cols = st.columns(4)
    
    for i, shoe_path in enumerate(shoe_assets[:4]):
        with cols[i % 4]:
            shoe_name = os.path.basename(shoe_path).split('.')[0]
            
            is_selected = (shoe_path == st.session_state.get('selected_shoe_path'))
            
            st.image(shoe_path, caption="", use_column_width=True)
            
            button_label = "✅ Dipilih" if is_selected else "Pilih"
            button_type = "secondary" if is_selected else "primary"
            
            if st.button(button_label, key=f'select_{shoe_name}', type=button_type, use_container_width=True):
                st.session_state['selected_shoe_path'] = shoe_path
                st.session_state['feet_input_data'] = None 
                st.rerun() 

col_input, col_result = st.columns([1, 1], gap="large")

with col_input:
    # Asset dummy (Ganti jika Anda memiliki folder assets/shoes)
    shoe_assets = get_asset_paths('shoes') or [f"https://placehold.co/256x256/2ecc71/ffffff?text=Sepatu+{i}" for i in range(4)]
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
            feet_assets = get_asset_paths('feet') or [f"https://placehold.co/256x256/3498db/ffffff?text=Kaki+{i}" for i in range(3)]
            feet_options = [os.path.basename(p) for p in feet_assets]
            feet_map = {os.path.basename(p): p for p in feet_assets}
            
            selected_feet_name = st.selectbox(
                "Pilih Bentuk Kaki Galeri:", 
                feet_options, 
                index=0, 
                key='select_feet_gallery'
            )
            input_feet_data = feet_map.get(selected_feet_name, selected_feet_name)

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
                st.image(input_feet_data, caption="Citra Kaki Input", use_column_width=True)
                st.session_state['feet_input_data'] = input_feet_data 
            except Exception as e:
                st.warning(f"Tidak dapat menampilkan pratinjau gambar: {e}")
            st.markdown("---")
        else:
            st.session_state['feet_input_data'] = None

        st.markdown("<br>", unsafe_allow_html=True)
        
        if st.button("✨ Terapkan Virtual Try-On", key='tryon_button', type="primary", use_container_width=True):
            if st.session_state['selected_shoe_path'] and st.session_state['feet_input_data']:
                process_inference(
                    st.session_state['selected_shoe_path'], 
                    st.session_state['feet_input_data'], 
                    netG, 
                    col_result, 
                    debug_mode,
                    mask_strategy # Kirim strategi masking yang dipilih
                )
            else:
                with col_result: 
                    st.warning("Mohon pilih sepatu dan sediakan citra kaki terlebih dahulu.")
else:
    with col_result:
        st.header("Selamat Datang!")
        st.info("👈 Silakan klik salah satu tombol 'Pilih' di katalog (kolom kiri) untuk memulai Virtual Try-On.")
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