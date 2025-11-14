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

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

st.set_page_config(
    page_title="Virtual Shoe Try-On App",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- 1. Definisi Arsitektur Model (Generator UNet) ---
# Arsitektur disinkronkan dengan training ringan (5 Layer, filter 32/64)
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
    """Generator U-Net 5 Lapisan/Layer (Sama dengan kode training terakhir)"""
    inputs = Input(shape=input_shape) 
    # Downsampling 5 Layer
    down_stack = [
        downsample(32, 4, apply_batchnorm=False), 
        downsample(64, 4),                        
        downsample(128, 4),                       
        downsample(256, 4),                       
        downsample(512, 4), # Bottleneck
    ]
    # Upsampling 4 Layer
    up_stack = [
        upsample(256, 4, apply_dropout=True), 
        upsample(128, 4),                       
        upsample(64, 4),                        
        upsample(32, 4),                        
    ]
    
    initializer = tf.random_normal_initializer(0., 0.02)
    last = Conv2DTranspose(output_channels, 4, strides=2, padding='same',
                            kernel_initializer=initializer, activation='tanh')

    x = inputs
    skips = []
    
    # Encoder
    for down in down_stack:
        x = down(x)
        skips.append(x)
    
    skips = reversed(skips[:-1]) # skips [512, 256, 128, 64]

    # Decoder (Manual Skip Connections)
    x = skips[0] 
    x = up_stack[0](x)
    x = Concatenate()([x, skips[1]]) # Skip 1 (dari Lapis 4 Down)

    for up, skip_idx in zip(up_stack[1:], [2, 3]): 
        x = up(x)
        x = Concatenate()([x, skips[skip_idx]])

    x = last(x)

    return Model(inputs=inputs, outputs=x, name='Generator')

# --- 2. Fungsi Pemuatan Model dan Aset ---

@st.cache_resource
def load_generator_model(model_path):
    """Memuat model generator (netG) dari path lokal."""
    
    # 1. Coba memuat model penuh/bobot (untuk Git LFS)
    if os.path.exists(model_path):
        try:
            netG = tf.keras.models.load_model(
                model_path, 
                compile=False, 
                custom_objects={'LeakyReLU': tf.keras.layers.LeakyReLU}
            )
            st.success("✅ Model Generator berhasil dimuat.")
            return netG
        except Exception as e:
            logger.error(f"Gagal memuat model penuh/bobot: {e}")
            st.error(f"❌ Gagal memuat model. Error: {e}")
            st.error("Ini mungkin berarti file model corrupt atau Git LFS gagal mengunduh data biner.")
            return None
    
    st.error(f"❌ KESALAHAN KRITIS: File model tidak ditemukan di: {model_path}.")
    st.error("Pastikan file model (200MB+) ada di `models/` dan di-*commit* menggunakan **Git LFS**.")
    return None


@lru_cache(maxsize=32)
def get_asset_paths(folder_name):
    """Mendapatkan daftar path file gambar dari folder assets."""
    asset_dir = os.path.join('assets', folder_name)
    if not os.path.isdir(asset_dir):
        os.makedirs(asset_dir, exist_ok=True)
        return ["_mock_file_placeholder.png"] 
        
    files = glob.glob(os.path.join(asset_dir, '*.[jp][pn]g'), recursive=True)
    files.extend(glob.glob(os.path.join(asset_dir, '*.jpeg'), recursive=True))
    return files if files else ["_mock_file_placeholder.png"]

# --- 3. Pre-pemrosesan dan Inferensi ---

def normalize(image):
    return (image / 127.5) - 1

def denormalize(prediction):
    prediction = (prediction * 0.5 + 0.5) * 255.0
    return prediction.clip(0, 255).astype(np.uint8)

def load_image(image_data):
    """Mengubah data gambar menjadi tensor yang diproses."""
    if isinstance(image_data, str) and os.path.exists(image_data) and "_mock_file_placeholder" not in image_data:
        img = Image.open(image_data).convert('RGB')
    elif hasattr(image_data, 'read'): 
        img = Image.open(image_data).convert('RGB')
    else:
        return None

    img = img.resize((IMG_SIZE, IMG_SIZE))
    img = np.array(img, dtype=np.float32)
    return img

def create_mask_from_shoe(shoe_img_norm):
    shoe_mask = np.mean(shoe_img_norm, axis=-1, keepdims=True)
    return np.where(np.abs(shoe_mask) > 0.1, 1.0, 0.0).astype(np.float32)

def blend_result_with_feet(feet_original_img, generated_shoe_img):
    
    generated_gray = np.mean(generated_shoe_img, axis=-1) / 255.0
    
    # Masker: Asumsi area yang digenerate adalah area yang ingin kita pertahankan
    mask = (generated_gray < 0.85).astype(np.float32) 
    mask = np.expand_dims(mask, axis=-1) 

    foreground = generated_shoe_img * mask
    background = feet_original_img * (1 - mask)
    
    blended_image = foreground + background
    
    return blended_image.clip(0, 255).astype(np.uint8)


def process_inference(shoe_path, feet_data, netG, result_container, mask_source, mask_value):
    
    shoe_img_orig = load_image(shoe_path)
    feet_img_orig = load_image(feet_data)

    if shoe_img_orig is None or feet_img_orig is None:
        result_container.error("Gagal memuat atau memproses salah satu gambar.")
        return 

    shoe_norm = normalize(shoe_img_orig) 
    feet_norm = normalize(feet_img_orig) 
    
    if mask_source == "Masker Sepatu (Berdasarkan Warna Sepatu)":
        mask_channel_float = create_mask_from_shoe(shoe_norm)
        mask_label = "Shoe Channel Mask"
    else:
        mask_channel_float = np.full((IMG_SIZE, IMG_SIZE, 1), mask_value, dtype=np.float32) 
        mask_label = f"Fixed Value: {mask_value}"
    
    input_tensor = np.concatenate([shoe_norm, feet_norm, mask_channel_float], axis=-1) 
    input_tensor = np.expand_dims(input_tensor, axis=0)

    with result_container:
        with st.spinner(f'⏳ Sedang Menerapkan Try-On Virtual (Mask Source: {mask_label})...'):
            try:
                prediction_norm = netG(input_tensor, training=False)[0].numpy()
                generated_shoe_img_uint8 = denormalize(prediction_norm)
                
                final_output = blend_result_with_feet(feet_img_orig, generated_shoe_img_uint8)
                
                st.subheader("🎉 Hasil Virtual Try-On (dengan Blending)")
                st.image(final_output, caption=f"Hasil Try-On (Mask Source: {mask_label})", use_column_width=True) 
                
                # --- TAMPILKAN DEBUGGING ---
                st.markdown("---")
                st.subheader("🛠️ Debugging Output")
                cols = st.columns(2)
                with cols[0]:
                    st.image(generated_shoe_img_uint8, caption="Output Mentah Model (Generated Shoe)", use_column_width=True)
                    
                with cols[1]:
                    if mask_source == "Masker Sepatu (Berdasarkan Warna Sepatu)":
                        mask_display = (mask_channel_float * 127.5 + 127.5).astype(np.uint8)
                    else:
                        mask_display = np.full((IMG_SIZE, IMG_SIZE, 1), (mask_value * 127.5 + 127.5), dtype=np.uint8)
                        
                    st.image(mask_display, caption=f"Input Masker ke-7 ({mask_label})", use_column_width=True)


            except Exception as e:
                st.error(f"❌ Terjadi kesalahan saat inferensi atau blending: {e}")
                logger.error(f"Inferensi error: {e}")

# ==============================================================================
# APLIKASI STREAMLIT UTAMA
# ==============================================================================

# Inisialisasi State (tetap sama)
if 'selected_shoe_path' not in st.session_state:
    st.session_state['selected_shoe_path'] = None
if 'feet_input_data' not in st.session_state:
    st.session_state['feet_input_data'] = None
if 'mask_value' not in st.session_state:
    st.session_state['mask_value'] = -1.0 
if 'mask_source' not in st.session_state:
    st.session_state['mask_source'] = "Masker Sepatu (Berdasarkan Warna Sepatu)"

# Sidebar untuk Model Debugging
with st.sidebar:
    st.title("⚙️ Pengaturan Model")
    
    # Muat Model
    netG = load_generator_model(MODEL_G_PATH)
    
    st.markdown("---")
    
    # DEBUG CONTROL KRITIS
    st.subheader("Debug Masker (7th Channel)")
    
    mask_source_options = [
        "Masker Sepatu (Berdasarkan Warna Sepatu)", 
        "Nilai Tetap (-1.0)", 
        "Nilai Tetap (0.0)", 
        "Nilai Tetap (1.0)",
        "Zero Mask (All 0s)", 
        "One Mask (All 1s)"  
    ]
    
    mask_source = st.selectbox(
        "7th Channel Mask Source:", 
        mask_source_options,
        index=0,
        key='mask_source_selector',
        help="Pilih sumber untuk channel ke-7. Jika hasil putih/abu-abu, coba ganti opsi ini."
    )
    st.session_state['mask_source'] = mask_source

    fixed_mask_value = None
    if "Nilai Tetap" in mask_source or "Mask" in mask_source:
        if "Nilai Tetap" in mask_source:
            fixed_mask_value = float(mask_source.split('(')[1].strip(')'))
        elif mask_source == "Zero Mask (All 0s)":
            fixed_mask_value = 0.0
        elif mask_source == "One Mask (All 1s)":
            fixed_mask_value = 1.0
             
        st.session_state['mask_value'] = fixed_mask_value
    
    st.markdown("---")
    st.caption("Pastikan file model besar Anda di-*commit* menggunakan **Git LFS**.")

# Kolom UI Utama (tetap sama)
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
            
            if shoe_name == "_mock_file_placeholder.png":
                st.image(Image.new('RGB', (IMG_SIZE, IMG_SIZE), color = '#ff4b4b'), caption="MOCK: File Hilang", use_column_width=True)
            else:
                try:
                    st.image(shoe_path, caption="", use_column_width=True)
                except Exception:
                    st.image(Image.new('RGB', (IMG_SIZE, IMG_SIZE), color = '#ff4b4b'), caption="File Hilang", use_column_width=True)
            
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

if st.session_state['selected_shoe_path'] and st.session_state['selected_shoe_path'] != "_mock_file_placeholder.png":
    
    with col_input:
        st.header("2. Sediakan Citra Kaki")
        
        input_method = st.radio(
            "Pilih Metode Input Kaki:",
            ("Pilih dari Galeri", "Unggah Citra Kaki Sendiri"),
            key='input_method_radio'
        )
        
        input_feet_data = None
        
        col_feet_input, col_feet_preview = st.columns(2)

        with col_feet_input:
            if input_method == "Pilih dari Galeri":
                feet_assets = get_asset_paths('feet')
                feet_options = [os.path.basename(p) for p in feet_assets if p != "_mock_file_placeholder.png"]

                if feet_options:
                    selected_feet_name = st.selectbox(
                        "Pilih Bentuk Kaki Galeri:", 
                        feet_options, 
                        index=0, 
                        key='select_feet_gallery'
                    )
                    input_feet_data = os.path.join(os.getcwd(), 'assets', 'feet', selected_feet_name)
                else:
                    st.warning("Tidak ada gambar kaki di galeri.")
            
            else:
                uploaded_file = st.file_uploader(
                    "Unggah Citra Kaki (JPG/PNG)", 
                    type=["jpg", "png", "jpeg"], 
                    key='feet_uploader'
                )
                if uploaded_file is not None:
                    input_feet_data = uploaded_file

        
        with col_feet_preview:
            if input_feet_data is not None:
                st.markdown("**Pratinjau Kaki:**")
                try:
                    loaded_img = load_image(input_feet_data)
                    if loaded_img is not None:
                        display_img = loaded_img.astype(np.uint8)
                        st.image(display_img, caption="Citra Kaki", use_column_width=True)
                        st.session_state['feet_input_data'] = input_feet_data 
                except Exception as e:
                    st.warning(f"Tidak dapat menampilkan pratinjau gambar: {e}")
            else:
                 st.session_state['feet_input_data'] = None

        
        st.markdown("<br>", unsafe_allow_html=True)
        # TOMBOL TRY-ON
        if st.session_state['feet_input_data']:
            if st.button("✨ Terapkan Virtual Try-On", key='tryon_button', type="primary", use_container_width=True):
                if netG is None:
                    col_result.error("Model Generator gagal dimuat. Harap periksa path dan keberadaan file model.")
                else:
                    process_inference(
                        st.session_state['selected_shoe_path'], 
                        st.session_state['feet_input_data'], 
                        netG, col_result, 
                        st.session_state['mask_source'],
                        st.session_state.get('mask_value', -1.0)
                    )
        else:
            st.warning("Silakan pilih atau unggah citra kaki.")
else:
    with col_result:
        st.header("Selamat Datang!")
        st.info("👈 Silakan klik salah satu tombol 'Pilih' di katalog (kolom kiri) untuk memulai Virtual Try-On.")
        st.markdown("""
        **Langkah Selanjutnya:**
        1. Pilih **Sepatu** dengan mengklik tombol 'Pilih'.
        2. Pilih sumber gambar **kaki** (Galeri atau Unggah).
        3. Coba ganti **'7th Channel Mask Source'** di **Sidebar** jika hasilnya tidak benar.
        4. Klik tombol Try-On untuk melihat hasilnya di kolom ini.
        """)

# CSS custom untuk tampilan Streamlit
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