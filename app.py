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
# PATH MODEL: models/pix2pix_tryon_G.h5 (Sesuai dengan struktur folder Anda)
MODEL_G_PATH = 'models/pix2pix_tryon_G.h5' 

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

    down_stack = [
        downsample(32, 4, apply_batchnorm=False), # L1
        downsample(64, 4),                       # L2
        downsample(128, 4),                      # L3
        downsample(256, 4),                      # L4
        downsample(512, 4, apply_batchnorm=False), # L5: Bottleneck
    ]

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
    
    # Koneksi skip
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
        st.error(f"❌ KESALAHAN KRITIS: File model tidak ditemukan di: {model_path}.")
        st.error("Pastikan Anda menggunakan **Git LFS** untuk model besar dan sudah di-*commit* dan di-*push*.")
        return None
    
    try:
        netG = tf.keras.models.load_model(model_path, compile=False, custom_objects={'LeakyReLU': LeakyReLU})
        st.success("✅ Model Generator berhasil dimuat menggunakan tf.keras.models.load_model.")
        return netG
    except Exception as e:
        logger.warning(f"Gagal memuat model penuh: {e}. Mencoba memuat weights ke arsitektur kustom.")
        try:
            netG = GeneratorUNet()
            dummy_input = np.zeros((1, IMG_SIZE, IMG_SIZE, 7), dtype=np.float32)
            netG(dummy_input) 
            netG.load_weights(model_path)
            st.success("✅ Model dimuat sukses dengan memuat weights.")
            return netG
        except Exception as e_fallback:
            logger.error(f"FAILED TO LOAD MODEL (Weights/Architecture): {e_fallback}")
            st.error(f"❌ GAGAL TOTAL: Gagal memuat model. Error: {e_fallback}")
            return None

@lru_cache(maxsize=32)
def get_asset_paths(folder_name):
    """Mendapatkan daftar path file gambar dari folder assets."""
    base_dir = os.path.join(os.getcwd(), 'assets', folder_name)
    files = glob.glob(os.path.join(base_dir, '*.[jp][pn]g'), recursive=True)
    files.extend(glob.glob(os.path.join(base_dir, '*.jpeg'), recursive=True))

    if not files:
        if not os.path.isdir(base_dir):
            os.makedirs(base_dir, exist_ok=True)
        # Tambahkan mock files agar UI tetap berjalan saat deployment pertama kali
        return ["_mock_file_placeholder.png"]
         
    return files

# --- 3. Pre-pemrosesan dan Inferensi ---

def normalize(image):
    return (image / 127.5) - 1

def denormalize(prediction):
    prediction = (prediction * 0.5 + 0.5) * 255.0
    return prediction.clip(0, 255).astype(np.uint8)

def load_image(image_data):
    """Mengubah data gambar menjadi tensor yang diproses."""
    # Menangani path string (termasuk mock)
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
    """Membuat mask sederhana dari gambar sepatu yang dinormalisasi."""
    # Mengambil rata-rata 3 channel, lalu mengubah ke 1 channel mask
    shoe_mask = np.mean(shoe_img_norm, axis=-1, keepdims=True)
    return shoe_mask

def blend_result_with_feet(feet_original_img, generated_shoe_img):
    """
    Menggabungkan (overlay) hasil prediksi (sepatu + background) 
    ke gambar kaki asli (feet_original_img).
    """
    
    # 1. Tentukan area yang akan di-overlay (area yang bukan latar belakang)
    # Asumsikan latar belakang putih atau abu-abu terang (nilai tinggi di [0, 255])
    # Kita buat thresholding sederhana: piksel gelap (sepatu) vs piksel terang (background)
    
    # Konversi ke skala abu-abu dan normalisasi ke [0, 1]
    generated_gray = np.mean(generated_shoe_img, axis=-1) / 255.0
    
    # Thresholding: 0.8 biasanya berfungsi untuk memisahkan objek gelap dari latar belakang terang
    # Area sepatu (foreground) akan bernilai 1.0, area background 0.0
    mask = (generated_gray < 0.8).astype(np.float32)
    mask = np.expand_dims(mask, axis=-1) # Kembalikan ke (H, W, 1)

    # 2. Blend: (Foreground * Mask) + (Background * (1 - Mask))
    
    # Sepatu yang dihasilkan (Foreground) dikalikan dengan Mask
    foreground = generated_shoe_img * mask

    # Kaki asli (Background) dikalikan dengan kebalikan dari Mask (area non-sepatu)
    background = feet_original_img * (1 - mask)
    
    # 3. Gabungkan
    blended_image = foreground + background
    
    return blended_image.clip(0, 255).astype(np.uint8)


def process_inference(shoe_path, feet_data, netG, result_container, mask_source, mask_value):
    """Memproses gambar, inferensi, dan menampilkan hasil."""
    
    shoe_img_orig = load_image(shoe_path)
    feet_img_orig = load_image(feet_data)

    if shoe_img_orig is None or feet_img_orig is None:
        result_container.error("Gagal memuat atau memproses salah satu gambar. Pastikan gambar ada dan valid.")
        return 

    # Normalisasi untuk MODEL
    shoe_norm = normalize(shoe_img_orig) 
    feet_norm = normalize(feet_img_orig) 
    
    # --- PENGATURAN MASK 7th CHANNEL (PILIHAN PENGGUNA) ---
    if mask_source == "Masker Sepatu (Berdasarkan Warna Sepatu)":
        mask_channel_float = create_mask_from_shoe(shoe_norm)
        mask_label = "Shoe Channel Mask"
    else:
        mask_channel_float = np.full((IMG_SIZE, IMG_SIZE, 1), mask_value, dtype=np.float32) 
        mask_label = f"Fixed Value: {mask_value}"
    
    # Concatenation: (H, W, 3 Sepatu) + (H, W, 3 Kaki) + (H, W, 1 Mask) = (H, W, 7)
    input_tensor = np.concatenate([shoe_norm, feet_norm, mask_channel_float], axis=-1) 
    input_tensor = np.expand_dims(input_tensor, axis=0) # Tambah dimensi batch (1, H, W, 7)

    with result_container:
        with st.spinner(f'⏳ Sedang Menerapkan Try-On Virtual (Mask Source: {mask_label})...'):
            try:
                # 1. Inferensi Model (Output 3 channel: Sepatu + background)
                prediction_norm = netG(input_tensor, training=False)[0].numpy()
                
                # 2. Denormalisasi Output (Rentang [0, 255])
                generated_shoe_img_uint8 = denormalize(prediction_norm)
                
                # 3. Blending (Overlay) ke Gambar Kaki Asli
                final_output = blend_result_with_feet(feet_img_orig, generated_shoe_img_uint8)
                
                # Tampilkan hasil
                st.subheader("🎉 Hasil Virtual Try-On (dengan Blending)")
                st.image(final_output, caption=f"Hasil Try-On (Mask Source: {mask_label})", use_column_width=True) 
                st.balloons()
                
            except Exception as e:
                st.error(f"❌ Terjadi kesalahan saat inferensi atau blending: {e}")
                logger.error(f"Inferensi error: {e}")

# ==============================================================================
# APLIKASI STREAMLIT UTAMA
# ==============================================================================

# Inisialisasi State
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
    
    mask_source = st.selectbox(
        "7th Channel Mask Source:", 
        ["Masker Sepatu (Berdasarkan Warna Sepatu)", "Nilai Tetap (-1.0)", "Nilai Tetap (0.0)", "Nilai Tetap (1.0)"],
        index=0,
        key='mask_source_selector',
        help="Pilih sumber untuk channel ke-7. Opsi 'Masker Sepatu' menggunakan citra sepatu sebagai sinyal posisi."
    )
    st.session_state['mask_source'] = mask_source

    fixed_mask_value = None
    if "Nilai Tetap" in mask_source:
        fixed_mask_value = float(mask_source.split('(')[1].strip(')'))
        st.session_state['mask_value'] = fixed_mask_value
    
    st.markdown("---")
    st.caption("Pastikan file model besar Anda di-*commit* menggunakan **Git LFS**.")

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

# --- Bagian Input Kaki dan Try-On ---
if st.session_state['selected_shoe_path'] and st.session_state['selected_shoe_path'] != "_mock_file_placeholder.png":
    
    with col_input:
        st.header("2. Sediakan Citra Kaki")
        
        input_method = st.radio(
            "Pilih Metode Input Kaki:",
            ("Pilih dari Galeri", "Unggah Citra Kaki Sendiri"),
            key='input_method_radio'
        )
        
        input_feet_data = None
        
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
                 st.warning("Tidak ada gambar kaki di galeri. Harap unggah manual atau tambahkan file ke `assets/feet`.")
            
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
                loaded_img = load_image(input_feet_data)
                if loaded_img is not None:
                    display_img = loaded_img.astype(np.uint8)
                    st.image(display_img, caption="Citra Kaki", use_column_width=True)
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