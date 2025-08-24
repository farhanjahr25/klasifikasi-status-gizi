import streamlit as st
import pickle
import pandas as pd
import os  # untuk path dinamis gambar lokal
import xgboost as xgb
import json


def gizi_prediction_system():
    # Load model sebagai Booster
    gizi_model = xgb.Booster()
    gizi_model.load_model('xgboost_model.json')

    # Load nama fitur
    with open('feature_names.json', 'r') as f:
        feature_dict = json.load(f)
    feature_names = feature_dict['feature_names']

    # Load scaler
    scaler = pickle.load(open('scalernew.pkl', 'rb'))

    # Input pengguna
    jk = st.selectbox("Jenis Kelamin", ["Laki-laki", "Perempuan"])
    usia_saat_ukur = st.number_input("Usia saat Ukur (bulan)", min_value=0)
    bb_lahir = st.number_input("Berat Badan Lahir (kg)", min_value=0.0, step=0.1)
    tb_lahir = st.number_input("Tinggi Badan Lahir (cm)", min_value=0.0, step=0.1)
    berat = st.number_input("Berat Badan Saat Ini (kg)", min_value=0.0, step=0.1)
    tinggi = st.number_input("Tinggi Badan Saat Ini (cm)", min_value=0.0, step=0.1)
    lila = st.number_input("Lingkar Lengan Atas (cm)", min_value=0.0, step=0.1)

    jk_numeric = 1 if jk == "Laki-laki" else 0

    input_dict = {
        "jk": jk_numeric,
        "usia_saat_ukur": usia_saat_ukur,
        "bb_lahir": bb_lahir,
        "tb_lahir": tb_lahir,
        "berat": berat,
        "tinggi": tinggi,
        "lila": lila
    }


    # Tombol submit
    if st.button("Prediksi Status Gizi"):
        jk_numeric = 1 if jk == "Laki-laki" else 0

        input_dict = {
            "jk": jk_numeric,
            "usia_saat_ukur": usia_saat_ukur,
            "bb_lahir": bb_lahir,
            "tb_lahir": tb_lahir,
            "berat": berat,
            "tinggi": tinggi,
            "lila": lila
        }

        input_df = pd.DataFrame([input_dict])
        input_df = input_df[feature_names]

        input_data_scaled = scaler.transform(input_df)
        dmatrix_input = xgb.DMatrix(input_data_scaled, feature_names=feature_names)

        # Prediksi
        pred_probs = gizi_model.predict(dmatrix_input)
        gizi_prediction_index = pred_probs.argmax(axis=1)[0]

        label_mapping = {
            0: "Normal",
            1: "Overweight",
            2: "Stunting",
            3: "Underweight",
            4: "Wasting"
        }
        gizi_prediction = label_mapping.get(gizi_prediction_index, "Tidak diketahui")

        st.subheader("Hasil Prediksi Status Gizi Anak")
        st.write(f"Status Gizi Anak: **{gizi_prediction}**")

# Sidebar navigasi
st.sidebar.title("Menu")
page = st.sidebar.radio("Pilih Halaman:", ["Status Gizi", "Klasifikasi", "About Me"])


# Path gambar lokal
img_path = os.path.join(os.path.dirname(__file__), "umri.jpg")

# Halaman 1: Status Gizi
if page == "Status Gizi":
    st.title("Informasi Status Gizi")
    st.write("""
Masalah gizi pada anak merupakan isu kesehatan global yang memengaruhi pertumbuhan fisik dan mental. Di Indonesia, gizi buruk menjadi salah satu penyebab utama kematian dan keterlambatan perkembangan anak. Status gizi anak mencerminkan keseimbangan antara asupan nutrisi dan kebutuhan tubuh; ketidakseimbangan, baik kekurangan maupun kelebihan, berdampak langsung pada kesehatan anak.

Malnutrisi meningkatkan risiko infeksi, melemahkan sistem imun, memperlambat pemulihan, serta memengaruhi perkembangan kognitif dan kemampuan belajar anak. Penilaian status gizi, terutama pada balita, biasanya dilakukan melalui antropometri—mengukur berat badan, tinggi badan, dan usia anak sesuai pedoman Kementerian Kesehatan RI. Indeks antropometri (BB/U, TB/U, BB/TB) memberikan gambaran pertumbuhan anak dan mendeteksi risiko malnutrisi sejak dini.

Metode antropometri diakui secara internasional dan digunakan luas karena memberikan penilaian objektif. Hasil pengukuran dikonversi ke Z-score, yang menunjukkan penyimpangan data individu dari median populasi rujukan, dengan rentang umum -3 SD hingga +3 SD (WHO).
""")

    # Header baru: Jenis Masalah Gizi
    st.subheader("Jenis Masalah Gizi")
    st.write("""
Berdasarkan nilai Z-score dari masing-masing indeks antropometri, status gizi anak dapat diklasifikasikan sebagai berikut:
""")

    st.markdown("""
1. **Stunting (pendek)**  
   Diukur berdasarkan tinggi badan menurut umur (TB/U). Dikatakan stunting jika nilai z-score kurang dari -2 SD dari median standar pertumbuhan anak.

2. **Wasting (kurus)**  
   Diukur berdasarkan berat badan menurut tinggi badan (BB/TB). Dikatakan wasting jika nilai z-score kurang dari -2 SD dari median.

3. **Underweight (berat badan kurang)**  
   Diukur berdasarkan berat badan menurut umur (BB/U). Dikatakan underweight jika nilai z-score kurang dari -2 SD dari median.

4. **Overweight (berat badan lebih)**  
   Diukur berdasarkan berat badan menurut tinggi (BB/TB). Dikatakan overweight jika nilai z-score lebih dari +2 SD.
""")

    # Tabel contoh
    data = pd.DataFrame({
        'Nama': ['A', 'B', 'C'],
        'Umur (bulan)': [30, 12, 60],
        'Status Gizi': ['Normal', 'Stunting', 'Underweight']
    })
    st.table(data)
# Halaman 2: Klasifikasi Status Gizi
elif page == "Klasifikasi":
    st.title("Klasifikasi Status Gizi Anak")
    gizi_prediction_system()  # panggil fungsi di sini
# Halaman 3: About Me
elif page == "About Me":
    st.title("About Me")

    # Layout kolom
    col1, col2 = st.columns([1, 3])

    with col1:
        st.markdown(
            """
            <style>
                .profile-pic {
                    border-radius: 50%;
                    width: 150px;
                    height: 150px;
                    object-fit: cover;
                }
            </style>
            """,
            unsafe_allow_html=True
        )
        st.image("farhan.jpg", width=150, caption="")  # Pastikan file ada di folder yang sama

    with col2:
        st.write(
            "Halo! Saya **Farhan Jahr Daffa**, aplikasi untuk klasifikasi status gizi ini dibuat sebagai bagian dari penelitian skripsi yang berjudul **Klasifikasi Status Gizi Anak Menggunakan Algoritma XGBoost dan Framework Streamlit** untuk syarat kelulusan S1 Teknik Informatika."
        )

    
# Footer
st.write("---")
col1, col2 = st.columns([1, 5])
with col1:
    st.image(img_path, width=60)
with col2:
    st.markdown("""
**Farhan Jahr Daffa** | 210401201 | Universitas Muhammadiyah Riau | Teknik Informatika
""")
