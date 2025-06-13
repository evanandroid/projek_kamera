import pandas as pd
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import LabelEncoder, StandardScaler
import plotly.express as px
from sklearn.model_selection import train_test_split
import streamlit as st

# Fungsi format harga ke jutaan rupiah
def format_rupiah_juta(nilai_int):
    juta = nilai_int / 1_000_000_00
    if juta.is_integer():
        juta = int(juta)
    return f"Rp{juta} jt"

def format_megapixel(mp):
    try:
        return f"{mp:,.1f}".replace(",", "X").replace(".", ",").replace("X", ".")
    except:
        return "-"

# Fungsi konversi string harga jadi integer
def rupiah_to_int(rp_str):
    if isinstance(rp_str, str):
        clean_str = rp_str.replace("Rp", "").replace(".", "").replace(",", "")
        try:
            return int(clean_str)
        except:
            return 0
    return 0

# LOAD & PREPROCESS DATA
df = pd.read_csv("camera_dataset.csv", sep=';')
df['price_int'] = df['price'].apply(rupiah_to_int)

df['megapixel'] = df['megapixel'].astype(str).str.replace(",", ".", regex=False)
df['megapixel'] = df['megapixel'].replace('[^0-9.]', '', regex=True)
df['megapixel'] = pd.to_numeric(df['megapixel'], errors='coerce')

df = df.dropna(subset=['camera_name', 'camera_type', 'megapixel', 'price_int', 'usage'])

le_type = LabelEncoder()
le_usage = LabelEncoder()
df['camera_type'] = df['camera_type'].astype(str)
df['usage'] = df['usage'].astype(str)
df['type_enc'] = le_type.fit_transform(df['camera_type'])
df['usage_enc'] = le_usage.fit_transform(df['usage'])

X = df[['megapixel', 'price_int', 'type_enc', 'usage_enc']]
y = df['camera_name']

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Bagi data dan latih model
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)
knn = KNeighborsClassifier(n_neighbors=3)
knn.fit(X_train, y_train)

# Evaluasi akurasi model
accuracy = knn.score(X_test, y_test)

# STREAMLIT UI
st.set_page_config(page_title="Rekomendasi Kamera AI", layout="wide")
st.title("🤖 Rekomendasi Kamera Berbasis AI (KNN)")
st.write("Temukan kamera yang cocok berdasarkan megapiksel, anggaran, jenis kamera, dan tujuan penggunaan.")

# Sidebar input
st.sidebar.header("🎛️ Filter Preferensi Anda")
megapixels = st.sidebar.slider("Megapiksel yang Diinginkan", 8, 35, 24)
price_million = st.sidebar.slider("Anggaran Maksimal (Juta Rupiah)", 2, 25, 10)
camera_type = st.sidebar.selectbox("Jenis Kamera", le_type.classes_)
usage = st.sidebar.selectbox("Kebutuhan Penggunaan", le_usage.classes_)

# Tampilkan akurasi model di sidebar
st.sidebar.markdown(f"🎯 Akurasi Model (Data Uji): **{accuracy:.2%}**")

# Proses input user
price_int = price_million * 1_000_000_00
type_encoded = le_type.transform([camera_type])[0]
usage_encoded = le_usage.transform([usage])[0]
user_input = [[megapixels, price_int, type_encoded, usage_encoded]]
user_input_scaled = scaler.transform(user_input)

# Salin data untuk visualisasi
df_plot = df.copy()
df_plot['price_million'] = df_plot['price_int'] / 1_000_000_00

# Rekomendasi dan visualisasi
if st.sidebar.button("🔍 Rekomendasikan Kamera"):
    import numpy as np

    # Hitung jarak Euclidean ke semua kamera
    distances = np.linalg.norm(X_scaled - user_input_scaled, axis=1)
    df_plot['euclidean_distance'] = distances
    df_sorted = df_plot.sort_values(by='euclidean_distance').reset_index(drop=True)
    
    # Ambil 3 kamera terdekat
    recommended_cameras = df_sorted.head(3)

    # Tampilkan rekomendasi kamera
    st.subheader("📸 Kamera yang Direkomendasikan:")
    for _, row in recommended_cameras.iterrows():
        st.markdown(f"""
         <div style="border:1px solid #ddd; border-radius:10px; padding:15px; margin-bottom:10px; display: flex; align-items: center;">
            <img src="{row['image_url']}" style="width:150px; height:auto; margin-right:15px; border-radius:10px;" alt="kamera">
            <div>
                <h4 style="margin:0;">{row['camera_name']}</h4>
                <ul>
                    <li><strong>Jenis:</strong> {row['camera_type']}</li>
                    <li><strong>Megapiksel:</strong> {format_megapixel(row['megapixel'])} MP</li>
                    <li><strong>Harga:</strong> {format_rupiah_juta(row['price_int'])}</li>
                    <li><strong>Penggunaan:</strong> {row['usage']}</li>
                    <li><strong>Jarak Euclidean:</strong> {row['euclidean_distance']:.4f}</li>
                </ul>
            </div>
         </div>
        """, unsafe_allow_html=True)

    # Tampilkan tabel semua jarak (opsional)
    st.subheader("📊 Tabel Jarak Euclidean ke Semua Kamera")
    st.dataframe(df_sorted[['camera_name', 'megapixel', 'price_int', 'camera_type', 'usage', 'euclidean_distance']])

# Plot scatter semua kamera
fig = px.scatter(
    df_plot,
    x='megapixel',
    y='price_million',
    color='usage',
    hover_name='camera_name',
    hover_data={
        'camera_type': True,
        'megapixel': ':.1f',
        'price_million': ':,.2f',
        'usage': True,
    },
    title="Visualisasi Kamera dan Titik Input Pengguna"
)

# Tambahkan titik input user ke dalam grafik
fig.add_scatter(
    x=[megapixels],
    y=[price_million],
    mode='markers+text',
    marker=dict(size=12, color='yellow', symbol='x'),
    text=["Input Anda"],
    textposition='bottom right',
    name='Input Pengguna'
)

# Tambahkan titik rekomendasi dan garis, jika sudah dihitung
if 'recommended_cameras' in locals():
    fig.add_scatter(
        x=recommended_cameras['megapixel'],
        y=recommended_cameras['price_int'] / 1_000_000_00,
        mode='markers+text',
        marker=dict(size=10, color='green', symbol='star'),
        text=recommended_cameras['camera_name'],
        textposition='top center',
        name='Rekomendasi'
    )

    for _, row in recommended_cameras.iterrows():
        fig.add_scatter(
            x=[megapixels, row['megapixel']],
            y=[price_million, row['price_int'] / 1_000_000_00],
            mode='lines',
            line=dict(color='gray', width=1, dash='dot'),
            showlegend=False
        )

# Tampilkan grafik
st.plotly_chart(fig, use_container_width=True)

