import pandas as pd
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import accuracy_score, confusion_matrix
import plotly.express as px
import streamlit as st
from sklearn.model_selection import train_test_split

# Helper
def rupiah_to_int(rp_str):
    if isinstance(rp_str, str):
        clean_str = rp_str.replace("Rp", "").replace(".", "").replace(",", "")
        return int(clean_str) if clean_str.isdigit() else 0
    return 0

# Load & Prepare Data
df = pd.read_csv("camera_dataset.csv", sep=';')
df['price_int'] = df['price'].apply(rupiah_to_int)
df['megapixel'] = df['megapixel'].astype(str).str.replace(",", ".", regex=False)
df['megapixel'] = df['megapixel'].replace('[^0-9.]', '', regex=True)
df['megapixel'] = pd.to_numeric(df['megapixel'], errors='coerce')
df = df.dropna(subset=['camera_name', 'megapixel', 'price_int', 'usage'])

le_usage = LabelEncoder()
df['usage_enc'] = le_usage.fit_transform(df['usage'].astype(str))

X = df[['megapixel', 'price_int']]
y = df['usage_enc']
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Train-Test Split
X_train, X_test, y_train, y_test, idx_train, idx_test = train_test_split(
    X_scaled, y, df.index, test_size=0.2, random_state=42
)

# UI
st.set_page_config(page_title="Klasifikasi Kamera", layout="wide")
st.title("📷 Klasifikasi Kebutuhan Kamera (KNN + Euclidean)")
st.sidebar.header("🎛️ Kontrol")

# Sidebar Inputs
megapixels = st.sidebar.slider("Megapiksel", 8, 35, 24)
price_million = st.sidebar.slider("Harga Maksimal (juta)", 2, 25, 10)
k_value = st.sidebar.slider("Jumlah Tetangga (k)", 1, 15, 3)

price_int = price_million * 1_000_000_00
user_input = [[megapixels, price_int]]
user_input_scaled = scaler.transform(user_input)

# Tombol aksi
run_predict = st.sidebar.button("🔍 Prediksi dari Input Pengguna")
run_evaluation = st.sidebar.button("📊 Evaluasi Model")

# Model awal
knn = KNeighborsClassifier(n_neighbors=k_value)
knn.fit(X_train, y_train)
y_pred = knn.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
conf_matrix = confusion_matrix(y_test, y_pred)

if run_predict:
    # Prediksi berdasarkan input pengguna
    pred_usage_enc = knn.predict(user_input_scaled)[0]
    pred_usage = le_usage.inverse_transform([pred_usage_enc])[0]

    st.subheader("🎯 Hasil Klasifikasi Kebutuhan Kamera:")
    st.success(f"Prediksi penggunaan: **{pred_usage}**")

    # Euclidean distance
    distances = np.linalg.norm(X_scaled - user_input_scaled, axis=1)
    df['euclidean_distance'] = distances
    df_sorted = df.sort_values(by='euclidean_distance').reset_index(drop=True)
    nearest_neighbors = df_sorted.head(k_value)

    st.info(f"Hasil voting dari {k_value} tetangga terdekat: **{nearest_neighbors['usage'].mode().iloc[0]}**")

    st.subheader("📊 Tabel Jarak Euclidean")
    st.dataframe(df_sorted[['camera_name', 'megapixel', 'price_int', 'usage', 'euclidean_distance']])

    # Visualisasi
    df['price_million'] = df['price_int'] / 1_000_000_00
    fig = px.scatter(
        df,
        x='megapixel',
        y='price_million',
        color='usage',
        hover_name='camera_name'
    )

    fig.add_scatter(
        x=[megapixels],
        y=[price_million],
        mode='markers+text',
        marker=dict(size=12, color='yellow', symbol='x'),
        text=["Input Anda"],
        textposition='bottom right',
        name='Input'
    )

    fig.add_scatter(
        x=nearest_neighbors['megapixel'],
        y=nearest_neighbors['price_int'] / 1_000_000_00,
        mode='markers+text',
        marker=dict(size=10, color='green', symbol='star'),
        text=nearest_neighbors['camera_name'],
        textposition='top center',
        name=f'{k_value} Tetangga'
    )

    for _, row in nearest_neighbors.iterrows():
        fig.add_scatter(
            x=[megapixels, row['megapixel']],
            y=[price_million, row['price_int'] / 1_000_000_00],
            mode='lines',
            line=dict(color='gray', width=1, dash='dot'),
            showlegend=False
        )

    st.plotly_chart(fig, use_container_width=True)

elif run_evaluation:
    st.subheader("📈 Evaluasi Model KNN")
    st.write(f"Akurasi: **{accuracy:.2%}**")

    cm_df = pd.DataFrame(conf_matrix, index=le_usage.classes_, columns=le_usage.classes_)
    st.write("Confusion Matrix:")
    st.dataframe(cm_df)

    st.subheader("📂 Data Training")
    st.dataframe(df.loc[idx_train, ['camera_name', 'megapixel', 'price_int', 'usage']])

    st.subheader("🧪 Data Testing")
    st.dataframe(df.loc[idx_test, ['camera_name', 'megapixel', 'price_int', 'usage']])
else:
    st.info("Klik salah satu tombol di sidebar untuk menjalankan prediksi atau evaluasi.")
