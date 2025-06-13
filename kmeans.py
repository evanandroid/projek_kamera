import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score
from scipy.cluster.hierarchy import linkage, dendrogram
import streamlit as st

# Load data
df = pd.read_csv("camera_dataset.csv", sep=";")
df['price_int'] = df['price'].replace('[Rp.,]', '', regex=True).astype(str).replace('', '0').astype(int)
df['megapixel'] = df['megapixel'].astype(str).str.replace(",", ".").replace('[^0-9.]', '', regex=True)
df['megapixel'] = pd.to_numeric(df['megapixel'], errors='coerce')
df.dropna(subset=['megapixel', 'price_int'], inplace=True)

# Preprocessing
X = df[['megapixel', 'price_int']]
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Streamlit UI
st.set_page_config(page_title="Evaluasi Clustering Kamera", layout="wide")
st.title("📊 Evaluasi Clustering Kamera dengan K-Means dan Hierarki")

# Sidebar: input jumlah cluster
k_input = st.sidebar.slider("Jumlah Cluster (k)", min_value=2, max_value=10, value=3)

# Evaluasi K-Means untuk berbagai k
inertias = []
silhouettes = []
K_range = range(2, 11)

for k in K_range:
    model = KMeans(n_clusters=k, random_state=42, n_init='auto')
    labels = model.fit_predict(X_scaled)
    inertias.append(model.inertia_)
    silhouettes.append(silhouette_score(X_scaled, labels))

# Elbow Plot
st.subheader("📉 Elbow Method (Inertia)")
fig1, ax1 = plt.subplots()
ax1.plot(K_range, inertias, marker='o')
ax1.set_xlabel("Jumlah Cluster (k)")
ax1.set_ylabel("Inertia (Total SSE)")
ax1.set_title("Elbow Curve untuk Menentukan k Optimal")
st.pyplot(fig1)
st.markdown("- **Inertia** menunjukkan total jarak kuadrat antar titik dengan pusat klusternya. Semakin kecil lebih baik, namun terlalu banyak klaster bisa menyebabkan overfitting.")
st.markdown("- Titik tekuk ('elbow') menunjukkan k yang seimbang antara kompleksitas dan akurasi.")

# Silhouette Plot
st.subheader("📈 Silhouette Score")
fig2, ax2 = plt.subplots()
ax2.plot(K_range, silhouettes, marker='o', color='green')
ax2.set_xlabel("Jumlah Cluster (k)")
ax2.set_ylabel("Silhouette Score")
ax2.set_title("Silhouette Score untuk Menilai Kualitas Klaster")
st.pyplot(fig2)
st.markdown("- **Silhouette Score** menunjukkan seberapa baik objek cocok dengan klasternya sendiri dibandingkan dengan klaster lain.")
st.markdown("- Nilai mendekati **1** menunjukkan klasterisasi yang baik, nilai mendekati **0** menunjukkan tumpang tindih antar klaster.")

# Clustering berdasarkan input K
st.subheader(f"📍 Clustering Hasil Akhir dengan k = {k_input}")
kmeans_final = KMeans(n_clusters=k_input, random_state=42, n_init='auto')
df['cluster'] = kmeans_final.fit_predict(X_scaled)
df['price_million'] = df['price_int'] / 1_000_000_00

fig3, ax3 = plt.subplots()
sns.scatterplot(data=df, x='megapixel', y='price_million', hue='cluster', palette='tab10', ax=ax3)
ax3.set_title(f"Hasil Clustering K-Means (k={k_input})")
ax3.set_xlabel("Megapiksel")
ax3.set_ylabel("Harga (juta)")
st.pyplot(fig3)

# Dendrogram
st.subheader("🌳 Dendrogram (Hierarchical Clustering Sample)")
subset = X_scaled[:50]  # Ambil sebagian data agar dendrogram tidak overload
linked = linkage(subset, method='ward')

fig4, ax4 = plt.subplots(figsize=(10, 6))
dendrogram(linked, ax=ax4)
ax4.set_title("Dendrogram (Ward Linkage, 50 Sampel Pertama)")
st.pyplot(fig4)
st.markdown("- **Dendrogram** digunakan untuk melihat hierarki pengelompokan data.")
st.markdown("- Panjang cabang menunjukkan jarak antar grup. Cocok untuk menentukan jumlah klaster awal sebelum KMeans.")

# Nilai Evaluasi untuk K Terpilih
st.subheader("📌 Ringkasan Evaluasi")
score_k = silhouette_score(X_scaled, kmeans_final.labels_)
st.markdown(f"- **Silhouette Score untuk k = {k_input}**: `{score_k:.3f}`")
st.markdown(f"- **Inertia untuk k = {k_input}**: `{kmeans_final.inertia_:.2f}`")

# Tabel hasil klaster
st.subheader("📂 Data Kamera dan Hasil Clustering")
st.dataframe(df[['camera_name', 'megapixel', 'price_int', 'cluster']])
