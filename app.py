import os
import json
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
from tensorflow.keras.models import load_model
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
# HAPUS r2_score karena tidak dipakai lagi
from sklearn.metrics import mean_absolute_error, mean_squared_error

# ==========================================
# 1. KONFIGURASI HALAMAN & CSS
# ==========================================
st.set_page_config(
    page_title="Sistem Prediksi Perceraian Jabar",
    page_icon="💔",
    layout="wide",
    initial_sidebar_state="expanded"
)

# CSS Custom
st.markdown("""
    <style>
        #MainMenu {visibility: hidden;}
        footer {visibility: hidden;}
        
        /* Styling Sidebar */
        [data-testid="stSidebar"] {
            background-color: #f8f9fa;
        }

        /* Styling Tabs agar lebih terlihat */
        .stTabs [data-baseweb="tab-list"] { gap: 8px; }
        .stTabs [data-baseweb="tab"] {
            height: 50px; 
            white-space: pre-wrap; 
            background-color: #f8f9fa; 
            border-radius: 5px 5px 0 0;
            border: 1px solid #ddd;
            border-bottom: none;
        }
        .stTabs [aria-selected="true"] { 
            background-color: #ffffff; 
            border-top: 3px solid #E63946;
            font-weight: bold;
        }
        
        /* Metric Box Styling */
        div[data-testid="stMetric"] {
            background-color: #ffffff; 
            border: 1px solid #e0e0e0; 
            padding: 15px; 
            border-radius: 8px;
            box-shadow: 0 2px 5px rgba(0,0,0,0.05);
        }

        /* Insight Box Style (Dark Theme Inspired) */
        .insight-box {
            background-color: #1E293B; /* Dark background */
            border-left: 5px solid #3B82F6; /* Blue accent */
            padding: 20px;
            border-radius: 8px;
            margin-top: 20px;
            margin-bottom: 20px;
            color: #F3F4F6; /* Light text */
            font-family: sans-serif;
            box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1);
        }
        .insight-title {
            display: flex;
            align-items: center;
            font-weight: bold;
            font-size: 1.2em;
            margin-bottom: 15px;
            color: #60A5FA; /* Lighter blue title */
        }
        .insight-content {
            font-size: 0.95em;
            line-height: 1.6;
            opacity: 0.95;
        }
        .insight-content strong {
            color: #FCA5A5; /* Reddish bold text */
        }
        
        /* Sidebar Copyright */
        .sidebar-copyright {
            position: fixed; bottom: 0; left: 0; width: 244px;
            padding: 15px; text-align: center; background-color: #ffffff;
            font-size: 12px; color: #666; border-top: 1px solid #e0e0e0;
            z-index: 1000;
        }
    </style>
""", unsafe_allow_html=True)

# ==========================================
# 2. SETUP PATH & KONSTANTA
# ==========================================
BASE_DIR = Path(__file__).parent
DATA_DIR = BASE_DIR / "data"
MODELS_DIR = BASE_DIR / "models"

DATA_FILE = DATA_DIR / "Dataset Jumlah Perceraian Kabupaten Kota Jawa Barat.csv"
GEOJSON_FILE = DATA_DIR / "Kabupaten-Kota (Provinsi Jawa Barat).geojson"

MODEL_MLP_FILE = MODELS_DIR / "model_mlp.h5"
MODEL_RF_FILE = MODELS_DIR / "model_rf.joblib"

TARGET_COL = "Jumlah"
YEAR_COL = "Tahun"
REGION_COL = "Kabupaten/Kota"

# Palet Warna
COLOR_MLP = "#E63946"     # Merah
COLOR_RF = "#457B9D"      # Biru
COLOR_ACTUAL = "#2A9D8F"  # Hijau

# Koordinat Manual (Backup)
JABAR_COORDS = {
    "KABUPATEN BOGOR": [-6.594, 106.789], "KABUPATEN SUKABUMI": [-6.921, 106.927],
    "KABUPATEN CIANJUR": [-6.817, 107.131], "KABUPATEN BANDUNG": [-7.025, 107.519],
    "KABUPATEN GARUT": [-7.202, 107.886], "KABUPATEN TASIKMALAYA": [-7.358, 108.106],
    "KABUPATEN CIAMIS": [-7.327, 108.354], "KABUPATEN KUNINGAN": [-6.976, 108.483],
    "KABUPATEN CIREBON": [-6.737, 108.549], "KABUPATEN MAJALENGKA": [-6.836, 108.227],
    "KABUPATEN SUMEDANG": [-6.858, 107.920], "KABUPATEN INDRAMAYU": [-6.327, 108.322],
    "KABUPATEN SUBANG": [-6.571, 107.760], "KABUPATEN PURWAKARTA": [-6.556, 107.444],
    "KABUPATEN KARAWANG": [-6.322, 107.306], "KABUPATEN BEKASI": [-6.241, 107.123],
    "KABUPATEN BANDUNG BARAT": [-6.843, 107.502], "KABUPATEN PANGANDARAN": [-7.696, 108.654],
    "KOTA BOGOR": [-6.597, 106.799], "KOTA SUKABUMI": [-6.927, 106.929],
    "KOTA BANDUNG": [-6.917, 107.619], "KOTA CIREBON": [-6.732, 108.552],
    "KOTA BEKASI": [-6.238, 106.975], "KOTA DEPOK": [-6.402, 106.794],
    "KOTA CIMAHI": [-6.873, 107.542], "KOTA TASIKMALAYA": [-7.327, 108.220],
    "KOTA BANJAR": [-7.374, 108.532]
}

# ==========================================
# 3. FUNGSI UTAMA (LOAD & CLEAN)
# ==========================================
@st.cache_data
def load_and_clean_data():
    """Memuat data dan membersihkan nama kolom."""
    if not DATA_FILE.exists():
        st.error(f"❌ File data tidak ditemukan di: {DATA_FILE}")
        st.stop()
    
    df = pd.read_csv(DATA_FILE)
    
    # --- CLEANING LABEL ---
    new_cols = []
    for col in df.columns:
        if col in [TARGET_COL, YEAR_COL, REGION_COL]:
            new_cols.append(col)
        else:
            # Hapus variasi kata panjang
            clean = col.replace("Faktor Penyebab Perceraian", "") \
                       .replace("Faktor Perceraian", "") \
                       .replace("Fakor Perceraian", "") \
                       .replace("Faktor Penyebab", "") \
                       .replace("Penyebab Perceraian", "") \
                       .replace("Penyebab", "") \
                       .replace("Faktor", "") \
                       .replace("Fakor", "") \
                       .replace("Nilai", "") \
                       .replace("-", "") \
                       .strip()
            
            # Jika masih ada sisa spasi
            new_cols.append(clean.strip())
            
    # Deduplikasi
    final_cols = []
    seen = {}
    for c in new_cols:
        if c in seen:
            seen[c] += 1
            final_cols.append(f"{c} ({seen[c]})")
        else:
            seen[c] = 0
            final_cols.append(c)
            
    df.columns = final_cols
    # Title case agar cocok dengan GeoJSON
    df[REGION_COL] = df[REGION_COL].str.strip()
    return df

@st.cache_data
def load_geojson():
    """Memuat file GeoJSON."""
    if not GEOJSON_FILE.exists():
        return None
    with open(GEOJSON_FILE, "r", encoding="utf-8") as f:
        geojson = json.load(f)
    return geojson

@st.cache_resource
def load_artifacts(df: pd.DataFrame):
    """Memuat Preprocessor dan Model."""
    try:
        all_cols = df.columns.tolist()
        feature_cols = [c for c in all_cols if c != TARGET_COL]
        categorical_cols = [REGION_COL]
        numeric_cols = [c for c in feature_cols if c not in categorical_cols]

        preprocessor = ColumnTransformer(
            transformers=[
                ("num", StandardScaler(), numeric_cols),
                ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), categorical_cols),
            ]
        )
        preprocessor.fit(df[feature_cols])

        mlp_model = load_model(MODEL_MLP_FILE, compile=False) if MODEL_MLP_FILE.exists() else None
        rf_model = joblib.load(MODEL_RF_FILE) if MODEL_RF_FILE.exists() else None

        factor_cols = [c for c in numeric_cols if c != YEAR_COL]

        return preprocessor, mlp_model, rf_model, feature_cols, factor_cols

    except Exception as e:
        st.error(f"❌ Gagal memuat sistem AI: {e}")
        st.stop()

# --- EKSEKUSI DATA ---
df = load_and_clean_data()
preprocessor, mlp_model, rf_model, feature_cols, factor_cols = load_artifacts(df)

years = sorted(df[YEAR_COL].unique())
regions = sorted(df[REGION_COL].unique())

# ==========================================
# 4. HEADER & SIDEBAR
# ==========================================
st.title("📊 Sistem Analisis & Prediksi Perceraian Jawa Barat")
st.caption("Platform komprehensif untuk memantau tren dan memprediksi angka perceraian menggunakan **Dual-Model AI (MLP & RF)**.")

with st.sidebar:
    st.image("https://cdn-icons-png.flaticon.com/512/2921/2921226.png", width=50)
    st.title("🎛️ Filter Global")
    
    selected_year = st.selectbox(
        "Pilih Tahun Analisis:",
        options=years,
        index=len(years) - 1,
        help="Filter ini akan mengubah data di Tab Eksplorasi dan Peta."
    )
    
    st.markdown("---")
    st.info("💡 **Tips:** Gunakan tombol 'Refresh Cache' jika data atau tampilan terasa tidak update.")
    
    if st.button("🔄 Refresh / Clear Cache"):
        st.cache_data.clear()
        st.rerun()
    
    st.markdown(
        """
        <div class='sidebar-copyright'>
            <b>Copyright © 2025</b><br>
            Developed By:<br>
            <b>Milda Nabilah Al-hamaz</b><br>
            NPM: 202210715079
        </div>
        """,
        unsafe_allow_html=True
    )

# ==========================================
# 5. TABS NAVIGASI UTAMA
# ==========================================
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "📈 Eksplorasi Daerah & Faktor",
    "🗺️ Peta Jawa Barat",
    "🔮 Prediksi (MLP vs RF)",
    "📑 Tabel Data",
    "📉 Evaluasi Model"
])

# ==========================================
# TAB 1: EKSPLORASI DAERAH & FAKTOR
# ==========================================
with tab1:
    st.subheader(f"📈 Eksplorasi Data Tahun {selected_year}")
    df_year = df[df[YEAR_COL] == selected_year].copy()

    # --- 1. Grafik Daerah (Horizontal) ---
    st.markdown("##### 🏙️ Sebaran Kasus per Wilayah")
    df_year_sorted = df_year.sort_values(TARGET_COL, ascending=True)

    fig_region = px.bar(
        df_year_sorted,
        x=TARGET_COL, y=REGION_COL, orientation="h",
        labels={REGION_COL: "Wilayah", TARGET_COL: "Total Kasus"},
        text_auto='.2s', template="plotly_white",
        color=REGION_COL # Warna beda tiap daerah
    )
    fig_region.update_layout(yaxis=dict(categoryorder="total ascending"), height=600, showlegend=False)
    st.plotly_chart(fig_region, use_container_width=True)

    # --- 2. Grafik Faktor (Bar Chart Horizontal) ---
    st.markdown("##### 🧩 Proporsi Faktor Penyebab Utama")
    
    valid_factors = [c for c in factor_cols if c in df_year.columns]
    
    if valid_factors:
        # Hitung total per faktor
        factor_sum = df_year[valid_factors].sum().sort_values(ascending=True)
        factor_df = factor_sum.reset_index()
        factor_df.columns = ["Faktor", "Total Kasus"]

        # Bar Chart Horizontal
        fig_factor = px.bar(
            factor_df,
            x="Total Kasus",
            y="Faktor",
            orientation="h",
            text_auto='.2s', 
            template="plotly_white",
            color="Total Kasus",
            color_continuous_scale="Reds",
            title=f"Total Kasus per Faktor Penyebab ({selected_year})"
        )
        fig_factor.update_layout(yaxis=dict(categoryorder="total ascending"), height=600)
        st.plotly_chart(fig_factor, use_container_width=True)
    else:
        st.warning("Data faktor tidak ditemukan untuk tahun ini.")
    
    # --- INSIGHT ---
    top_region = df_year_sorted.iloc[-1][REGION_COL]
    top_val = df_year_sorted.iloc[-1][TARGET_COL]
    
    st.markdown(f"""
    <div class='insight-box'>
        <div class='insight-title'>💡 Analisis Eksploratif</div>
        <div class='insight-content'>
            <p>Berdasarkan data tahun <b>{selected_year}</b>, wilayah <b>{top_region}</b> mencatatkan angka perceraian tertinggi 
            di Jawa Barat dengan total <b>{top_val:,.0f}</b> kasus. Grafik batang di atas memvisualisasikan disparitas 
            kasus antar wilayah secara jelas.</p>
            <p>Pada grafik kedua (Bar Chart), Anda dapat melihat faktor apa yang paling dominan menyebabkan perceraian di tahun tersebut secara keseluruhan di Jawa Barat.</p>
        </div>
    </div>""", unsafe_allow_html=True)


# ==========================================
# TAB 2: PETA JAWA BARAT
# ==========================================
with tab2:
    st.subheader(f"🗺️ Geografis Sebaran Kasus ({selected_year})")
    df_map = df[df[YEAR_COL] == selected_year].copy()
    geojson = load_geojson()

    # Prioritas 1: Peta Wilayah (Choropleth)
    map_success = False
    if geojson:
        try:
            fig_map = px.choropleth(
                df_map,
                geojson=geojson,
                locations=REGION_COL,
                featureidkey="properties.NAME_2", 
                color=TARGET_COL,
                color_continuous_scale="Reds",
                hover_name=REGION_COL,
                title=f"Peta Intensitas Perceraian ({selected_year})"
            )
            fig_map.update_geos(fitbounds="locations", visible=False)
            fig_map.update_layout(height=600, margin={"r":0,"t":40,"l":0,"b":0})
            st.plotly_chart(fig_map, use_container_width=True)
            map_success = True
        except Exception:
            pass # Fallback silent

    # Prioritas 2: Peta Titik (Scatter Mapbox) jika Choropleth gagal
    if not map_success:
        lats, lons = [], []
        for reg in df_map[REGION_COL]:
            coords = JABAR_COORDS.get(reg.upper(), [-6.9, 107.6])
            lats.append(coords[0])
            lons.append(coords[1])
            
        df_map['lat'] = lats
        df_map['lon'] = lons
        
        fig_map = px.scatter_mapbox(
            df_map, lat="lat", lon="lon", size=TARGET_COL, color=TARGET_COL,
            hover_name=REGION_COL, color_continuous_scale="Reds", size_max=45, zoom=7.2,
            mapbox_style="carto-positron", title=f"Peta Titik Panas ({selected_year})"
        )
        fig_map.update_layout(height=600, margin={"r":0,"t":40,"l":0,"b":0})
        st.plotly_chart(fig_map, use_container_width=True)
        if not geojson:
            st.caption("ℹ️ Menampilkan peta titik koordinat (File GeoJSON tidak ditemukan/cocok).")

    # --- INSIGHT ---
    st.markdown(f"""
    <div class='insight-box'>
        <div class='insight-title'>💡 Interpretasi Spasial</div>
        <div class='insight-content'>
            <p>Peta ini menggambarkan konsentrasi kasus perceraian secara geografis. 
            Wilayah dengan warna <b>merah pekat</b> (atau lingkaran besar) menunjukkan zona dengan tingkat perceraian yang kritis.</p>
            <p>Informasi ini penting bagi pembuat kebijakan untuk memprioritaskan alokasi sumber daya penyuluhan dan intervensi sosial 
            ke wilayah-wilayah yang paling terdampak.</p>
        </div>
    </div>""", unsafe_allow_html=True)


# ==========================================
# TAB 3: PREDIKSI (MULTISELECT)
# ==========================================
with tab3:
    st.subheader("🔮 Simulasi & Prediksi (MLP vs RF)")
    st.info("Pilih wilayah, tahun masa depan, dan faktor penyebab. Faktor yang dipilih akan otomatis diisi nilai **Median**, sisanya **0**.")

    with st.form("prediction_form"):
        c1, c2 = st.columns(2)
        with c1:
            st.markdown("**1. Parameter Wilayah**")
            regions_input = st.multiselect("Pilih Kabupaten/Kota:", options=regions, default=[regions[0]])
            years_input = st.multiselect("Pilih Tahun Prediksi:", options=list(range(2020, 2031)), default=[2025])
        
        with c2:
            st.markdown("**2. Parameter Faktor Penyebab**")
            st.caption("Pilih faktor yang diasumsikan terjadi:")
            # Faktor bersih (tanpa kata Nilai/Faktor di label)
            factor_inputs = st.multiselect("Faktor:", options=factor_cols)
            
            # Input Dinamis hanya untuk faktor terpilih
            inputs = {}
            if factor_inputs:
                for f in factor_inputs:
                    inputs[f] = st.number_input(f"{f}:", min_value=0, value=0)

        submit_pred = st.form_submit_button("🚀 Jalankan Prediksi")

    if submit_pred:
        if not regions_input or not years_input:
            st.warning("Mohon lengkapi pilihan wilayah dan tahun.")
        elif not mlp_model or not rf_model:
            st.error("Model AI belum siap.")
        else:
            rows = []
            for r in regions_input:
                for y in years_input:
                    row = {REGION_COL: r, YEAR_COL: y}
                    # Isi faktor
                    for f in factor_cols:
                        if f in inputs:
                            row[f] = inputs[f]
                        else:
                            row[f] = 0.0
                    rows.append(row)

            # DataFrame Input
            input_df = pd.DataFrame(rows)
            input_df_final = input_df[feature_cols]

            try:
                # Preprocessing
                X_pred = preprocessor.transform(input_df_final)
                
                # Prediksi
                y_mlp = mlp_model.predict(X_pred).flatten()
                y_rf = rf_model.predict(X_pred)
                
                # Format Output
                res_display = input_df[[REGION_COL, YEAR_COL]].copy()
                res_display["Prediksi MLP"] = [f"{val:,.0f}" for val in y_mlp]
                res_display["Prediksi RF"] = [f"{val:,.0f}" for val in y_rf]
                res_display["Selisih"] = [f"{abs(m - r):,.0f}" for m, r in zip(y_mlp, y_rf)]
                
                st.success("✅ Prediksi Selesai!")
                st.dataframe(res_display, use_container_width=True)
                
                # Visualisasi
                res_plot = input_df[[REGION_COL, YEAR_COL]].copy()
                res_plot["Prediksi MLP"] = y_mlp
                res_plot["Prediksi RF"] = y_rf
                
                melted_res = res_plot.melt(id_vars=[REGION_COL, YEAR_COL], 
                                           value_vars=["Prediksi MLP", "Prediksi RF"],
                                           var_name="Model", value_name="Nilai")
                
                fig_comp = px.bar(melted_res, x="Nilai", y=REGION_COL, color="Model", barmode="group",
                                  title="Perbandingan Hasil Prediksi Model", orientation='h',
                                  color_discrete_map={"Prediksi MLP": COLOR_MLP, "Prediksi RF": COLOR_RF})
                st.plotly_chart(fig_comp, use_container_width=True)
                
                # --- INSIGHT ---
                diff_avg = np.mean(np.abs(y_mlp - y_rf))
                higher_model = "MLP" if np.mean(y_mlp) > np.mean(y_rf) else "Random Forest"
                
                st.markdown(f"""
                <div class='insight-box'>
                    <div class='insight-title'>💡 Analisis Hasil Prediksi</div>
                    <div class='insight-content'>
                        <p>Simulasi menunjukkan bahwa model <b>{higher_model}</b> cenderung memberikan estimasi angka yang lebih tinggi 
                        dibandingkan model lainnya.</p>
                        <p>Rata-rata selisih prediksi antara kedua algoritma adalah <b>{diff_avg:,.0f}</b> kasus. 
                        Perbedaan ini wajar terjadi karena karakteristik matematis yang berbeda:</p>
                        <ul>
                            <li><b>MLP (Neural Network):</b> Belajar pola global yang kompleks dan non-linear.</li>
                            <li><b>Random Forest:</b> Menggunakan <i>ensemble trees</i> yang lebih stabil terhadap data tabular.</li>
                        </ul>
                        <p><b>Rekomendasi Strategis:</b> Untuk keperluan perencanaan anggaran atau sumber daya penanganan, disarankan menggunakan 
                        angka prediksi tertinggi sebagai langkah antisipatif (konservatif).</p>
                    </div>
                </div>""", unsafe_allow_html=True)

            except Exception as e:
                st.error(f"Error prediksi: {e}")


# ==========================================
# TAB 4: TABEL DATA
# ==========================================
with tab4:
    st.subheader("📑 Eksplorasi Data Tabular")
    c1, c2 = st.columns(2)
    reg_f = c1.selectbox("Filter Wilayah:", ["(Semua)"] + regions)
    yr_f = c2.selectbox("Filter Tahun:", ["(Semua)"] + [str(y) for y in years])
    
    df_tbl = df.copy()
    if reg_f != "(Semua)": df_tbl = df_tbl[df_tbl[REGION_COL] == reg_f]
    if yr_f != "(Semua)": df_tbl = df_tbl[df_tbl[YEAR_COL] == int(yr_f)]
    
    st.dataframe(df_tbl, use_container_width=True)


# ==========================================
# TAB 5: EVALUASI MODEL (NO R2)
# ==========================================
with tab5:
    st.subheader("📉 Evaluasi Performa Model")
    test_yr = years[-1]
    
    df_test = df[df[YEAR_COL] == test_yr].copy()
    if df_test.empty:
        st.warning("Data uji kosong.")
    else:
        # Hitung Metrik
        X_t = preprocessor.transform(df_test[feature_cols])
        y_true = df_test[TARGET_COL].values
        
        p_mlp = mlp_model.predict(X_t).flatten()
        p_rf = rf_model.predict(X_t)
        
        mae_mlp = mean_absolute_error(y_true, p_mlp)
        rmse_mlp = np.sqrt(mean_squared_error(y_true, p_mlp))
        
        mae_rf = mean_absolute_error(y_true, p_rf)
        rmse_rf = np.sqrt(mean_squared_error(y_true, p_rf))

        # Kartu Metrik (TANPA R2)
        best_model_name = "MLP (Neural Network)" if mae_mlp < mae_rf else "Random Forest"
        best_mae_val = min(mae_mlp, mae_rf)

        c1, c2 = st.columns(2)
        c1.metric("🏆 Model Terbaik", best_model_name)
        c2.metric("Rata-rata Error (MAE)", f"{best_mae_val:.0f} Kasus")

        st.markdown("---")

        # Tabel Metrik (TANPA R2)
        met_df = pd.DataFrame({
            "Model": ["MLP (Neural Network)", "Random Forest"],
            "MAE (Rata-rata Error)": [mae_mlp, mae_rf],
            "RMSE (Error Kuadrat)": [rmse_mlp, rmse_rf]
        })
        st.table(met_df.set_index("Model").style.format("{:.2f}"))

        # Scatter Plot
        fig_sc = go.Figure()
        fig_sc.add_trace(go.Scatter(x=y_true, y=p_mlp, mode='markers', name='MLP', marker=dict(color=COLOR_MLP, opacity=0.7)))
        fig_sc.add_trace(go.Scatter(x=y_true, y=p_rf, mode='markers', name='RF', marker=dict(color=COLOR_RF, symbol='x')))
        fig_sc.add_trace(go.Scatter(x=[y_true.min(), y_true.max()], y=[y_true.min(), y_true.max()], 
                                    mode='lines', name='Garis Ideal', line=dict(color='gray', dash='dash')))
        fig_sc.update_layout(title="Akurasi: Aktual vs Prediksi", xaxis_title="Jumlah Aktual", yaxis_title="Jumlah Prediksi")
        st.plotly_chart(fig_sc, use_container_width=True)
        
        # --- INSIGHT KAYA ---
        improvement = abs(mae_mlp - mae_rf)
        
        insight_html = f"""
<div class='insight-box'>
    <div class='insight-title'>💡 Kesimpulan Evaluasi Menyeluruh</div>
    <div class='insight-content'>
        <p>Berdasarkan pengujian data tahun terakhir (<b>{test_yr}</b>), model <b>{best_model_name}</b> menunjukkan performa yang lebih unggul.</p>
        <p><b>Temuan Penting:</b></p>
        <ul>
            <li><b>Akurasi:</b> Model {best_model_name} memiliki error <b>{improvement:.2f}</b> poin lebih kecil.</li>
            <li><b>Konsistensi:</b> Sebaran prediksi model ini lebih mendekati garis ideal pada grafik di atas.</li>
            <li><b>Rekomendasi:</b> Gunakan model ini untuk prediksi kebijakan jangka pendek.</li>
        </ul>
    </div>
</div>
"""
        st.markdown(insight_html, unsafe_allow_html=True)
