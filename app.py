import streamlit as st
import numpy as np
import pickle
import neurokit2 as nk
import antropy
import matplotlib.pyplot as plt

st.set_page_config(
    page_title="Uyku Apnesi Dedektörü",
    page_icon="🫀",
    layout="centered"
)

@st.cache_resource
def modelleri_yukle():
    with open("best_xgb_optimized.pkl", "rb") as f: model = pickle.load(f)
    with open("scaler2.pkl",            "rb") as f: scaler = pickle.load(f)
    with open("selector2.pkl",          "rb") as f: selector = pickle.load(f)
    with open("imputer.pkl",            "rb") as f: imputer = pickle.load(f)
    return model, scaler, selector, imputer

model, scaler, selector, imputer = modelleri_yukle()

def ozellik_cikar(rr):
    rr = np.array(rr, dtype=float)
    if len(rr) < 3:
        return None
    f = {}
    f['mean_rr']     = np.mean(rr)
    f['std_rr']      = np.std(rr)
    f['rmssd']       = np.sqrt(np.mean(np.diff(rr)**2))
    f['pnn50']       = np.sum(np.abs(np.diff(rr)) > 50) / len(rr) * 100
    f['min_rr']      = np.min(rr)
    f['max_rr']      = np.max(rr)
    f['range_rr']    = np.max(rr) - np.min(rr)
    f['lf_hf_ratio'] = f['std_rr'] / (f['rmssd'] + 1e-6)
    sd1 = np.std(np.diff(rr)) / np.sqrt(2)
    sd2 = np.std(rr)
    f['sd1']         = sd1
    f['sd2']         = sd2
    f['sd1_sd2']     = sd1 / (sd2 + 1e-6)
    try:
        f['sample_entropy'] = antropy.sample_entropy(rr)
        f['dfa_alpha']      = antropy.detrended_fluctuation(rr)
    except:
        f['sample_entropy'] = 0
        f['dfa_alpha']      = 0
    return list(f.values())

def tahmin_et(ecg, fs=100):
    try:
        signals, info = nk.ecg_process(ecg, sampling_rate=fs)
        r_tepeleri = info['ECG_R_Peaks']
    except:
        return None, None, None
    if len(r_tepeleri) < 5:
        return None, None, None
    rr = np.diff(r_tepeleri) / fs * 1000
    ozellikler = ozellik_cikar(rr)
    if ozellikler is None:
        return None, None, None
    X = np.array([ozellikler])
    X = imputer.transform(X)
    X = scaler.transform(X)
    X = selector.transform(X)
    olasilik = model.predict_proba(X)[0][1]
    tahmin   = int(olasilik >= 0.5)
    return tahmin, olasilik, r_tepeleri

# ── ARAYÜZ ─────────────────────────────────────────────────────────────────────

st.markdown("""
<div style='text-align:center; padding: 20px 0 10px 0'>
    <h1 style='color:#1E2761; font-size:2.2rem; margin-bottom:4px'>🫀 Uyku Apnesi Dedektörü</h1>
    <p style='color:#64748B; font-size:1rem'>ECG sinyalinden uyku apnesi tespiti — XGBoost · AUC 0.85</p>
</div>
<hr style='border:1px solid #E2E8F0; margin-bottom:24px'>
""", unsafe_allow_html=True)

st.markdown("### 📁 ECG Segmenti Yükle")
st.caption("1 dakikalık ECG segmenti — .npy formatında, 6000 örnek (100 Hz × 60 sn)")
yuklenen = st.file_uploader("Dosya seç", type=["npy"], label_visibility="collapsed")

st.markdown("### 🎯 veya Demo Segment Kullan")
demo_sec = st.radio(
    "Segment türü",
    ["😴 Apneli Segment", "😊 Normal Segment"],
    horizontal=True,
    label_visibility="collapsed"
)

st.markdown("<br>", unsafe_allow_html=True)
analiz_btn = st.button("🔍 Analiz Et", use_container_width=True, type="primary")

if analiz_btn:
    fs = 100

    if yuklenen is not None:
        ecg = np.load(yuklenen)
        if len(ecg) > 6000:
            ecg = ecg[:6000]
        st.info("✅ Yüklenen dosya kullanılıyor.")
    else:
        st.info("ℹ️ Demo segment oluşturuluyor...")
        np.random.seed(42)
        t = np.linspace(0, 60, 6000)
        if "Apneli" in demo_sec:
            # Apneli: yüksek frekanslı, düzensiz
            ecg = (1.5 * np.sin(2 * np.pi * 1.1 * t) +
                   0.3 * np.sin(2 * np.pi * 0.15 * t) +
                   0.25 * np.random.randn(6000))
            # Birkaç apne periyodu ekle (sinyal düşer)
            for start in [1000, 2500, 4000]:
                ecg[start:start+400] *= 0.3
        else:
            # Normal: düzenli sinüs ritmi
            ecg = (1.5 * np.sin(2 * np.pi * 1.2 * t) +
                   0.1 * np.random.randn(6000))

    with st.spinner("⏳ Analiz ediliyor, lütfen bekleyin..."):
        tahmin, olasilik, r_tepeleri = tahmin_et(ecg, fs)

    st.markdown("---")

    if tahmin is None:
        st.error("❌ R tepesi tespit edilemedi. Farklı bir segment deneyin.")
    else:
        # Sonuç kartı
        if tahmin == 1:
            st.markdown(f"""
            <div style='background:#FEE2E2; border:2px solid #F87171; border-radius:16px;
                        padding:24px; text-align:center; margin:16px 0'>
                <div style='font-size:3rem'>⚠️</div>
                <h2 style='color:#DC2626; margin:8px 0'>APNE TESPİT EDİLDİ</h2>
                <p style='color:#991B1B; font-size:1.1rem; margin:0'>
                    Apne olasılığı: <strong>{olasilik*100:.1f}%</strong>
                </p>
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown(f"""
            <div style='background:#D1FAE5; border:2px solid #34D399; border-radius:16px;
                        padding:24px; text-align:center; margin:16px 0'>
                <div style='font-size:3rem'>✅</div>
                <h2 style='color:#065F46; margin:8px 0'>NORMAL UYKU</h2>
                <p style='color:#047857; font-size:1.1rem; margin:0'>
                    Normal olasılığı: <strong>{(1-olasilik)*100:.1f}%</strong>
                </p>
            </div>
            """, unsafe_allow_html=True)

        # ECG grafiği
        st.markdown("#### 📈 ECG Sinyali ve R Tepeleri")
        fig, ax = plt.subplots(figsize=(10, 3))
        ax.plot(ecg, color="#1E2761", linewidth=0.7, alpha=0.85, label="ECG")
        ax.scatter(r_tepeleri, ecg[r_tepeleri], color="#DC2626",
                   s=25, zorder=5, label=f"R tepeleri ({len(r_tepeleri)} adet)")
        ax.set_xlabel("Örnek (100 Hz)")
        ax.set_ylabel("Amplitüd (mV)")
        ax.legend(fontsize=9)
        ax.spines[['top', 'right']].set_visible(False)
        ax.set_facecolor("#F8FAFC")
        fig.patch.set_facecolor("#F8FAFC")
        st.pyplot(fig)
        plt.close()

        # HRV metrikleri
        rr = np.diff(r_tepeleri) / fs * 1000
        st.markdown("#### 📊 HRV Özellikleri")
        k1, k2, k3, k4 = st.columns(4)
        k1.metric("Ortalama RR", f"{np.mean(rr):.0f} ms")
        k2.metric("RMSSD",       f"{np.sqrt(np.mean(np.diff(rr)**2)):.1f} ms")
        k3.metric("pNN50",       f"{np.sum(np.abs(np.diff(rr)) > 50) / len(rr) * 100:.1f}%")
        k4.metric("R Tepesi",    f"{len(r_tepeleri)} adet")

        # Model bilgisi
        with st.expander("ℹ️ Model hakkında"):
            st.markdown("""
            | Özellik | Değer |
            |---------|-------|
            | Model | XGBoost (optimize edilmiş) |
            | AUC (test) | 0.8479 |
            | Apne recall | %80 |
            | Veri seti | PhysioNet Apnea-ECG (70 kayıt) |
            | Özellik sayısı | 10 HRV özelliği |
            """)

st.markdown("""
<hr style='margin-top:40px'>
<p style='text-align:center; color:#94A3B8; font-size:0.8rem'>
    Pattern Recognition Dersi Projesi · PhysioNet Apnea-ECG Veri Seti
</p>
""", unsafe_allow_html=True)