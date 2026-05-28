import streamlit as st
import numpy as np
import pickle
import neurokit2 as nk
import antropy
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import io

st.set_page_config(
    page_title="Uyku Apnesi Dedektörü",
    page_icon="🫀",
    layout="centered"
)

@st.cache_resource
def modelleri_yukle():
    with open("full_pipeline_v4.pkl", "rb") as f:
        p = pickle.load(f)
    return p['model'], p['scaler'], p['selector'], p['col_means']

model, scaler, selector, col_means = modelleri_yukle()

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

def tahmin_yap(ozellikler):
    X = np.array([ozellikler])
    X = np.where(np.isinf(X), np.nan, X)
    for i in range(X.shape[1]):
        mask = np.isnan(X[:, i])
        X[mask, i] = col_means[i]
    X = scaler.transform(X)
    X = selector.transform(X)
    olasilik = model.predict_proba(X)[0][1]
    tahmin   = int(olasilik >= 0.55)
    return tahmin, olasilik

def gece_analiz(ecg_uzun, fs=100, ilerleme_cubugu=None):
    samples_per_min = 60 * fs
    n_dakika = len(ecg_uzun) // samples_per_min
    sonuclar = []
    for i in range(n_dakika):
        seg = ecg_uzun[i*samples_per_min:(i+1)*samples_per_min]
        try:
            signals, info = nk.ecg_process(seg, sampling_rate=fs)
            r_tepeleri = info['ECG_R_Peaks']
            if len(r_tepeleri) < 5:
                sonuclar.append({'dakika': i+1, 'tahmin': None, 'olasilik': None})
                continue
            rr = np.diff(r_tepeleri) / fs * 1000
            ozellikler = ozellik_cikar(rr)
            if ozellikler is None:
                sonuclar.append({'dakika': i+1, 'tahmin': None, 'olasilik': None})
                continue
            tahmin, olasilik = tahmin_yap(ozellikler)
            sonuclar.append({'dakika': i+1, 'tahmin': tahmin, 'olasilik': olasilik})
        except:
            sonuclar.append({'dakika': i+1, 'tahmin': None, 'olasilik': None})
        if ilerleme_cubugu:
            ilerleme_cubugu.progress((i+1)/n_dakika, text=f"Dakika {i+1}/{n_dakika} analiz ediliyor...")
    return sonuclar

# ── ARAYÜZ ─────────────────────────────────────────────────────────────────────

st.markdown("""
<div style='text-align:center; padding:24px 0 8px 0'>
    <h1 style='color:#1E2761; font-size:2.4rem; margin-bottom:6px'>🫀 Uyku Apnesi Dedektörü</h1>
    <p style='color:#64748B; font-size:1.05rem; margin:0'>
        Tek kanallı ECG sinyalinden uyku apnesi tespiti
    </p>
    <p style='color:#94A3B8; font-size:0.9rem; margin-top:4px'>
        XGBoost · HRV Özellikleri · AUC 0.85 · PhysioNet Apnea-ECG
    </p>
</div>
<hr style='border:1px solid #E2E8F0; margin:16px 0 24px 0'>
""", unsafe_allow_html=True)

# Nasıl çalışır
with st.expander("ℹ️ Nasıl çalışır?"):
    st.markdown("""
    1. **ECG yükle** — Tüm gece kaydedilmiş single-lead ECG (.npy formatında)
    2. **Dakika dakika analiz** — Her 1 dakikalık segment için HRV özellikleri hesaplanır
    3. **Sınıflandırma** — XGBoost modeli apne olup olmadığına karar verir
    4. **Tanı** — Apneli dakika oranına göre normal/hafif/orta-ağır sınıflandırması yapılır
    
    **Özellikler:** mean_rr, std_rr, rmssd, pnn50, min_rr, range_rr, lf_hf_ratio, sd2, sd1_sd2, sample_entropy, dfa_alpha
    """)

st.markdown("### 📁 ECG Kaydı Yükle")
st.caption("Tüm gece ECG kaydı — .npy formatında (minimum 60 dakika önerilir)")
yuklenen = st.file_uploader("Dosya seç", type=["npy"], label_visibility="collapsed")

st.markdown("<br>", unsafe_allow_html=True)
analiz_btn = st.button("🔬 Analiz Başlat", use_container_width=True, type="primary")

if analiz_btn:
    fs = 100

    if yuklenen is not None:
        ecg_uzun = np.load(io.BytesIO(yuklenen.read())).flatten()
        n_dk = len(ecg_uzun) // 6000
        st.info(f"✅ {n_dk} dakikalık ECG yüklendi.")
    else:
        st.warning("⚠️ Dosya yüklenmedi. Lütfen bir ECG dosyası yükleyin.")
        st.stop()

    # İlerleme çubuğu
    progress = st.progress(0, text="Analiz başlıyor...")

    sonuclar = gece_analiz(ecg_uzun, fs, progress)
    progress.empty()

    gecerli = [s for s in sonuclar if s['tahmin'] is not None]

    if not gecerli:
        st.error("❌ Hiçbir segmentte R tepesi tespit edilemedi.")
        st.stop()

    apne_sayisi  = sum(1 for s in gecerli if s['tahmin'] == 1)
    normal_sayisi = sum(1 for s in gecerli if s['tahmin'] == 0)
    apne_orani   = apne_sayisi / len(gecerli) * 100

    st.markdown("---")

    # Sonuç kartı
    if apne_orani >= 50:
        st.markdown(f"""
        <div style='background:#FEE2E2;border:2px solid #DC2626;border-radius:16px;
                    padding:28px;text-align:center;margin:16px 0'>
            <div style='font-size:3.5rem'>🚨</div>
            <h2 style='color:#DC2626;margin:8px 0;font-size:1.8rem'>ORTA / AĞIR UYKU APNESİ</h2>
            <p style='color:#991B1B;font-size:1.1rem;margin:4px 0'>
                {apne_sayisi} / {len(gecerli)} dakikada apne tespit edildi
                &nbsp;·&nbsp; <strong>{apne_orani:.1f}%</strong>
            </p>
            <p style='color:#B91C1C;font-size:0.9rem;margin-top:8px'>
                Bir uyku uzmanına başvurmanız önerilir.
            </p>
        </div>""", unsafe_allow_html=True)
    elif apne_orani >= 33:
        st.markdown(f"""
        <div style='background:#FEF3C7;border:2px solid #F59E0B;border-radius:16px;
                    padding:28px;text-align:center;margin:16px 0'>
            <div style='font-size:3.5rem'>⚠️</div>
            <h2 style='color:#D97706;margin:8px 0;font-size:1.8rem'>HAFİF UYKU APNESİ</h2>
            <p style='color:#92400E;font-size:1.1rem;margin:4px 0'>
                {apne_sayisi} / {len(gecerli)} dakikada apne tespit edildi
                &nbsp;·&nbsp; <strong>{apne_orani:.1f}%</strong>
            </p>
            <p style='color:#B45309;font-size:0.9rem;margin-top:8px'>
                Takip önerilir.
            </p>
        </div>""", unsafe_allow_html=True)
    else:
        st.markdown(f"""
        <div style='background:#D1FAE5;border:2px solid #34D399;border-radius:16px;
                    padding:28px;text-align:center;margin:16px 0'>
            <div style='font-size:3.5rem'>✅</div>
            <h2 style='color:#065F46;margin:8px 0;font-size:1.8rem'>NORMAL UYKU</h2>
            <p style='color:#047857;font-size:1.1rem;margin:4px 0'>
                {apne_sayisi} / {len(gecerli)} dakikada apne tespit edildi
                &nbsp;·&nbsp; <strong>{apne_orani:.1f}%</strong>
            </p>
        </div>""", unsafe_allow_html=True)

    # Özet metrikler
    st.markdown("#### 📊 Özet")
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Toplam Dakika",  f"{len(gecerli)}")
    m2.metric("Apneli Dakika",  f"{apne_sayisi}")
    m3.metric("Normal Dakika",  f"{normal_sayisi}")
    m4.metric("Apne Oranı",     f"{apne_orani:.1f}%")

    # Dakika dakika harita
    st.markdown("#### 🗺️ Gece Boyu Apne Haritası")
    fig_w = max(10, len(gecerli) * 0.15)
    fig, ax = plt.subplots(figsize=(min(fig_w, 20), 3))
    for s in gecerli:
        renk  = "#DC2626" if s['tahmin'] == 1 else "#10B981"
        alpha = s['olasilik'] if s['tahmin'] == 1 else (1 - s['olasilik'])
        ax.bar(s['dakika'], 1, color=renk, alpha=max(0.35, alpha), width=0.9)
    apne_patch   = mpatches.Patch(color='#DC2626', label='Apne')
    normal_patch = mpatches.Patch(color='#10B981', label='Normal')
    ax.legend(handles=[apne_patch, normal_patch], loc='upper right', fontsize=9)
    ax.set_xlabel("Dakika")
    ax.set_ylabel("")
    ax.set_yticks([])
    ax.set_xlim(0, len(gecerli) + 1)
    ax.spines[['top','right','left']].set_visible(False)
    ax.set_facecolor("#F8FAFC")
    fig.patch.set_facecolor("#F8FAFC")
    st.pyplot(fig)
    plt.close()

    # Detay tablosu
    with st.expander("📋 Dakika Dakika Detay"):
        cols = st.columns(4)
        for idx, s in enumerate(gecerli):
            durum = "🔴 Apne" if s['tahmin'] == 1 else "🟢 Normal"
            cols[idx % 4].write(f"**Dk {s['dakika']}:** {durum} ({s['olasilik']*100:.0f}%)")

# Model bilgisi
st.markdown("---")
col1, col2, col3 = st.columns(3)
col1.metric("Model AUC", "0.8479")
col2.metric("Apne Recall", "%80")
col3.metric("Veri Seti", "70 kayıt")

st.markdown("""
<p style='text-align:center; color:#94A3B8; font-size:0.8rem; margin-top:16px'>
    ⚠️ Bu sistem yalnızca araştırma amaçlıdır, tıbbi tanı koymaz.
    <br>Pattern Recognition Dersi Projesi · PhysioNet Apnea-ECG Veri Seti
</p>
""", unsafe_allow_html=True)
