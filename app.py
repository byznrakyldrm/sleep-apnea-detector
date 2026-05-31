import streamlit as st
import numpy as np
import pickle
import neurokit2 as nk
import antropy
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import io

st.set_page_config(
    page_title="ApneaWatch — Sleep Apnea Detector",
    page_icon="🫀",
    layout="wide"
)

st.markdown("""
<style>
[data-testid="stAppViewContainer"] { background: #0F1117; }
[data-testid="stSidebar"] { background: #161B27; border-right: 1px solid #1E2A3A; }

.aw-header {
    display: flex; align-items: center; gap: 16px;
    padding: 1.5rem 0 1rem;
    border-bottom: 1px solid #1E2A3A;
    margin-bottom: 1.5rem;
}
.aw-logo {
    width: 44px; height: 44px; border-radius: 12px;
    background: linear-gradient(135deg, #1D4ED8, #7C3AED);
    display: flex; align-items: center; justify-content: center;
    font-size: 22px;
}
.aw-title { font-size: 1.4rem; font-weight: 700; color: #F1F5F9; margin: 0; }
.aw-sub   { font-size: 0.8rem; color: #64748B; margin: 2px 0 0; }

.badge {
    display: inline-block; padding: 3px 12px; border-radius: 999px;
    font-size: 11px; font-weight: 600; letter-spacing: .03em;
}
.badge-demo { background: #1E293B; color: #94A3B8; border: 1px solid #334155; }

.result-card {
    border-radius: 16px; padding: 2rem; text-align: center;
    margin: 1rem 0;
}
.result-card.danger  { background: #1c0a0a; border: 1.5px solid #7f1d1d; }
.result-card.warning { background: #1c1300; border: 1.5px solid #78350f; }
.result-card.success { background: #052e16; border: 1.5px solid #14532d; }

.result-card .rc-icon  { font-size: 3rem; margin-bottom: .5rem; }
.result-card .rc-title { font-size: 1.5rem; font-weight: 700; margin: .25rem 0; }
.result-card .rc-sub   { font-size: .9rem; margin: .25rem 0; opacity: .8; }

.result-card.danger  .rc-title { color: #fca5a5; }
.result-card.danger  .rc-sub   { color: #f87171; }
.result-card.warning .rc-title { color: #fcd34d; }
.result-card.warning .rc-sub   { color: #fbbf24; }
.result-card.success .rc-title { color: #86efac; }
.result-card.success .rc-sub   { color: #4ade80; }

.metric-box {
    background: #161B27; border: 1px solid #1E2A3A;
    border-radius: 12px; padding: 1rem 1.25rem;
    text-align: center;
}
.metric-box .mb-val { font-size: 1.8rem; font-weight: 700; color: #F1F5F9; }
.metric-box .mb-lbl { font-size: .75rem; color: #64748B; margin-top: 4px; text-transform: uppercase; letter-spacing: .05em; }

.upload-zone {
    border: 1.5px dashed #1E3A5F; border-radius: 16px;
    padding: 2.5rem; text-align: center; background: #0D1520;
}
.upload-zone h3 { color: #93C5FD; margin: .5rem 0 .25rem; font-size: 1rem; }
.upload-zone p  { color: #475569; font-size: .85rem; margin: 0; }

.night-map  { display: flex; flex-wrap: wrap; gap: 3px; margin: .75rem 0; }
.nm-min     { width: 12px; height: 28px; border-radius: 3px; }
</style>
""", unsafe_allow_html=True)

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

def gece_analiz_canli(ecg_uzun, fs=100):
    samples_per_min = 60 * fs
    n_dakika = len(ecg_uzun) // samples_per_min
    sonuclar = []

    progress    = st.progress(0, text="Analysis starting...")
    grafik_alani = st.empty()
    durum_alani  = st.empty()

    for i in range(n_dakika):
        seg = ecg_uzun[i*samples_per_min:(i+1)*samples_per_min]
        try:
            signals, info = nk.ecg_process(seg, sampling_rate=fs)
            r_tepeleri = info['ECG_R_Peaks']
            if len(r_tepeleri) < 5:
                sonuclar.append({'dakika': i+1, 'tahmin': None, 'olasilik': None, 'rr': None})
            else:
                rr = np.diff(r_tepeleri) / fs * 1000
                ozellikler = ozellik_cikar(rr)
                if ozellikler is None:
                    sonuclar.append({'dakika': i+1, 'tahmin': None, 'olasilik': None, 'rr': None})
                else:
                    tahmin, olasilik = tahmin_yap(ozellikler)
                    sonuclar.append({'dakika': i+1, 'tahmin': tahmin, 'olasilik': olasilik, 'rr': rr})
        except:
            sonuclar.append({'dakika': i+1, 'tahmin': None, 'olasilik': None, 'rr': None})

        # Her dakika grafiği güncelle
        gecerli_su_ana = [s for s in sonuclar if s['tahmin'] is not None]
        if gecerli_su_ana:
            fig, ax = plt.subplots(figsize=(14, 3))
            fig.patch.set_facecolor('#0F1117')
            ax.set_facecolor('#0D1520')
            dklar  = [s['dakika']   for s in gecerli_su_ana]
            olasil = [s['olasilik'] for s in gecerli_su_ana]
            renkler = ['#ef4444' if s['tahmin'] == 1 else '#22c55e' for s in gecerli_su_ana]
            ax.bar(dklar, olasil, color=renkler, width=0.85, alpha=0.85)
            ax.axhline(0.55, color='#F59E0B', linewidth=1, linestyle='--', label='Threshold (0.55)')
            ax.set_xlim(0, n_dakika + 1)
            ax.set_ylim(0, 1)
            ax.set_xlabel("Minute", color='#64748B', fontsize=9)
            ax.set_ylabel("Apnea Probability", color='#64748B', fontsize=9)
            ax.tick_params(colors='#64748B', labelsize=8)
            for spine in ax.spines.values():
                spine.set_edgecolor('#1E2A3A')
            ax.legend(fontsize=8, facecolor='#161B27', labelcolor='#94A3B8', edgecolor='#1E2A3A')
            grafik_alani.pyplot(fig)
            plt.close()

            # Son dakikanın durumunu göster
            son = gecerli_su_ana[-1]
            durum_renk = "#ef4444" if son['tahmin'] == 1 else "#22c55e"
            durum_metin = "APNEA" if son['tahmin'] == 1 else "NORMAL"
            durum_alani.markdown(
                "<div style='background:#161B27;border:1px solid #1E2A3A;border-radius:10px;"
                "padding:.6rem 1rem;display:flex;justify-content:space-between;align-items:center'>"
                "<span style='color:#64748B;font-size:.85rem'>Minute " + str(son['dakika']) + "</span>"
                "<span style='color:" + durum_renk + ";font-weight:700;font-size:.95rem'>" + durum_metin + " — " + f"{son['olasilik']*100:.0f}%" + "</span>"
                "</div>",
                unsafe_allow_html=True
            )

        progress.progress((i+1)/n_dakika, text="Analyzing... " + str(i+1) + "/" + str(n_dakika) + " minutes")

    progress.empty()
    durum_alani.empty()
    return sonuclar

# ── SIDEBAR ────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("""
    <div style='padding:.5rem 0 1.5rem'>
        <div style='font-size:1.1rem;font-weight:700;color:#F1F5F9;margin-bottom:4px'>🫀 ApneaWatch</div>
        <span class='badge badge-demo'>Demo Mode</span>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("<div style='font-size:.7rem;color:#475569;text-transform:uppercase;letter-spacing:.08em;margin-bottom:.5rem'>Device Status</div>", unsafe_allow_html=True)
    st.markdown("""
    <div style='background:#0D1520;border:1px solid #1E2A3A;border-radius:10px;padding:.75rem 1rem;margin-bottom:1rem'>
        <div style='display:flex;justify-content:space-between;margin-bottom:.3rem'>
            <span style='color:#94A3B8;font-size:.85rem'>Connection</span>
            <span style='color:#4ade80;font-size:.85rem;font-weight:600'>● Simulated</span>
        </div>
        <div style='display:flex;justify-content:space-between;margin-bottom:.3rem'>
            <span style='color:#94A3B8;font-size:.85rem'>Sampling Rate</span>
            <span style='color:#CBD5E1;font-size:.85rem'>100 Hz</span>
        </div>
        <div style='display:flex;justify-content:space-between;margin-bottom:.3rem'>
            <span style='color:#94A3B8;font-size:.85rem'>Channel</span>
            <span style='color:#CBD5E1;font-size:.85rem'>Single-lead ECG</span>
        </div>
        <div style='display:flex;justify-content:space-between'>
            <span style='color:#94A3B8;font-size:.85rem'>Format</span>
            <span style='color:#CBD5E1;font-size:.85rem'>.npy</span>
        </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("<div style='font-size:.7rem;color:#475569;text-transform:uppercase;letter-spacing:.08em;margin-bottom:.5rem'>Model Information</div>", unsafe_allow_html=True)
    st.markdown("""
    <div style='background:#0D1520;border:1px solid #1E2A3A;border-radius:10px;padding:.75rem 1rem;margin-bottom:1rem'>
        <div style='display:flex;justify-content:space-between;margin-bottom:.3rem'>
            <span style='color:#94A3B8;font-size:.85rem'>Algorithm</span>
            <span style='color:#93C5FD;font-size:.85rem'>XGBoost</span>
        </div>
        <div style='display:flex;justify-content:space-between;margin-bottom:.3rem'>
            <span style='color:#94A3B8;font-size:.85rem'>AUC</span>
            <span style='color:#CBD5E1;font-size:.85rem'>0.8479</span>
        </div>
        <div style='display:flex;justify-content:space-between;margin-bottom:.3rem'>
            <span style='color:#94A3B8;font-size:.85rem'>Apnea Recall</span>
            <span style='color:#CBD5E1;font-size:.85rem'>80%</span>
        </div>
        <div style='display:flex;justify-content:space-between'>
            <span style='color:#94A3B8;font-size:.85rem'>Training Data</span>
            <span style='color:#CBD5E1;font-size:.85rem'>70 records · PhysioNet</span>
        </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("<div style='font-size:.7rem;color:#475569;text-transform:uppercase;letter-spacing:.08em;margin-bottom:.5rem'>HRV Features</div>", unsafe_allow_html=True)
    for o in ["mean_rr","std_rr","rmssd","pnn50","range_rr","lf_hf_ratio","sd2","sd1_sd2","sample_entropy","dfa_alpha"]:
        st.markdown("<div style='font-size:.78rem;color:#475569;padding:2px 0'>· " + o + "</div>", unsafe_allow_html=True)

    st.markdown("""
    <div style='font-size:.72rem;color:#334155;border-top:1px solid #1E2A3A;padding-top:.75rem;margin-top:1rem'>
        ⚠️ For research purposes only.<br>Does not provide medical diagnosis.<br><br>
        Pattern Recognition Course · 2025
    </div>
    """, unsafe_allow_html=True)

# ── ANA İÇERİK ─────────────────────────────────────────────────────────────────
st.markdown("""
<div class='aw-header'>
    <div class='aw-logo'>🫀</div>
    <div>
        <div class='aw-title'>ApneaWatch</div>
        <div class='aw-sub'>ECG-based sleep apnea detection system &nbsp;·&nbsp; <span class='badge badge-demo'>Demo</span></div>
    </div>
</div>
""", unsafe_allow_html=True)

st.markdown("""
<div class='upload-zone'>
    <div style='font-size:2rem'>📡</div>
    <h3>Upload ECG Record</h3>
    <p>PhysioNet Apnea-ECG format · .npy · Minimum 10 minutes</p>
</div>
""", unsafe_allow_html=True)

yuklenen = st.file_uploader("ECG file", type=["npy"], label_visibility="collapsed")
st.markdown("<br>", unsafe_allow_html=True)
analiz_btn = st.button("▶  Start Analysis", use_container_width=True, type="primary")

if analiz_btn:
    fs = 100

    if yuklenen is None:
        st.error("Please upload an ECG file first.")
        st.stop()

    ecg_uzun = np.load(io.BytesIO(yuklenen.read())).flatten()
    n_dk = len(ecg_uzun) // 6000

    if n_dk < 1:
        st.error("File is too short. Minimum 1 minute of recording is required.")
        st.stop()

    st.info("**" + str(n_dk) + "-minute** ECG successfully loaded. Starting analysis...")
    
    sonuclar = gece_analiz_canli(ecg_uzun, fs)

    gecerli       = [s for s in sonuclar if s['tahmin'] is not None]
    apne_sayisi   = sum(1 for s in gecerli if s['tahmin'] == 1)
    normal_sayisi = sum(1 for s in gecerli if s['tahmin'] == 0)
    apne_orani    = apne_sayisi / len(gecerli) * 100 if gecerli else 0

    rr_vals = [s['rr'] for s in gecerli if s['rr'] is not None]
    if rr_vals:
        tum_rr    = np.concatenate(rr_vals)
        ort_rmssd = np.sqrt(np.mean(np.diff(tum_rr)**2))
        ort_pnn50 = np.sum(np.abs(np.diff(tum_rr)) > 50) / len(tum_rr) * 100
        ort_hr    = 60000 / np.mean(tum_rr)
    else:
        ort_rmssd = ort_pnn50 = ort_hr = 0

    st.markdown("---")

    if apne_orani >= 50:
        st.markdown(
            "<div class='result-card danger'>"
            "<div class='rc-icon'>🚨</div>"
            "<div class='rc-title'>Moderate / Severe Sleep Apnea</div>"
            "<div class='rc-sub'>Apnea detected in " + str(apne_sayisi) + " / " + str(len(gecerli)) + " minutes · <strong>" + f"{apne_orani:.1f}" + "%</strong></div>"
            "<div class='rc-sub' style='margin-top:.5rem;font-size:.8rem'>Consulting a sleep specialist is highly recommended.</div>"
            "</div>", unsafe_allow_html=True)
    elif apne_orani >= 36:
        st.markdown(
            "<div class='result-card warning'>"
            "<div class='rc-icon'>⚠️</div>"
            "<div class='rc-title'>Mild Sleep Apnea</div>"
            "<div class='rc-sub'>Apnea detected in " + str(apne_sayisi) + " / " + str(len(gecerli)) + " minutes · <strong>" + f"{apne_orani:.1f}" + "%</strong></div>"
            "<div class='rc-sub' style='margin-top:.5rem;font-size:.8rem'>Regular clinical follow-up is recommended.</div>"
            "</div>", unsafe_allow_html=True)
    else:
        st.markdown(
            "<div class='result-card success'>"
            "<div class='rc-icon'>✅</div>"
            "<div class='rc-title'>Normal Sleep Pattern</div>"
            "<div class='rc-sub'>Apnea detected in " + str(apne_sayisi) + " / " + str(len(gecerli)) + " minutes · <strong>" + f"{apne_orani:.1f}" + "%</strong></div>"
            "</div>", unsafe_allow_html=True)

    c1, c2, c3, c4, c5 = st.columns(5)
    for col, val, lbl in [
        (c1, str(len(gecerli)),          "Total Minutes"),
        (c2, str(apne_sayisi),           "Apnea Minutes"),
        (c3, str(normal_sayisi),         "Normal Minutes"),
        (c4, f"{ort_hr:.0f} bpm",        "Avg Heart Rate"),
        (c5, f"{ort_rmssd:.1f} ms",      "RMSSD"),
    ]:
        col.markdown(
            "<div class='metric-box'>"
            "<div class='mb-val'>" + val + "</div>"
            "<div class='mb-lbl'>" + lbl + "</div>"
            "</div>", unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown("#### Overnight Apnea Map")

    harita_html = "<div class='night-map'>"
    for s in gecerli:
        if s['tahmin'] == 1:
            alpha = max(0.4, s['olasilik'])
            renk  = "rgba(239,68,68," + f"{alpha:.2f}" + ")"
        else:
            alpha = max(0.35, 1 - s['olasilik'])
            renk  = "rgba(34,197,94," + f"{alpha:.2f}" + ")"
        durum = "Apnea" if s['tahmin'] == 1 else "Normal"
        prob  = f"{s['olasilik']*100:.0f}"
        harita_html += "<div class='nm-min' style='background:" + renk + "' title='Min " + str(s['dakika']) + ": " + durum + " (" + prob + "%)'></div>"
    harita_html += "</div>"
    harita_html += (
        "<div style='display:flex;gap:16px;font-size:.78rem;color:#64748B;margin-top:4px'>"
        "<span><span style='display:inline-block;width:10px;height:10px;border-radius:2px;background:#ef4444;margin-right:5px'></span>Apnea</span>"
        "<span><span style='display:inline-block;width:10px;height:10px;border-radius:2px;background:#22c55e;margin-right:5px'></span>Normal</span>"
        "</div>"
    )
    st.markdown(harita_html, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown("#### Minute-by-Minute Apnea Probability")

    fig, ax = plt.subplots(figsize=(14, 3))
    fig.patch.set_facecolor('#0F1117')
    ax.set_facecolor('#0D1520')
    dakikalar   = [s['dakika']   for s in gecerli]
    olasiliklar = [s['olasilik'] for s in gecerli]
    renkler     = ['#ef4444' if s['tahmin'] == 1 else '#22c55e' for s in gecerli]
    ax.bar(dakikalar, olasiliklar, color=renkler, width=0.85, alpha=0.85)
    ax.axhline(0.55, color='#F59E0B', linewidth=1, linestyle='--', label='Threshold (0.55)')
    ax.set_ylim(0, 1)
    ax.set_xlabel("Minute", color='#64748B', fontsize=9)
    ax.set_ylabel("Apnea Probability", color='#64748B', fontsize=9)
    ax.tick_params(colors='#64748B', labelsize=8)
    for spine in ax.spines.values():
        spine.set_edgecolor('#1E2A3A')
    ax.legend(fontsize=8, facecolor='#161B27', labelcolor='#94A3B8', edgecolor='#1E2A3A')
    st.pyplot(fig)
    plt.close()

    with st.expander("📋 Minute-by-Minute Details"):
        cols = st.columns(5)
        for idx, s in enumerate(gecerli):
            durum = "🔴 Apnea" if s['tahmin'] == 1 else "🟢 Normal"
            prob  = f"{s['olasilik']*100:.0f}"
            cols[idx % 5].markdown(
                "<div style='font-size:.8rem;color:#CBD5E1;padding:2px 0'><b>Min " + str(s['dakika']) + ":</b> " + durum + " (" + prob + "%)</div>",
                unsafe_allow_html=True)