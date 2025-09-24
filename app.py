import os, json, re, numpy as np, streamlit as st, joblib
from transformers import pipeline

st.set_page_config(page_title="Chatbot Laporan Publik", page_icon="🛠️")
st.title("🛠️ Chatbot Laporan Publik")

# ==== Paths (sesuaikan kalau perlu) ====
MODEL_PATH     = "models/text_cls_tfidf_lr.joblib"
LABELS_PATH    = "models/text_labels.json"
NER_MODEL_DIR  = "export/ner_loc/best"   # atau "models/ner_loc/best" jika dipindah

# ==== Load models ====
@st.cache_resource(show_spinner=False)
def load_cls():
    pipe = joblib.load(MODEL_PATH)
    with open(LABELS_PATH) as f: labels = json.load(f)
    return pipe, labels

@st.cache_resource(show_spinner=False)
def load_ner():
    return pipeline("ner", model=NER_MODEL_DIR, tokenizer=NER_MODEL_DIR, aggregation_strategy="simple")

cls_pipe, LABELS = load_cls()
ner_pipe = load_ner()

UNKNOWN_LABEL = "tidak_tahu"
thresh = st.sidebar.slider("Ambang 'tidak_tahu'", 0.30, 0.90, 0.60, 0.01)

# ==== Helpers ====
def classify_text(text: str, unknown_threshold: float):
    probs = cls_pipe.predict_proba([text])[0]
    idx = int(np.argmax(probs))
    label, conf = cls_pipe.classes_[idx], float(probs[idx])
    if label != UNKNOWN_LABEL and conf < unknown_threshold: return UNKNOWN_LABEL, conf
    return label, conf

STREET_PREFIXES = ("jalan", "jl", "jln", "gang", "gg")
def _clean(s): return re.sub(r"\s+"," ",s.strip(" ,.;:()[]{}"))
def _title(s): return " ".join(w.upper() if w.upper() in {"RT","RW","GG","PJU"} else w.capitalize() for w in s.split())
def _looks_street(x): return any(x.lower().startswith(p+" ") for p in STREET_PREFIXES)
def _looks_prov(x):
    t=_clean(x)
    if len(t)>30: return False
    return bool(re.match(r"(?i)^provinsi\s+[A-Za-z][A-Za-z .\-]+$",t) or
                re.match(r"(?i)^(DI|Daerah Istimewa)\s+[A-Za-z][A-Za-z .\-]+$",t))

def _normalize_city(x: str) -> str:
    t = _clean(x); tl = t.lower()
    for p in ["kota","kabupaten","kab","kodya","kec","kecamatan"]:
        if tl.startswith(p+" "): t=t[len(p)+1:]; break
    return _title(t)

def extract_location(text: str):
    ents = ner_pipe(text)
    locs = [e["word"].strip() for e in ents if e.get("entity_group","")!="O"]
    out = {"street":None,"city":None,"province":None}
    # street
    streets=[s for s in locs if _looks_street(s)]
    if streets:
        streets.sort(key=len, reverse=True)
        out["street"] = _title(_clean(re.sub(r"\b(ada|banyak|gundukan|sampah|lubang|rusak|mati|menumpuk).*$","",streets[0],flags=re.IGNORECASE)))
    # province
    provs=[s for s in locs if _looks_prov(s)]
    if provs:
        provs.sort(key=len)
        out["province"] = _title(_clean(provs[0]))
    # city (sisa terpendek / setelah “di ”)
    rem=[]
    for s in locs:
        sc=_clean(s)
        if out.get("street") and sc.lower()==out["street"].lower(): continue
        if out.get("province") and sc.lower()==out["province"].lower(): continue
        rem.append(sc)
    if rem:
        tl=f" {text} ".lower()
        after_di=[s for s in rem if f" di {s.lower()}" in tl]
        cand = after_di if after_di else rem
        cand=[c for c in cand if not _looks_street(c) and not _looks_prov(c)]
        if cand:
            cand.sort(key=len)
            out["city"] = _normalize_city(cand[0])
    # pecah kasus gabungan: "Jalan X Kota Y"
    if out.get("street"):
        for kw in [" Jakarta"," Bandung"," Surabaya"," Semarang"," Yogyakarta"," Medan"," Makassar"]:
            if kw.lower().strip() in out["street"].lower():
                out["street"] = _title(_clean(out["street"].replace(kw.strip(), "")))
                out["city"] = _normalize_city(kw.strip())
                break
    return {k:v for k,v in out.items() if v}

def format_loc(f):
    p=[]
    if f.get("street"): p.append(f"Jalan/Gang: {f['street']}")
    if f.get("city"): p.append(f"Kota/Kab: {f['city']}")
    if f.get("province"): p.append(f"Provinsi: {f['province']}")
    return " | ".join(p) if p else "(lokasi belum terdeteksi)"

# ==== UI ====
user_text = st.chat_input("Tulis laporan Anda…")
if user_text:
    with st.chat_message("user"): st.write(user_text)

    label, conf = classify_text(user_text, thresh)
    loc = extract_location(user_text)

    tujuan_map = {
        "jalan_berlubang": ("Dinas Bina Marga / PU","bina-marga@pemda.go.id"),
        "lampu_jalan_mati":("Dinas Perhubungan (PJU)","dishub@pemda.go.id"),
        "rambu_rusak":     ("Dinas Perhubungan (Lalu Lintas)","dishub@pemda.go.id"),
        "sampah_menumpuk": ("DLH / Dinas Kebersihan","dlh@pemda.go.id"),
        "tidak_tahu":      ("Pusat Layanan Publik","contact-center@pemda.go.id"),
    }
    instansi = tujuan_map.get(label, tujuan_map["tidak_tahu"])

    reply = (
        f"Kategori terdeteksi: **{label}** (conf {conf:.2f})\n"
        f"Lokasi: {format_loc(loc)}\n"
        f"Instansi tujuan: {instansi[0]} ({instansi[1]})\n\n"
        f"Kirim sekarang? (y/n)"
    )
    with st.chat_message("assistant"):
        st.write(reply)
