import os, re, json, time, uuid
import numpy as np
import streamlit as st
import joblib, torch
from transformers import AutoTokenizer, AutoModelForTokenClassification

# ====== UI ======
st.set_page_config(page_title="Chatbot Laporan Publik", page_icon="🛠️")
st.title("🛠️ Chatbot Laporan Publik (Prototype)")

# ====== Paths & Config ======
CLS_MODEL_PATH  = "models/text_cls_tfidf_lr.joblib"
CLS_LABELS_PATH = "models/text_labels.json"
NER_LOCAL_DIR   = "export/ner_loc/best"  # opsional jika kamu commit folder model
UNKNOWN_LABEL   = "tidak_tahu"

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
THRESHOLD = st.sidebar.slider("Ambang 'tidak_tahu'", 0.30, 0.90, 0.60, 0.01)

AGENCY_MAP = {
    "jalan_berlubang":  ("Dinas Bina Marga / PU",           "bina-marga@pemda.go.id"),
    "lampu_jalan_mati": ("Dinas Perhubungan (PJU)",          "dishub@pemda.go.id"),
    "rambu_rusak":      ("Dinas Perhubungan (Lalu Lintas)",  "dishub@pemda.go.id"),
    "sampah_menumpuk":  ("DLH / Dinas Kebersihan",           "dlh@pemda.go.id"),
    "tidak_tahu":       ("Pusat Layanan Publik",             "contact-center@pemda.go.id"),
}

# ====== Loaders ======
@st.cache_resource(show_spinner=False)
def load_cls():
    pipe = joblib.load(CLS_MODEL_PATH)
    with open(CLS_LABELS_PATH, "r", encoding="utf-8") as f:
        labels = json.load(f)
    return pipe, labels

@st.cache_resource(show_spinner=False)
def load_ner_manual():
    # 1) lokal
    if os.path.isdir(NER_LOCAL_DIR) and os.path.isfile(os.path.join(NER_LOCAL_DIR, "config.json")):
        src   = NER_LOCAL_DIR
        token = None
    else:
        # 2) Hugging Face Hub
        src   = st.secrets.get("NER_MODEL_ID", None)
        token = st.secrets.get("HF_API_TOKEN", None)
        if not src:
            st.warning("NER model tidak ditemukan (NER_MODEL_ID kosong). NER dimatikan.")
            return None, None
    tok = AutoTokenizer.from_pretrained(src, use_auth_token=token, use_fast=True)
    mdl = AutoModelForTokenClassification.from_pretrained(src, use_auth_token=token).to(DEVICE).eval()
    return tok, mdl

cls_pipe, CLS_LABELS = load_cls()
tok_ner, mdl_ner = load_ner_manual()

# ====== Helpers: Klasifikasi, NER, Ringkasan ======
def classify_text(text: str, threshold: float = 0.60):
    probs = cls_pipe.predict_proba([text])[0]
    idx   = int(np.argmax(probs))
    label, conf = cls_pipe.classes_[idx], float(probs[idx])
    if label != UNKNOWN_LABEL and conf < threshold:
        return UNKNOWN_LABEL, conf
    return label, conf

def ner_predict(text: str):
    if not tok_ner or not mdl_ner:
        return []
    enc = tok_ner(text, return_offsets_mapping=True, return_tensors="pt", truncation=True, max_length=256)
    offsets = enc.pop("offset_mapping")[0].tolist()
    with torch.no_grad():
        logits = mdl_ner(**{k: v.to(DEVICE) for k, v in enc.items()}).logits[0].cpu().numpy()
    pred_ids = logits.argmax(-1).tolist()
    id2label = mdl_ner.config.id2label

    ents, cur = [], None
    for (start, end), idx in zip(offsets, pred_ids):
        if start == end:
            continue
        label = id2label[int(idx)]
        if label == "O":
            if cur: ents.append(cur); cur = None
            continue
        if label.startswith("B-"):
            if cur: ents.append(cur)
            cur = {"entity_group": label[2:], "start": start, "end": end}
        elif label.startswith("I-"):
            grp = label[2:]
            if cur and cur["entity_group"] == grp:
                cur["end"] = end
            else:
                if cur: ents.append(cur)
                cur = {"entity_group": grp, "start": start, "end": end}
    if cur: ents.append(cur)
    for e in ents:
        e["word"] = text[e["start"]:e["end"]]
    return ents

def ner_extract(text: str):
    ents = ner_predict(text)
    locs = [e["word"].strip() for e in ents if e.get("entity_group") and e["entity_group"]!="O"]
    out = {"street": None, "city": None, "province": None}

    def _clean(s):  return re.sub(r"\s+", " ", s.strip(" ,.;:()[]{}"))
    def _title(s):  return " ".join(w.upper() if w.upper() in {"RT","RW","GG","PJU"} else w.capitalize() for w in s.split())
    def _looks_prov(s):
        t=_clean(s)
        return bool(re.match(r"(?i)^provinsi\s+\w+", t) or re.match(r"(?i)^(DI|Daerah Istimewa)\s+\w+", t))

    street_prefixes = ("jalan","jl","jln","gang","gg")
    streets = [s for s in locs if s.lower().startswith(street_prefixes)]
    if streets:
        streets.sort(key=len, reverse=True)
        out["street"] = _title(_clean(re.sub(r"\b(ada|banyak|gundukan|sampah|lubang|rusak|mati|menumpuk).*$","",streets[0], flags=re.IGNORECASE)))

    provs = [s for s in locs if _looks_prov(s)]
    if provs:
        provs.sort(key=len)
        out["province"] = _title(_clean(provs[0]))

    rem=[]
    for s in locs:
        sc=_clean(s)
        if out["street"] and sc.lower()==(out["street"] or "").lower():    continue
        if out["province"] and sc.lower()==(out["province"] or "").lower(): continue
        rem.append(sc)
    if rem:
        rem.sort(key=len)
        out["city"] = _title(rem[0])

    if out.get("street"):
        for kw in [" Jakarta"," Bandung"," Surabaya"," Semarang"," Yogyakarta"," Medan"," Makassar"," Malang"," Depok"," Bekasi"]:
            if kw.lower().strip() in out["street"].lower():
                out["street"] = _title(_clean(out["street"].replace(kw.strip(), "")))
                out["city"]   = _title(kw.strip())
                break
    return {k:v for k,v in out.items() if v}

def format_loc(d):
    parts=[]
    if d.get("street"):   parts.append(f"Jalan/Gang: {d['street']}")
    if d.get("city"):     parts.append(f"Kota/Kab: {d['city']}")
    if d.get("province"): parts.append(f"Provinsi: {d['province']}")
    return " | ".join(parts) if parts else "(lokasi belum terdeteksi)"

def ticket_id():
    return "TCK-" + time.strftime("%Y%m%d") + "-" + uuid.uuid4().hex[:6].upper()

def normalize_yesno(s: str) -> str:
    s = (s or "").strip().lower()
    yes = {"y","ya","yes","ok","oke","kirim","send","iya","yup"}
    no  = {"n","no","tidak","jangan","batal","cancel","nggak","ga"}
    if s in yes: return "yes"
    if s in no:  return "no"
    return ""

def build_confirmation(label: str, conf: float, loc: dict):
    tujuan = AGENCY_MAP.get(label, AGENCY_MAP[UNKNOWN_LABEL])
    return (
        f"Kategori: **{label}** (conf {conf:.2f})\n"
        f"Lokasi: {format_loc(loc)}\n"
        f"Instansi: {tujuan[0]} ({tujuan[1]})\n\n"
        f"Kirim sekarang? (y/n)"
    )

# ====== Session State ======
if "messages" not in st.session_state: st.session_state.messages = []
if "pending"  not in st.session_state: st.session_state.pending  = None  # {'text','label','conf','loc'}

# render history
for role, content in st.session_state.messages:
    with st.chat_message(role):
        st.write(content)

# ====== Chat Loop ======
user_text = st.chat_input("Tulis laporan Anda…")
if user_text:
    with st.chat_message("user"): st.write(user_text)
    st.session_state.messages.append(("user", user_text))

    # jika sedang menunggu konfirmasi
    if st.session_state.pending:
        ans = normalize_yesno(user_text)
        if ans == "yes":
            t = ticket_id()
            label = st.session_state.pending["label"]
            tujuan = AGENCY_MAP.get(label, AGENCY_MAP[UNKNOWN_LABEL])
            sent = (
                f"📨 Terkirim!\n"
                f"• Nomor tiket : {t}\n"
                f"• Tujuan      : {tujuan[0]} ({tujuan[1]})\n"
                f"• Ringkasan   : {label} (conf {st.session_state.pending['conf']:.2f}) | {format_loc(st.session_state.pending['loc'])}"
            )
            with st.chat_message("assistant"): st.write(sent)
            st.session_state.messages.append(("assistant", sent))
            st.session_state.pending = None
            st.stop()

        elif ans == "no":
            st.session_state.pending = None
            with st.chat_message("assistant"):
                st.write("Baik, dibatalkan. Silakan kirim ulang laporan dengan perbaikan yang diperlukan.")
            st.session_state.messages.append(("assistant", "Dibatalkan oleh pengguna."))
            st.stop()

        else:
            # anggap ini info tambahan: gabungkan & re-run NER/klasifikasi
            base_text = st.session_state.pending["text"]
            merged = base_text.strip() + ". " + user_text.strip()
            label, conf = classify_text(merged, THRESHOLD)
            loc = ner_extract(merged)
            st.session_state.pending = {"text": merged, "label": label, "conf": conf, "loc": loc}
            reply = build_confirmation(label, conf, loc)
            with st.chat_message("assistant"): st.write(reply)
            st.session_state.messages.append(("assistant", reply))
            st.stop()

    # tidak sedang menunggu: proses teks baru
    label, conf = classify_text(user_text, THRESHOLD)
    loc = ner_extract(user_text)
    st.session_state.pending = {"text": user_text, "label": label, "conf": conf, "loc": loc}
    reply = build_confirmation(label, conf, loc)
    with st.chat_message("assistant"): st.write(reply)
    st.session_state.messages.append(("assistant", reply))
