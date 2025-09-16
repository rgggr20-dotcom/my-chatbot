import os, json, time, uuid, random
import numpy as np
import pandas as pd
import streamlit as st
import joblib, requests, httpx

st.set_page_config(page_title="Chatbot Laporan Publik", page_icon="🛠️")
st.title("🛠️ Chatbot Laporan Publik (Prototype)")

#Threshold if don't know
UNKNOWN_THRESHOLD = st.sidebar.slider("Ambang 'tidak_tahu'", 0.3, 0.9, 0.55, 0.01)

# Model & labels
MODEL_PATH = "models/text_cls_tfidf_lr.joblib"
LABELS_PATH = "models/text_labels.json"

#test other models soon
OPENROUTER_MODEL = "google/gemma-2-9b-it"
HF_MODEL         = "mistralai/Mistral-7B-Instruct-v0.2"

#mapping
AGENCY_MAP = {
    "jalan_berlubang":  {"nama": "Dinas Bina Marga / PU",          "kontak": "bina-marga@pemda.go.id"},
    "lampu_jalan_mati": {"nama": "Dinas Perhubungan (PJU)",         "kontak": "dishub@pemda.go.id"},
    "rambu_rusak":      {"nama": "Dinas Perhubungan (Lalu Lintas)", "kontak": "dishub@pemda.go.id"},
    "sampah_menumpuk":  {"nama": "DLH / Dinas Kebersihan",          "kontak": "dlh@pemda.go.id"},
    "tidak_tahu":       {"nama": "Pusat Layanan Publik",            "kontak": "contact-center@pemda.go.id"},
}

@st.cache_resource(show_spinner=False)
def load_text_model():
    pipe = joblib.load(MODEL_PATH)
    with open(LABELS_PATH) as f:
        labels = json.load(f)
    return pipe, labels

pipe, LABELS = load_text_model()

def classify_text_ml(txt: str, unknown_threshold=0.55):
    probs = pipe.predict_proba([txt])[0]
    idx = int(np.argmax(probs))
    label, conf = pipe.classes_[idx], float(probs[idx])
    scores = dict(zip(pipe.classes_, probs))
    if label != "tidak_tahu" and conf < unknown_threshold:
        return "tidak_tahu", conf, scores
    return label, conf, scores
def clean_val(s: str) -> str:
    import re
    s = s.strip(" ,.;:()[]{}")
    return re.sub(r'\s+', ' ', s)

import re
RE_STREET = re.compile(r'\b(?:jalan|jl|jln|gang|gg)\.?\s+([A-Za-z0-9 .\-]+)', re.IGNORECASE)
RE_CITY   = re.compile(r'\b(kota|kab(?:upaten)?|kec(?:amatan)?)\s+([A-Za-z .\-]+)', re.IGNORECASE)
RE_CITY_FALLBACK = re.compile(r'\bdi\s+([A-Z][A-Za-z .\-]+)$')
KNOWN_PROV = ["DKI Jakarta","Jawa Barat","Jawa Tengah","DI Yogyakarta","Jawa Timur","Banten"]
RE_PROV   = re.compile("|".join(map(re.escape, KNOWN_PROV)), re.IGNORECASE)

def extract_fields(text: str) -> dict:
    out = {}
    m = RE_STREET.search(text)
    if m: out["street"] = clean_val(m.group(1))
    m = RE_CITY.search(text)
    if m: out["city"] = clean_val(m.group(2))
    else:
        m2 = RE_CITY_FALLBACK.search(text)
        if m2: out["city"] = clean_val(m2.group(1))
    m = RE_PROV.search(text)
    if m: out["province"] = m.group(0)
    return out

def update_form_from_text(text, form, detected_label=None):
    form["raw_texts"].append(text)
    if detected_label: form["category"] = detected_label
    fx = extract_fields(text)
    for k in ["street","city","province"]:
        if k in fx and fx[k]:
            form[k] = fx[k]
    return form

def next_missing_slot(form):
    for k in ["street","city","province"]:
        if not form.get(k):
            return k
    return None

def ask_for(slot):
    if slot == "street": return "Alamat jalannya apa ya?"
    if slot == "city": return "Ini di kota/kabupaten mana ya?"
    if slot == "province": return "Provinsinya di mana ya?"
    return None

def format_location(form):
    parts = []
    if form.get("street"):   parts.append(f"Jalan/Gang: {form['street']}")
    if form.get("city"):     parts.append(f"Kota/Kab: {form['city']}")
    if form.get("province"): parts.append(f"Provinsi: {form['province']}")
    return " | ".join(parts) if parts else "(lokasi belum lengkap)"
    
def llm_openrouter(system_prompt: str, user_prompt: str) -> str | None:
    api_key = st.secrets.get("OPENROUTER_API_KEY", None)
    if not api_key: return None
    url = "https://openrouter.ai/api/v1/chat/completions"
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }
    body = {
        "model": OPENROUTER_MODEL,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user",   "content": user_prompt},
        ],
        "temperature": 0.6,
        "max_tokens": 220,
    }
    try:
        r = httpx.post(url, headers=headers, json=body, timeout=60)
        r.raise_for_status()
        return r.json()["choices"][0]["message"]["content"].strip()
    except Exception as e:
        st.warning(f"[OpenRouter] {e}")
        return None

def llm_hf_inference(system_prompt: str, user_prompt: str) -> str | None:
    token = st.secrets.get("HF_API_TOKEN", None)
    if not token: return None
    url = f"https://api-inference.huggingface.co/models/{HF_MODEL}"
    headers = {"Authorization": f"Bearer {token}"}
    prompt = f"[SYSTEM]\n{system_prompt}\n\n[USER]\n{user_prompt}\n\n[ASSISTANT]\n"
    payload = {"inputs": prompt, "parameters": {"max_new_tokens": 220, "temperature": 0.6}}
    try:
        r = requests.post(url, headers=headers, json=payload, timeout=60)
        r.raise_for_status()
        data = r.json()
        if isinstance(data, list) and data and "generated_text" in data[0]:
            out = data[0]["generated_text"]
            return out.split("[ASSISTANT]")[-1].strip()
        elif isinstance(data, dict) and "error" in data:
            st.warning(f"[HF Inference] {data['error']}")
            return None
        return None
    except Exception as e:
        st.warning(f"[HF Inference] {e}")
        return None

def llm_reply(system_prompt: str, user_prompt: str) -> str:
    # 1) test OpenRouter
    out = llm_openrouter(system_prompt, user_prompt)
    if out: return out
    # 2) test HF Inference
    out = llm_hf_inference(system_prompt, user_prompt)
    if out: return out
    # 3) fallback template
    return "Terima kasih. Laporan Anda sudah kami terima dan akan diproses. (Catatan: LLM tidak aktif)"

#prompt
SYSTEM_CHAT = (
  "Kamu adalah asisten layanan publik Indonesia."
  "Balas dengan sopan dan ringkas. Mulai dengan berterima kasih pada user, kemudian sebutkan kategori laporan, jika ada sebutkan lokasi tempat (seperti jalan/gang apa, kota, & provinsi) , sebutkan juga instansi tujuan. Sebutkan dalam format list"
  "Kategori ada jalan berlubang, sampah menumpuk, lampu jalan rusak, dan rambu rusak, jika ada yang tidak cocok dengan kategori, generate kategori baru"
  "Akhiri dengan pertanyaan: 'Kirim sekarang? (y/n)'."
)
#prompt untuk tiket
SYSTEM_SUMMARY = (
  "Kamu adalah sistem pencatat tiket. Buat ringkasan 1–2 kalimat yang padat, "
  "sebutkan kategori, lokasi (jika ada), dampak/urgensi jika tersirat, dan tanpa bertanya balik. "
  "JANGAN akhiri dengan pertanyaan."
)
def build_user_prompt(user_text: str, label: str, conf: float) -> str:
    tujuan = AGENCY_MAP.get(label, AGENCY_MAP["tidak_tahu"])
    context = (
      f"Kategori terdeteksi: {label} (confidence {conf:.2f}). "
      f"Instansi: {tujuan['nama']} | Kontak: {tujuan['kontak']}."
    )
    return f"{context}\n\nTeks pengguna: {user_text}"

def ticket_id():
    return "TCK-" + time.strftime("%Y%m%d") + "-" + uuid.uuid4().hex[:6].upper()

if "messages" not in st.session_state:
    st.session_state.messages = []
if "pending" not in st.session_state:
    st.session_state.pending = None  # {label, conf, text}
if "form" not in st.session_state:
    st.session_state.form = {"category": None, "street": None, "city": None, "province": None, "raw_texts": []}


for role, content in st.session_state.messages:
    with st.chat_message(role):
        st.write(content)

user_text = st.chat_input("Tulis laporan Anda…")
if user_text:
    # tampilkan user
    st.session_state.messages.append(("user", user_text))
    with st.chat_message("user"):
        st.write(user_text)

    # klasifikasi
    label, conf, scores = classify_text_ml(user_text, unknown_threshold=UNKNOWN_THRESHOLD)
    st.session_state.pending = {"label": label, "conf": conf, "text": user_text}

    # update form (overwrite kalau ada koreksi)
    st.session_state.form = update_form_from_text(
        user_text,
        st.session_state.form,
        detected_label=label
    )

    # cek slot kosong
    need = next_missing_slot(st.session_state.form)
    if need:
        q = ask_for(need)
        bot_msg = (
            f"Kategori terdeteksi: **{label}** (conf {conf:.2f})\n"
            f"Lokasi saat ini: {format_location(st.session_state.form)}\n\n"
            f"{q}"
        )
        st.session_state.messages.append(("assistant", bot_msg))
        with st.chat_message("assistant"): st.write(bot_msg)

    else:
        # semua slot lengkap → minta konfirmasi via LLM
        tujuan = AGENCY_MAP.get(label, AGENCY_MAP["tidak_tahu"])
        uprompt = build_user_prompt(
            f"{user_text}\n\nLokasi terstruktur: {format_location(st.session_state.form)}",
            label, conf
        )
        reply = llm_reply(SYSTEM_CHAT, uprompt)

        st.session_state.messages.append(("assistant", reply))
        with st.chat_message("assistant"):
            st.write(reply)
            st.button("Kirim ke instansi", key=f"send_{len(st.session_state.messages)}")

    st.rerun()
# send button
clicked_keys = [k for k in st.session_state.keys() if str(k).startswith("send_")]
if clicked_keys and st.session_state.pending:
    t = ticket_id()
    label = st.session_state.pending["label"]
    conf  = st.session_state.pending["conf"]
    text_ = st.session_state.pending["text"]
    tujuan = AGENCY_MAP.get(label, AGENCY_MAP["tidak_tahu"])
    loc_struct = format_location(st.session_state.form)

    summary_user_prompt = (
        f"Kategori: {label} (confidence {conf:.2f}). "
        f"Lokasi: {loc_struct}. "
        f"Instansi tujuan: {tujuan['nama']} ({tujuan['kontak']}). "
        f"Teks pengguna (gabungan): {' | '.join(st.session_state.form['raw_texts'])}\n\n"
        "Buat ringkasan tiket singkat (1–2 kalimat), tanpa bertanya balik."
    )
    summary = llm_reply(SYSTEM_SUMMARY, summary_user_prompt)

    confirmation = (
      f"📨 Terkirim!\n"
      f"• Nomor tiket : {t}\n"
      f"• Tujuan      : {tujuan['nama']} ({tujuan['kontak']})\n"
      f"• Lokasi      : {loc_struct}\n"
      f"• Ringkasan   : {summary}"
    )
    st.session_state.messages.append(("assistant", confirmation))

    # reset state untuk laporan berikut
    st.session_state.pending = None
    st.session_state.form = {"category": None, "street": None, "city": None, "province": None, "raw_texts": []}
    st.rerun()

