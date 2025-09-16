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

if "messages" not in st.session_state: st.session_state.messages = []
if "pending"  not in st.session_state: st.session_state.pending = None  # simpan {label, conf, text}

for role, content in st.session_state.messages:
    with st.chat_message(role):
        st.write(content)

user_text = st.chat_input("Tulis laporan Anda…")
if user_text:
    # 1) show user message
    st.session_state.messages.append(("user", user_text))
    with st.chat_message("user"):
        st.write(user_text)

    # 2) classify
    label, conf, scores = classify_text_ml(user_text, unknown_threshold=UNKNOWN_THRESHOLD)
    st.session_state.pending = {"label": label, "conf": conf, "text": user_text}

    # 3) get LLM's response
    uprompt = build_user_prompt(user_text, label, conf)
    reply = llm_reply(SYSTEM_CHAT, uprompt)

    with st.chat_message("assistant"):
        st.write(reply)
        st.button(" Kirim ke instansi", key=f"send_{len(st.session_state.messages)}")

    st.session_state.messages.append(("assistant", reply))
    st.rerun()
# send button
clicked_keys = [k for k in st.session_state.keys() if str(k).startswith("send_")]
if clicked_keys and st.session_state.pending:
    # 1) buat nomor tiket & tentukan tujuan
    t = ticket_id()
    label = st.session_state.pending["label"]
    conf  = st.session_state.pending["conf"]
    text_ = st.session_state.pending["text"]
    tujuan = AGENCY_MAP.get(label, AGENCY_MAP["tidak_tahu"])

    # 2) ringkasan tiket
    summary_user_prompt = (
        f"Kategori terdeteksi: {label} (confidence {conf:.2f}). "
        f"Instansi tujuan: {tujuan['nama']} ({tujuan['kontak']}). "
        f"Teks pengguna: {text_}\n\n"
        "Tolong buat ringkasan tiket singkat (1–2 kalimat), tanpa bertanya balik."
    )
    summary = llm_reply(SYSTEM_SUMMARY, summary_user_prompt)

    # 3) tampilkan konfirmasi terkirim + RINGKASAN dari LLM
    confirmation = (
      f"📨 Terkirim!\n"
      f"• Nomor tiket : {t}\n"
      f"• Tujuan      : {tujuan['nama']} ({tujuan['kontak']})\n"
      f"• Ringkasan   : {summary}"
    )
    st.session_state.messages.append(("assistant", confirmation))

    # 4) bersihkan pending
    st.session_state.pending = None
    st.rerun()
