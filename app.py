# app.py
import os, json, time, uuid, re
import numpy as np
import streamlit as st
import joblib, requests, httpx
    
# =========[ UI & CONFIG ]=========
st.set_page_config(page_title="Chatbot Laporan Publik", page_icon="🛠️")
st.title("🛠️ Chatbot Laporan Publik (Prototype)")

UNKNOWN_LABEL     = "tidak_tahu"
DEFAULT_THRESHOLD = 0.60
UNKNOWN_THRESHOLD = st.sidebar.slider("Ambang 'tidak_tahu'", 0.30, 0.90, DEFAULT_THRESHOLD, 0.01)

MODEL_PATH  = "models/text_cls_tfidf_lr.joblib"
LABELS_PATH = "models/text_labels.json"

# pilih salah satu yang tersedia di secrets
OPENROUTER_MODEL = "google/gemma-2-9b-it"
HF_MODEL         = "mistralai/Mistral-7B-Instruct-v0.2"

AGENCY_MAP = {
    "jalan_berlubang":  {"nama": "Dinas Bina Marga / PU",          "kontak": "bina-marga@pemda.go.id"},
    "lampu_jalan_mati": {"nama": "Dinas Perhubungan (PJU)",         "kontak": "dishub@pemda.go.id"},
    "rambu_rusak":      {"nama": "Dinas Perhubungan (Lalu Lintas)", "kontak": "dishub@pemda.go.id"},
    "sampah_menumpuk":  {"nama": "DLH / Dinas Kebersihan",          "kontak": "dlh@pemda.go.id"},
    "tidak_tahu":       {"nama": "Pusat Layanan Publik",            "kontak": "contact-center@pemda.go.id"},
}

# =========[ LOAD TEXT CLASSIFIER ]=========
@st.cache_resource(show_spinner=False)
def load_text_model():
    pipe = joblib.load(MODEL_PATH)
    with open(LABELS_PATH) as f:
        labels = json.load(f)
    return pipe, labels

pipe, LABELS = load_text_model()

def classify_text_ml(txt: str, unknown_threshold=DEFAULT_THRESHOLD):
    probs = pipe.predict_proba([txt])[0]
    idx = int(np.argmax(probs))
    label, conf = pipe.classes_[idx], float(probs[idx])
    scores = dict(zip(pipe.classes_, probs))
    if label != UNKNOWN_LABEL and conf < unknown_threshold:
        return UNKNOWN_LABEL, conf, scores
    return label, conf, scores

# =========[ LLM BRIDGES ]=========
def llm_openrouter(system_prompt: str, user_prompt: str) -> str | None:
    api_key = st.secrets.get("OPENROUTER_API_KEY")
    if not api_key: return None
    url = "https://openrouter.ai/api/v1/chat/completions"
    headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
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
    token = st.secrets.get("HF_API_TOKEN")
    if not token: return None
    url = f"https://api-inference.huggingface.co/models/{HF_MODEL}"
    headers = {"Authorization": f"Bearer {token}"}
    prompt  = f"[SYSTEM]\n{system_prompt}\n\n[USER]\n{user_prompt}\n\n[ASSISTANT]\n"
    payload = {"inputs": prompt, "parameters": {"max_new_tokens": 220, "temperature": 0.6}}
    try:
        r = requests.post(url, headers=headers, json=payload, timeout=60)
        r.raise_for_status()
        data = r.json()
        if isinstance(data, list) and data and "generated_text" in data[0]:
            out = data[0]["generated_text"]
            return out.split("[ASSISTANT]")[-1].strip()
        if isinstance(data, dict) and "error" in data:
            st.warning(f"[HF Inference] {data['error']}")
        return None
    except Exception as e:
        st.warning(f"[HF Inference] {e}")
        return None

def llm_reply(system_prompt: str, user_prompt: str) -> str:
    out = llm_openrouter(system_prompt, user_prompt)
    if out: return out
    out = llm_hf_inference(system_prompt, user_prompt)
    if out: return out
    return "Terima kasih. (Catatan: LLM tidak aktif)"

# =========[ PROMPTS ]=========
SYSTEM_CHAT = (
  "Kamu asisten layanan publik Indonesia. Balas sopan & ringkas; "
  "sebutkan kategori, lokasi (jalan/kota/provinsi jika terdeteksi), serta instansi tujuan. "
  "Akhiri dengan pertanyaan: 'Kirim sekarang? (y/n)'."
)
SYSTEM_SUMMARY = (
  "Kamu sistem pencatat tiket. Buat ringkasan 1–2 kalimat yang padat, "
  "sebutkan kategori, lokasi (jalan/kota/provinsi), dan dampak/urgensi jika tersirat. "
  "JANGAN bertanya balik."
)

# =========[ NER: IndoBERT (tanpa CITY2PROV) ]=========
from transformers import AutoTokenizer, AutoModelForTokenClassification, pipeline

@st.cache_resource(show_spinner=False)
def load_ner_model():
    MODEL_NER = "cahya/bert-base-indonesian-NER"
    tok = AutoTokenizer.from_pretrained(MODEL_NER)
    ner_model = AutoModelForTokenClassification.from_pretrained(MODEL_NER)
    nlp = pipeline("ner", model=ner_model, tokenizer=tok, aggregation_strategy="simple")
    return nlp

ner_pipe = load_ner_model()

STREET_PREFIXES = ("jalan", "jl", "jln", "gang", "gg")

def _clean(s: str) -> str:
    s = re.sub(r"\s+", " ", s.strip(" ,.;:()[]{}"))
    return s

def _title_keep_upper(s: str) -> str:
    return " ".join(w.upper() if w.upper() in {"RT","RW","GG","PJU"} else w.capitalize() for w in s.split())

def _looks_like_street(text_span: str) -> bool:
    t = text_span.strip().lower()
    return any(t.startswith(p + " ") for p in STREET_PREFIXES)

def _looks_like_province(text_span: str) -> bool:
    # Tanpa kamus: terima eksplisit "provinsi xxx" atau "DI Yogyakarta"
    t = text_span.strip()
    if re.search(r"\bprovinsi\s+[A-Za-z .-]+", t, flags=re.IGNORECASE):
        return True
    if re.match(r"^(DI|Daerah Istimewa)\b", t, flags=re.IGNORECASE):
        return True
    return False

def _pick_city(text: str, loc_spans: list[str]) -> str | None:
    """
    Pilih city dari LOC (tanpa kamus):
    1) LOC yang muncul setelah ' di '
    2) kalau tidak ada, pilih LOC terpendek (1–2 kata), bukan street/province
    """
    tlow = f" {text} ".lower()
    after_di = []
    for span in loc_spans:
        s = span.strip()
        if f" di {s.lower()}" in tlow:
            after_di.append(s)

    candidates = after_di if after_di else loc_spans
    candidates = [c for c in candidates if not _looks_like_street(c) and not _looks_like_province(c)]
    if not candidates:
        return None
    candidates.sort(key=lambda x: len(x))
    return _title_keep_upper(_clean(candidates[0]))
    
def normalize_city_name(s: str) -> str:
    t = _clean(s)
    tl = t.lower()
    for p in ["kota", "kabupaten", "kab", "kodya", "kec", "kecamatan"]:
        if tl.startswith(p + " "):
            t = t[len(p) + 1:]  # buang prefix + spasi
            break
    return _title_keep_upper(t)

def extract_fields(text: str) -> dict:
    """
    Ekstraksi lokasi via IndoBERT NER (tanpa mapping kota→provinsi).
    - street: LOC diawali 'jalan/jl/jln/gg'
    - city  : LOC lain (heuristik: setelah 'di', atau LOC terpendek)
    - province: hanya jika user sebut eksplisit ('provinsi ...' / 'DI ...')
    """
    out = {"street": None, "city": None, "province": None}

    ents = ner_pipe(text)
    locs = [e["word"].strip() for e in ents if e.get("entity_group", "") == "LOC"]
    if not locs:
        return out

    # street
    streets = [s for s in locs if _looks_like_street(s)]
    if streets:
        streets.sort(key=lambda x: len(x), reverse=True)
        st_val = _title_keep_upper(_clean(re.sub(
            r"\b(ada|banyak|gundukan|sampah|lubang|rusak|mati|menumpuk).*$",
            "", streets[0], flags=re.IGNORECASE
        )))
        out["street"] = st_val if st_val else None

    # province (eksplisit)
    prov_spans = [s for s in locs if _looks_like_province(s)]
    if prov_spans:
        out["province"] = _title_keep_upper(_clean(prov_spans[0]))

    # city dari sisa LOC
    remaining = []
    for s in locs:
        if out.get("street") and s.strip().lower() == out["street"].lower():
            continue
        if out.get("province") and s.strip().lower() == out["province"].lower():
            continue
        remaining.append(s)
    city = _pick_city(text, remaining)
    if city:
        out["city"] = city

    return {k: v for k, v in out.items() if v}

# =========[ SLOT-FILLING HELPERS ]=========
def format_location(form):
    parts = []
    if form.get("street"):   parts.append(f"Jalan/Gang: {form['street']}")
    if form.get("city"):     parts.append(f"Kota/Kab: {form['city']}")
    if form.get("province"): parts.append(f"Provinsi: {form['province']}")
    return " | ".join(parts) if parts else "(lokasi belum lengkap)"

def update_form_from_text(text, form, detected_label=None):
    form["raw_texts"].append(text)
    if detected_label:
        form["category"] = detected_label
    fx = extract_fields(text)
    if fx.get("street"):   form["street"]   = fx["street"]
    if fx.get("city"):     form["city"]     = fx["city"]
    if fx.get("province"): form["province"] = fx["province"]
    return form

def next_missing_slot(form):
    for k in ["street","city","province"]:
        if not form.get(k):
            return k
    return None

def ask_for(slot):
    return {
        "street":   "Alamat jalannya apa ya? (contoh: Jalan Malaka)",
        "city":     "Ini di kota/kabupaten mana ya?",
        "province": "Provinsinya di mana ya?",
    }.get(slot)

def build_user_prompt(user_text: str, label: str, conf: float, loc_struct: str) -> str:
    tujuan = AGENCY_MAP.get(label, AGENCY_MAP[UNKNOWN_LABEL])
    context = (
      f"Kategori terdeteksi: {label} (confidence {conf:.2f}). "
      f"Instansi: {tujuan['nama']} | Kontak: {tujuan['kontak']}. "
      f"Lokasi terstruktur: {loc_struct}."
    )
    return f"{context}\n\nTeks pengguna: {user_text}"

def build_ticket_prompt(label, conf, loc_struct, tujuan, raw_texts):
    return (
        f"Kategori: {label} (confidence {conf:.2f}). "
        f"Lokasi: {loc_struct}. "
        f"Instansi tujuan: {tujuan['nama']} ({tujuan['kontak']}). "
        f"Teks pengguna (gabungan): {' | '.join(raw_texts)}\n\n"
        "Buat ringkasan tiket singkat (1–2 kalimat), tanpa bertanya balik."
    )

def ticket_id():
    return "TCK-" + time.strftime("%Y%m%d") + "-" + uuid.uuid4().hex[:6].upper()

# =========[ SESSION STATE ]=========
if "messages" not in st.session_state: st.session_state.messages = []
if "pending"  not in st.session_state: st.session_state.pending = None  # {label, conf, text}
if "form"     not in st.session_state: st.session_state.form = {"category": None, "street": None, "city": None, "province": None, "raw_texts": []}
if "mode"     not in st.session_state: st.session_state.mode = "awaiting_report"   # awaiting_report / awaiting_city / awaiting_province

# render history
for role, content in st.session_state.messages:
    with st.chat_message(role):
        st.write(content)

# =========[ CHAT LOOP ]=========
user_text = st.chat_input("Tulis laporan Anda…")
if user_text:
    st.session_state.messages.append(("user", user_text))
    with st.chat_message("user"): st.write(user_text)

    mode = st.session_state.mode

    if mode == "awaiting_city":
    st.session_state.form["city"] = normalize_city_name(user_text)
    need = next_missing_slot(st.session_state.form)
    if need:
        st.session_state.mode = f"awaiting_{need}"
        q = ask_for(need)
        bot_msg = (
            f"Terima kasih. Lokasi sekarang: {format_location(st.session_state.form)}\n\n{q}"
        )
        st.session_state.messages.append(("assistant", bot_msg))
        with st.chat_message("assistant"): st.write(bot_msg)
        st.rerun()
    else:
        st.session_state.mode = "awaiting_report"
        label = st.session_state.pending["label"] if st.session_state.pending else st.session_state.form.get("category", "tidak_tahu")
        conf  = st.session_state.pending["conf"]  if st.session_state.pending else 0.5
        last_text = st.session_state.form["raw_texts"][-1] if st.session_state.form["raw_texts"] else user_text
        loc_struct = format_location(st.session_state.form)
        uprompt = build_user_prompt(last_text, label, conf, loc_struct)
        reply = llm_reply(SYSTEM_CHAT, uprompt)

        st.session_state.messages.append(("assistant", reply))
        with st.chat_message("assistant"):
            st.write(reply)
            st.button("✅ Kirim ke instansi", key=f"send_{len(st.session_state.messages)}")
        st.rerun()

    # ---- MODE: mengisi Province saja (tanpa klasifikasi) ----
    if mode == "awaiting_province":
        st.session_state.form["province"] = _title_keep_upper(_clean(user_text))
        st.session_state.mode = "awaiting_report"
        bot_msg = f"Terima kasih. Lokasi sekarang: {format_location(st.session_state.form)}"
        st.session_state.messages.append(("assistant", bot_msg))
        with st.chat_message("assistant"): st.write(bot_msg)
        st.rerun()

    # ---- MODE: laporan baru / normal (klasifikasi + NER) ----
    label, conf, _ = classify_text_ml(user_text, unknown_threshold=UNKNOWN_THRESHOLD)
    st.session_state.pending = {"label": label, "conf": conf, "text": user_text}

    # update form via NER
    st.session_state.form = update_form_from_text(user_text, st.session_state.form, detected_label=label)

    # cek slot yang masih kosong
    need = next_missing_slot(st.session_state.form)
    if need:
        st.session_state.mode = f"awaiting_{need}"
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
        loc_struct = format_location(st.session_state.form)
        uprompt    = build_user_prompt(user_text, label, conf, loc_struct)
        reply      = llm_reply(SYSTEM_CHAT, uprompt)

        st.session_state.messages.append(("assistant", reply))
        with st.chat_message("assistant"):
            st.write(reply)
            st.button("✅ Kirim ke instansi", key=f"send_{len(st.session_state.messages)}")

    st.rerun()

# =========[ SEND BUTTON HANDLER ]=========
clicked_keys = [k for k in st.session_state.keys() if str(k).startswith("send_")]
if clicked_keys and st.session_state.pending:
    t      = ticket_id()
    label  = st.session_state.pending["label"]
    conf   = st.session_state.pending["conf"]
    text_  = st.session_state.pending["text"]
    tujuan = AGENCY_MAP.get(label, AGENCY_MAP[UNKNOWN_LABEL])
    loc_struct = format_location(st.session_state.form)

    summary_prompt = build_ticket_prompt(label, conf, loc_struct, tujuan, st.session_state.form["raw_texts"])
    summary = llm_reply(SYSTEM_SUMMARY, summary_prompt)

    confirmation = (
      f"📨 Terkirim!\n"
      f"• Nomor tiket : {t}\n"
      f"• Tujuan      : {tujuan['nama']} ({tujuan['kontak']})\n"
      f"• Lokasi      : {loc_struct}\n"
      f"• Ringkasan   : {summary}"
    )
    st.session_state.messages.append(("assistant", confirmation))

    # reset untuk laporan berikutnya
    st.session_state.pending = None
    st.session_state.form = {"category": None, "street": None, "city": None, "province": None, "raw_texts": []}
    st.session_state.mode = "awaiting_report"
    st.rerun()
