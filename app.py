import os, json, time, uuid, re
import numpy as np
import streamlit as st
import joblib, requests, httpx

st.set_page_config(page_title="Chatbot Laporan Publik", page_icon="🛠️")
st.title("🛠️ Chatbot Laporan Publik")

UNKNOWN_LABEL = "tidak_tahu"
THRESH = st.sidebar.slider("Ambang 'tidak_tahu'", 0.30, 0.90, 0.60, 0.01)

MODEL_PATH, LABELS_PATH = "models/text_cls_tfidf_lr.joblib", "models/text_labels.json"
OPENROUTER_MODEL, HF_MODEL = "google/gemma-2-9b-it", "mistralai/Mistral-7B-Instruct-v0.2"

AGENCY = {
  "jalan_berlubang":{"nama":"Dinas Bina Marga / PU","kontak":"bina-marga@pemda.go.id"},
  "lampu_jalan_mati":{"nama":"Dinas Perhubungan (PJU)","kontak":"dishub@pemda.go.id"},
  "rambu_rusak":{"nama":"Dinas Perhubungan (Lalu Lintas)","kontak":"dishub@pemda.go.id"},
  "sampah_menumpuk":{"nama":"DLH / Dinas Kebersihan","kontak":"dlh@pemda.go.id"},
  "tidak_tahu":{"nama":"Pusat Layanan Publik","kontak":"contact-center@pemda.go.id"},
}

@st.cache_resource(show_spinner=False)
def load_text_model():
    pipe = joblib.load(MODEL_PATH)
    with open(LABELS_PATH) as f: labels = json.load(f)
    return pipe, labels

pipe, LABELS = load_text_model()

def classify_text(txt: str, th=THRESH):
    p = pipe.predict_proba([txt])[0]
    i = int(np.argmax(p)); lab, conf = pipe.classes_[i], float(p[i])
    if lab != UNKNOWN_LABEL and conf < th: return UNKNOWN_LABEL, conf, dict(zip(pipe.classes_, p))
    return lab, conf, dict(zip(pipe.classes_, p))

def llm_openrouter(sys: str, usr: str):
    key = st.secrets.get("OPENROUTER_API_KEY"); 
    if not key: return None
    try:
        r = httpx.post("https://openrouter.ai/api/v1/chat/completions",
            headers={"Authorization":f"Bearer {key}","Content-Type":"application/json"},
            json={"model":OPENROUTER_MODEL,"messages":[{"role":"system","content":sys},{"role":"user","content":usr}],
                  "temperature":0.6,"max_tokens":220}, timeout=60)
        r.raise_for_status(); return r.json()["choices"][0]["message"]["content"].strip()
    except Exception as e:
        st.warning(f"[OpenRouter] {e}"); return None

def llm_hf(sys: str, usr: str):
    tok = st.secrets.get("HF_API_TOKEN"); 
    if not tok: return None
    try:
        r = requests.post(f"https://api-inference.huggingface.co/models/{HF_MODEL}",
            headers={"Authorization":f"Bearer {tok}"},
            json={"inputs":f"[SYSTEM]\n{sys}\n\n[USER]\n{usr}\n\n[ASSISTANT]\n",
                  "parameters":{"max_new_tokens":220,"temperature":0.6}}, timeout=60)
        r.raise_for_status(); data = r.json()
        if isinstance(data, list) and data and "generated_text" in data[0]:
            return data[0]["generated_text"].split("[ASSISTANT]")[-1].strip()
        if isinstance(data, dict) and "error" in data: st.warning(f"[HF] {data['error']}")
    except Exception as e:
        st.warning(f"[HF] {e}")
    return None

def llm_reply(sys: str, usr: str):
    return llm_openrouter(sys, usr) or llm_hf(sys, usr) or "Terima kasih. (LLM tidak aktif)"

SYSTEM_CHAT = ("Kamu asisten layanan publik Indonesia. Balas ringkas; sebutkan kategori, lokasi (jalan/kota/provinsi jika ada), "
               "serta instansi tujuan. Akhiri dengan: 'Kirim sekarang? (y/n)'.")
SYSTEM_SUMM = ("Buat ringkasan tiket 1–2 kalimat, sebutkan kategori dan lokasi (jalan/kota/provinsi). Jangan bertanya balik.")

# ==== NER loader (robust) ====
try:
    from transformers import AutoTokenizer, AutoModelForTokenClassification, pipeline
except Exception as e:
    st.error("Gagal import transformers. Pastikan requirements sesuai."); st.stop()

def _extract_fields_regex(text: str) -> dict:
    out, t = {}, " " + text.strip() + " "
    m = re.search(r'\b(?:jalan|jl|jln|gang|gg)\.?\s+(.+?)(?=,|\s+di\s|\s$)', t, flags=re.IGNORECASE)
    if m: out["street"] = re.sub(r"\s+"," ",m.group(1)).strip().title()
    m2 = re.search(r'\bdi\s+([A-Z][A-Za-z .\-]+)', text)
    if m2: out["city"] = re.sub(r"\s+"," ",m2.group(1)).strip().title()
    m3 = re.search(r'\bprovinsi\s+([A-Za-z .\-]+)', text, flags=re.IGNORECASE)
    if m3: out["province"] = re.sub(r"\s+"," ",m3.group(1)).strip().title()
    m4 = re.search(r'^(DI|Daerah Istimewa)\s+[A-Za-z .\-]+$', text.strip(), flags=re.IGNORECASE)
    if m4: out["province"] = text.strip().title()
    return out

@st.cache_resource(show_spinner=False)
def load_ner():
    mid = st.secrets.get("NER_MODEL", "cahya/bert-base-indonesian-NER")
    cache_dir = os.path.join(os.path.dirname(__file__), "hf_cache"); os.makedirs(cache_dir, exist_ok=True)
    kw = {"cache_dir":cache_dir}
    tok = AutoTokenizer.from_pretrained(mid, **kw)
    mdl = AutoModelForTokenClassification.from_pretrained(mid, **kw)
    return pipeline("ner", model=mdl, tokenizer=tok, aggregation_strategy="simple")

try:
    ner_pipe = load_ner()
except Exception as e:
    st.warning(f"NER offline, fallback regex. Detail: {e}")
    ner_pipe = None

STREET_PREFIXES = ("jalan","jl","jln","gang","gg")
def _clean(s:str): return re.sub(r"\s+"," ",s.strip(" ,.;:()[]{}"))
def _title_keep(s:str): return " ".join(w.upper() if w.upper() in {"RT","RW","GG","PJU"} else w.capitalize() for w in s.split())
def normalize_city_name(s:str):
    t=_clean(s); tl=t.lower()
    for p in ["kota","kabupaten","kab","kodya","kec","kecamatan"]:
        if tl.startswith(p+" "): t=t[len(p)+1:]; break
    return _title_keep(t)
def _looks_street(span:str): return any(span.strip().lower().startswith(p+" ") for p in STREET_PREFIXES)
def _looks_prov(span:str):
    t=_clean(span); 
    if len(t)>30: return False
    return bool(re.match(r"^(?i)provinsi\s+[A-Za-z][A-Za-z .\-]+$",t) or re.match(r"^(?i)(DI|Daerah Istimewa)\s+[A-Za-z][A-Za-z .\-]+$",t))
def _pick_city(text:str, spans:list[str]):
    tl=f" {text} ".lower()
    c=[s for s in spans if f" di {s.lower()}" in tl] or spans
    c=[x for x in c if not _looks_street(x) and not _looks_prov(x)]
    if not c: return None
    c.sort(key=len); return _title_keep(_clean(c[0]))

def extract_fields(text:str)->dict:
    if ner_pipe is None: return _extract_fields_regex(text)
    out={"street":None,"city":None,"province":None}
    locs=[e["word"].strip() for e in ner_pipe(text) if e.get("entity_group","")=="LOC"]
    if not locs: return _extract_fields_regex(text)
    stc=[s for s in locs if _looks_street(s)]
    if stc:
        stc.sort(key=len, reverse=True)
        sv=_title_keep(_clean(re.sub(r"\b(ada|banyak|gundukan|sampah|lubang|rusak|mati|menumpuk).*$","",stc[0],flags=re.IGNORECASE)))
        out["street"]=sv or None
    prov=[s for s in locs if _looks_prov(s)]
    if prov:
        prov.sort(key=len); out["province"]=_title_keep(_clean(prov[0]))
    rem=[]
    for s in locs:
        sc=_clean(s)
        if out.get("street") and sc.lower()==out["street"].lower(): continue
        if out.get("province") and sc.lower()==out["province"].lower(): continue
        rem.append(sc)
    city=_pick_city(text, rem)
    if city: out["city"]=normalize_city_name(city)
    return {k:v for k,v in out.items() if v} if any(out.values()) else _extract_fields_regex(text)

def format_loc(f):
    p=[]
    if f.get("street"): p.append(f"Jalan/Gang: {f['street']}")
    if f.get("city"): p.append(f"Kota/Kab: {f['city']}")
    if f.get("province"): p.append(f"Provinsi: {f['province']}")
    return " | ".join(p) if p else "(lokasi belum lengkap)"

def update_form(text, form, label=None):
    form["raw_texts"].append(text)
    if label: form["category"]=label
    fx=extract_fields(text)
    if fx.get("street"): form["street"]=fx["street"]
    if fx.get("city"): form["city"]=normalize_city_name(fx["city"])
    if fx.get("province"): form["province"]=_title_keep(_clean(fx["province"]))
    return form

def next_slot(f):
    for k in ["street","city","province"]:
        if not f.get(k): return k
    return None

def ask_for(s):
    return {"street":"Alamat jalannya apa ya? (contoh: Jalan Harummanis)",
            "city":"Ini di kota/kabupaten mana ya?",
            "province":"Provinsinya di mana ya?"}.get(s)

def build_user_prompt(txt, label, conf, loc):
    t=AGENCY.get(label, AGENCY[UNKNOWN_LABEL])
    return (f"Kategori terdeteksi: {label} (confidence {conf:.2f}). "
            f"Instansi: {t['nama']} | Kontak: {t['kontak']}. "
            f"Lokasi terstruktur: {loc}.\n\nTeks pengguna: {txt}")

def build_ticket_prompt(label, conf, loc, tujuan, raws):
    return (f"Kategori: {label} (confidence {conf:.2f}). Lokasi: {loc}. "
            f"Instansi tujuan: {tujuan['nama']} ({tujuan['kontak']}). "
            f"Teks pengguna: {' | '.join(raws)}\n\nBuat ringkasan tiket singkat 1–2 kalimat, tanpa bertanya balik.")

def ticket_id(): return "TCK-"+time.strftime("%Y%m%d")+"-"+uuid.uuid4().hex[:6].upper()

if "messages" not in st.session_state: st.session_state.messages=[]
if "pending"  not in st.session_state: st.session_state.pending=None
if "form"     not in st.session_state: st.session_state.form={"category":None,"street":None,"city":None,"province":None,"raw_texts":[]}
if "mode"     not in st.session_state: st.session_state.mode="awaiting_report"

for role,content in st.session_state.messages:
    with st.chat_message(role): st.write(content)

user_text = st.chat_input("Tulis laporan Anda…")
if user_text:
    st.session_state.messages.append(("user", user_text))
    with st.chat_message("user"): st.write(user_text)
    mode = st.session_state.mode

    if mode == "awaiting_city":
        st.session_state.form["city"] = normalize_city_name(user_text)
        need = next_slot(st.session_state.form)
        if need:
            st.session_state.mode = f"awaiting_{need}"
            msg = f"Terima kasih. Lokasi sekarang: {format_loc(st.session_state.form)}\n\n{ask_for(need)}"
            st.session_state.messages.append(("assistant", msg))
            with st.chat_message("assistant"): st.write(msg)
            st.rerun()
        else:
            st.session_state.mode = "awaiting_report"
            lab = st.session_state.pending["label"] if st.session_state.pending else (st.session_state.form.get("category") or UNKNOWN_LABEL)
            cf  = st.session_state.pending["conf"]  if st.session_state.pending else 0.5
            last = st.session_state.form["raw_texts"][-1] if st.session_state.form["raw_texts"] else user_text
            loc = format_loc(st.session_state.form)
            rep = llm_reply(SYSTEM_CHAT, build_user_prompt(last, lab, cf, loc))
            st.session_state.messages.append(("assistant", rep))
            with st.chat_message("assistant"):
                st.write(rep); st.button("✅ Kirim ke instansi", key=f"send_{len(st.session_state.messages)}")
            st.rerun()

    if mode == "awaiting_province":
        st.session_state.form["province"] = _title_keep(_clean(user_text))
        need = next_slot(st.session_state.form)
        if need:
            st.session_state.mode = f"awaiting_{need}"
            msg = f"Terima kasih. Lokasi sekarang: {format_loc(st.session_state.form)}\n\n{ask_for(need)}"
            st.session_state.messages.append(("assistant", msg))
            with st.chat_message("assistant"): st.write(msg)
            st.rerun()
        else:
            st.session_state.mode = "awaiting_report"
            lab = st.session_state.pending["label"] if st.session_state.pending else (st.session_state.form.get("category") or UNKNOWN_LABEL)
            cf  = st.session_state.pending["conf"]  if st.session_state.pending else 0.5
            last = st.session_state.form["raw_texts"][-1] if st.session_state.form["raw_texts"] else user_text
            loc = format_loc(st.session_state.form)
            rep = llm_reply(SYSTEM_CHAT, build_user_prompt(last, lab, cf, loc))
            st.session_state.messages.append(("assistant", rep))
            with st.chat_message("assistant"):
                st.write(rep); st.button("✅ Kirim ke instansi", key=f"send_{len(st.session_state.messages)}")
            st.rerun()

    label, conf, _ = classify_text(user_text, THRESH)
    st.session_state.pending = {"label":label,"conf":conf,"text":user_text}
    st.session_state.form = update_form(user_text, st.session_state.form, label)

    need = next_slot(st.session_state.form)
    if need:
        st.session_state.mode = f"awaiting_{need}"
        msg = (f"Kategori terdeteksi: **{label}** (conf {conf:.2f})\n"
               f"Lokasi saat ini: {format_loc(st.session_state.form)}\n\n{ask_for(need)}")
        st.session_state.messages.append(("assistant", msg))
        with st.chat_message("assistant"): st.write(msg)
    else:
        st.session_state.mode = "awaiting_report"
        loc = format_loc(st.session_state.form)
        rep = llm_reply(SYSTEM_CHAT, build_user_prompt(user_text, label, conf, loc))
        st.session_state.messages.append(("assistant", rep))
        with st.chat_message("assistant"):
            st.write(rep); st.button("✅ Kirim ke instansi", key=f"send_{len(st.session_state.messages)}")
    st.rerun()

clicked = [k for k in st.session_state.keys() if str(k).startswith("send_")]
if clicked and st.session_state.pending:
    t = ticket_id()
    lab = st.session_state.pending["label"]; cf = st.session_state.pending["conf"]
    tujuan = AGENCY.get(lab, AGENCY[UNKNOWN_LABEL])
    loc = format_loc(st.session_state.form)
    summ = llm_reply(SYSTEM_SUMM, build_ticket_prompt(lab, cf, loc, tujuan, st.session_state.form["raw_texts"]))
    confirm = (f"📨 Terkirim!\n• Nomor tiket : {t}\n• Tujuan      : {tujuan['nama']} ({tujuan['kontak']})\n"
               f"• Lokasi      : {loc}\n• Ringkasan   : {summ}")
    st.session_state.messages.append(("assistant", confirm))
    st.session_state.pending=None
    st.session_state.form={"category":None,"street":None,"city":None,"province":None,"raw_texts":[]}
    st.session_state.mode="awaiting_report"
    st.rerun()
