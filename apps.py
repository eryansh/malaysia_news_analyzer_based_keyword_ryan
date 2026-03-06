import streamlit as st
import feedparser
from groq import Groq
import json
import urllib.parse
import re
from collections import Counter
import pandas as pd

# Setup Groq Client
try:
    API_KEY = st.secrets["GROQ_API_KEY"]
except KeyError:
    st.error("Kunci API Hilang. Sila tetapkan GROQ_API_KEY di .streamlit/secrets.toml")
    st.stop()

client = Groq(api_key=API_KEY)

# --- FUNGSI BANTUAN ---

@st.cache_data(show_spinner=False, ttl=3600)
def get_news(keyword, limit):
    clean_keyword = keyword.strip()
    if ' ' in clean_keyword and not (clean_keyword.startswith('"') and clean_keyword.endswith('"')):
        refined_keyword = f'"{clean_keyword}"'
    else:
        refined_keyword = clean_keyword

    encoded_keyword = urllib.parse.quote(refined_keyword)
    url = f"https://news.google.com/rss/search?q={encoded_keyword}&hl=ms&gl=MY&ceid=MY:ms"
    
    feed = feedparser.parse(url)
    return [entry.title for entry in feed.entries[:limit]] 

# Nota: Kita BUANG st.cache_data di sini supaya jika berlaku ralat, 
# ia tidak akan 'tersangkut' dan akan cuba menganalisis semula bila butang ditekan.
def analyze_with_llm(titles):
    clean_titles = [re.sub(r'[\\"\'“”‘’]', ' ', t).replace('/', ' ') for t in titles]
    titles_string = "\n".join([f"{i+1}. {t}" for i, t in enumerate(clean_titles)])
    top_n = min(15, len(clean_titles))
    
    # Arahan sistem dikemaskini untuk menghalang AI terjemah 'Keys' JSON
    system_prompt = """Anda adalah seorang Pakar Analisis Data. 
    AMARAN KERAS: 
    1. Anda MESTI membalas dengan format JSON yang sah.
    2. JANGAN TERJEMAH JSON KEYS. Kekalkan keys dalam Bahasa Inggeris ("individual_analysis", "deep_summary", dll).
    3. Nilai (Values) di dalam JSON mestilah dalam Bahasa Melayu yang profesional."""
    
    user_prompt = f"""
    Berdasarkan {len(clean_titles)} tajuk berita berikut:
    {titles_string}

    Berikan objek JSON dengan struktur TEPAT ini:
    {{
        "individual_analysis": [
            {{"id": 1, "sentiment": "Positif/Negatif/Neutral"}}
        ],
        "deep_summary": "Tulis satu laporan rumusan yang sangat terperinci dan panjang (minimum 5 perenggan, 800+ patah perkataan). Huraikan secara mendalam setiap isu dan kaitan antara berita-berita ini bergaya penceritaan (storytelling) yang sangat menarik, mengalir (flow) dengan baik, dan tidak berulang-ulang. Gunakan gaya bahasa kewartawanan atau majalah yang memikat pembaca, menceritakan jatuh bangun, cabaran, dan pencapaian berkaitan topik ini dalam 4 hingga 5 perenggan yang padat.",
        "categories": ["Kategori 1", "Kategori 2"],
        "strategic_actions": ["Tindakan 1", "Tindakan 2", "Tindakan 3"],
        "dominant_vibe": "Sentimen utama keseluruhan",
        "sentiment_percentage": {{"Positif": 40, "Negatif": 30, "Neutral": 30}}
    }}
    
    Nota PENTING: Untuk individual_analysis, senaraikan HANYA 'id' dan 'sentiment' untuk {top_n} berita pertama. JANGAN masukkan teks tajuk berita ke dalam JSON.
    """

    try:
        completion = client.chat.completions.create(
            model="llama-3.3-70b-versatile", # MODEL DITUKAR: Lebih bijak dan stabil
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            response_format={"type": "json_object"},
            temperature=0.3 
        )
        return json.loads(completion.choices[0].message.content)
    except Exception as e:
        return {"error": str(e)}

def get_top_words(titles, top_n=25):
    text = " ".join(titles).lower()
    # Hanya ambil perkataan yang panjangnya 3 huruf ke atas
    words = re.findall(r'\b[a-z]{3,}\b', text)
    
    # Penapis asas perkataan BM (Stopwords) supaya jadual lebih relevan
    stopwords = {'dan', 'yang', 'untuk', 'pada', 'dengan', 'dari', 'dalam', 'kepada', 'oleh', 'akan', 'ini', 'itu', 'ada', 'tak', 'nak', 'buat', 'apa', 'lagi', 'telah', 'tidak', 'iaitu'}
    words = [w for w in words if w not in stopwords]
    
    word_counts = Counter(words)
    return word_counts.most_common(top_n)

# --- APLIKASI UTAMA STREAMLIT ---

st.set_page_config(page_title="Analisis Berita", page_icon="📰", layout="wide")

st.markdown("<h1 style='text-align: center; color: #1a73e8;'>ANALISIS BERITA BERDASARKAN KATA KUNCI</h1>", unsafe_allow_html=True)
st.divider()

col1, col2, col3 = st.columns([4, 2, 2])

with col1:
    topic = st.text_input("Topik Analisis:", value="Sabah")
with col2:
    selected_limit = st.selectbox("Maksimum Berita:", [10, 20, 50, 100], index=2)
with col3:
    st.markdown("<br>", unsafe_allow_html=True)
    analyze_btn = st.button("JANA ANALISIS", type="primary", use_container_width=True)

if analyze_btn:
    if not topic:
        st.warning("Sila masukkan topik!")
    else:
        with st.status(f"🔍 Memproses '{topic}'...", expanded=True) as status:
            
            st.write(f"Mengambil maksimum {selected_limit} artikel berita...")
            titles = get_news(topic, selected_limit)
            
            if not titles:
                status.update(label="Tiada berita dijumpai.", state="error", expanded=True)
                st.error("❌ Tiada berita dijumpai untuk topik ini.")
            else:
                st.write(f"🧠 AI sedang menganalisis {len(titles)} artikel untuk menjana maklumat (mungkin mengambil masa 10-20 saat)...")
                data = analyze_with_llm(titles)
                
                if "error" in data:
                    status.update(label="Analisis gagal.", state="error", expanded=True)
                    st.error(f"❌ RALAT: {data['error']}")
                else:
                    status.update(label="Analisis selesai!", state="complete", expanded=True)

                    st.success(f"**Sentimen Dominan:** {data.get('dominant_vibe', 'Tidak Nyata')}")
                    
                    percents = data.get('sentiment_percentage', {})
                    met_col1, met_col2, met_col3 = st.columns(3)
                    met_col1.metric("🟢 Positif", f"{percents.get('Positif', percents.get('Positive', 0))}%")
                    met_col2.metric("🔴 Negatif", f"{percents.get('Negatif', percents.get('Negative', 0))}%")
                    met_col3.metric("⚪ Neutral", f"{percents.get('Neutral', 0)}%")
                    
                    st.divider()
                    
                    st.subheader("📝 Ringkasan Eksekutif")
                    summary = data.get('deep_summary', 'Tiada ringkasan dijana.')
                    if isinstance(summary, dict):
                        summary = " ".join(summary.values())
                    st.markdown(f"<div style='text-align: justify; line-height: 1.6;'>{summary}</div>", unsafe_allow_html=True)
                    
                    st.subheader("🚀 Tindakan Strategik")
                    actions = data.get('strategic_actions', [])
                    for i, action in enumerate(actions):
                        clean_action = action.get('action', action.get('text', str(action))) if isinstance(action, dict) else action
                        st.markdown(f"**{i+1}.** {clean_action}")
                    
                    st.markdown(f"**📂 Kategori Tumpuan:** {', '.join(data.get('categories', []))}")
                    st.divider()

                    st.subheader("📊 Kekerapan 25 Kata Kunci Teratas")
                    top_words = get_top_words(titles, 25)
                    if top_words:
                        df_words = pd.DataFrame(top_words, columns=["Perkataan", "Kekerapan"])
                        # Guna st.dataframe supaya UI tak kosong atau lari alignment
                        st.dataframe(df_words, use_container_width=True, hide_index=True)
                    
                    st.divider()

                    st.subheader(f"🔍 Analisis Sentimen Individu (Top {min(15, len(titles))})")
                    individual_data = data.get('individual_analysis', [])
                    if individual_data:
                        for item in individual_data:
                            idx = item.get('id', 1) - 1
                            if 0 <= idx < len(titles):
                                actual_title = titles[idx]
                            else:
                                continue
                                
                            s = item.get('sentiment', 'Neutral')
                            icon = "🟢" if s in ["Positif", "Positive"] else "🔴" if s in ["Negatif", "Negative"] else "⚪"
                            display_s = "Positif" if s in ["Positif", "Positive"] else "Negatif" if s in ["Negatif", "Negative"] else "Neutral"
                            
                            st.markdown(f"{icon} **[{display_s}]** {actual_title}")
                    else:
                        st.info("Tiada analisis individu dijana.")




