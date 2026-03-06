import streamlit as st
import feedparser
from groq import Groq
import json
import urllib.parse
import re
from collections import Counter
import pandas as pd

# Setup Groq Client menggunakan Streamlit Secrets
try:
    API_KEY = st.secrets["GROQ_API_KEY"]
except KeyError:
    st.error("Kunci API Hilang. Sila tetapkan GROQ_API_KEY di .streamlit/secrets.toml")
    st.stop()

client = Groq(api_key=API_KEY)

# --- FUNGSI BANTUAN ---

@st.cache_data(show_spinner=False)
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

@st.cache_data(show_spinner=False)
def analyze_with_llm(titles):
    # Membersihkan tajuk: buang quote, ganti slash dengan space
    clean_titles = [re.sub(r'[\\"\'“”‘’]', ' ', t).replace('/', ' ') for t in titles]
    
    titles_string = "\n".join([f"{i+1}. {t}" for i, t in enumerate(clean_titles)])
    system_prompt = "Anda adalah seorang Pakar Analisis Data Strategik. Anda MESTI mengeluarkan HANYA format JSON mentah yang sah."
    top_n = min(15, len(clean_titles))
    
    # PROMPT DIUBAH: 100% Bahasa Melayu untuk elak LLM keliru
    user_prompt = f"""
    Berdasarkan {len(clean_titles)} tajuk berita berikut:
    {titles_string}

    Sila berikan analisis mendalam. 
    PENTING: Gunakan Bahasa Melayu sepenuhnya untuk semua teks yang dijanakan.
    Balas HANYA dengan objek JSON yang sah menggunakan struktur TEPAT ini:
    {{
        "individual_analysis": [
            {{"id": 1, "sentiment": "Positif/Negatif/Neutral"}}
        ],
        "deep_summary": "Tulis satu ringkasan esei yang sangat terperinci dan menyeluruh (minimum 5 perenggan yang panjang, sasaran 800+ patah perkataan). Huraikan secara mendalam setiap isu, trend, dan naratif yang terdapat dalam berita. Jangan tulis terlalu ringkas.",
        "categories": ["Kategori 1", "Kategori 2"],
        "strategic_actions": ["Tindakan 1", "Tindakan 2", "Tindakan 3"],
        "dominant_vibe": "Sentimen utama keseluruhan",
        "sentiment_percentage": {{"Positif": 40, "Negatif": 30, "Neutral": 30}}
    }}
    
    Nota: Untuk individual_analysis, HANYA keluarkan 'id' (integer) dan 'sentiment'. JANGAN keluarkan teks tajuk. Berikan ID untuk {top_n} berita teratas.
    """

    try:
        completion = client.chat.completions.create(
            model="llama-3.1-8b-instant",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            response_format={"type": "json_object"},
            temperature=0.4 
        )
        return json.loads(completion.choices[0].message.content)
    except Exception as e:
        return {"error": str(e)}

def get_top_words(titles, top_n=25):
    text = " ".join(titles).lower()
    # Hanya ambil huruf (termasuk huruf rumi Melayu), abaikan nombor dan simbol
    # Tidak menggunakan pembuangan 'stopword' bagi mengekalkan struktur teks asal
    words = re.findall(r'\b[a-z]+\b', text)
    word_counts = Counter(words)
    return word_counts.most_common(top_n)

# --- APLIKASI UTAMA STREAMLIT ---

st.set_page_config(page_title="Analisis Berita", page_icon="📰", layout="wide")

st.markdown("<h1 style='text-align: center; color: #1a73e8;'>ANALISIS BERITA BERDASARKAN KATA KUNCI</h1>", unsafe_allow_html=True)
st.divider()

# Layout diperkemas kerana dropdown bahasa telah dibuang
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
                st.write(f"🧠 AI sedang menganalisis {len(titles)} artikel untuk menjana maklumat...")
                data = analyze_with_llm(titles)
                
                if "error" in data:
                    status.update(label="Analisis gagal.", state="error", expanded=True)
                    st.error(f"❌ RALAT: {data['error']}")
                else:
                    status.update(label="Analisis selesai!", state="complete", expanded=True)

                    st.success(f"**Sentimen Dominan:** {data.get('dominant_vibe')}")
                    
                    percents = data.get('sentiment_percentage', {})
                    met_col1, met_col2, met_col3 = st.columns(3)
                    met_col1.metric("🟢 Positif", f"{percents.get('Positif', percents.get('Positive', 0))}%")
                    met_col2.metric("🔴 Negatif", f"{percents.get('Negatif', percents.get('Negative', 0))}%")
                    met_col3.metric("⚪ Neutral", f"{percents.get('Neutral', 0)}%")
                    
                    st.divider()
                    
                    st.subheader("📝 Ringkasan Eksekutif")
                    summary = data.get('deep_summary')
                    if isinstance(summary, dict):
                        summary = " ".join(summary.values())
                    st.write(summary)
                    
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
                        df_words.index = df_words.index + 1
                        st.table(df_words)
                    
                    st.divider()

                    st.subheader(f"🔍 Analisis Sentimen Individu (Top {min(15, len(titles))})")
                    for item in data.get('individual_analysis', []):
                        idx = item.get('id', 1) - 1
                        
                        if 0 <= idx < len(titles):
                            actual_title = titles[idx]
                        else:
                            continue
                            
                        s = item.get('sentiment', 'Neutral')
                        icon = "🟢" if s in ["Positif", "Positive"] else "🔴" if s in ["Negatif", "Negative"] else "⚪"
                        
                        # Papar dalam BM
                        display_s = "Positif" if s in ["Positif", "Positive"] else "Negatif" if s in ["Negatif", "Negative"] else "Neutral"
                        
                        st.markdown(f"{icon} **[{display_s}]** {actual_title}")
