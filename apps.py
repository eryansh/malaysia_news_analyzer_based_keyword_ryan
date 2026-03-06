import streamlit as st
import feedparser
from groq import Groq
import json
import urllib.parse
import re
from collections import Counter
import pandas as pd

# Setup Groq Client securely using Streamlit Secrets
try:
    API_KEY = st.secrets["GROQ_API_KEY"]
except KeyError:
    st.error("Missing API Key. Please set GROQ_API_KEY in .streamlit/secrets.toml")
    st.stop()

client = Groq(api_key=API_KEY)

# --- HELPER FUNCTIONS ---

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
def analyze_with_llm(titles, target_lang):
    # THE BULLETPROOF FIX: Use regex to strip all quotes (standard & smart) and replace forward slashes with spaces
    clean_titles = [re.sub(r'[\\"\'“”‘’]', ' ', t).replace('/', ' ') for t in titles]
    
    titles_string = "\n".join([f"{i+1}. {t}" for i, t in enumerate(clean_titles)])
    system_prompt = "You are a Strategic Data Scientist. You MUST output ONLY raw, valid JSON."
    top_n = min(15, len(clean_titles))
    
    user_prompt = f"""
    Based on these {len(clean_titles)} news titles:
    {titles_string}

    Please provide a deep analysis. 
    IMPORTANT: Use {target_lang} for all text fields.
    Respond ONLY with a valid JSON object using this EXACT structure:
    {{
        "individual_analysis": [
            {{"id": 1, "title": "News title here", "sentiment": "Positive/Negative/Neutral"}}
        ],
        "deep_summary": "A detailed 2-paragraph summary based on all news.",
        "categories": ["Category 1", "Category 2"],
        "strategic_actions": ["Action 1", "Action 2", "Action 3"],
        "dominant_vibe": "Overall main sentiment",
        "sentiment_percentage": {{"Positive": 40, "Negative": 30, "Neutral": 30}}
    }}
    
    Note: ONLY list the top {top_n} most impactful news in the individual_analysis array.
    """

    try:
        completion = client.chat.completions.create(
            model="llama-3.1-8b-instant",
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
    # Combine all titles into one lowercase string
    text = " ".join(titles).lower()
    # Extract all words using regular expressions
    words = re.findall(r'\b\w+\b', text)
    # Count frequencies
    word_counts = Counter(words)
    return word_counts.most_common(top_n)

# --- MAIN STREAMLIT APP ---

st.set_page_config(page_title="News Analysis", page_icon="📰", layout="wide")

# Header
st.markdown("<h1 style='text-align: center; color: #1a73e8;'>NEWS ANALYSIS BASED TOPIC KEYWORD</h1>", unsafe_allow_html=True)
st.divider()

# Controls
col1, col2, col3, col4 = st.columns([3, 2, 2, 2])

with col1:
    topic = st.text_input("Topik Analisis:", value="Sabah")
with col2:
    selected_lang = st.selectbox("Bahasa:", ["English", "Bahasa Malaysia", "Bahasa Indonesia"], index=1)
with col3:
    selected_limit = st.selectbox("Max Berita:", [10, 20, 50, 100], index=2)
with col4:
    st.markdown("<br>", unsafe_allow_html=True)
    analyze_btn = st.button("JANA ANALISIS", type="primary", use_container_width=True)

# Processing Logic
if analyze_btn:
    if not topic:
        st.warning("Please enter a topic!")
    else:
        with st.status(f"🔍 Processing '{topic}'...", expanded=True) as status:
            
            st.write(f"Fetching max {selected_limit} news articles in {selected_lang}...")
            titles = get_news(topic, selected_limit)
            
            if not titles:
                status.update(label="No news found.", state="error", expanded=True)
                st.error("❌ No news found.")
            else:
                st.write(f"🧠 AI is analyzing {len(titles)} articles to generate insights...")
                data = analyze_with_llm(titles, selected_lang)
                
                if "error" in data:
                    status.update(label="Analysis failed.", state="error", expanded=True)
                    st.error(f"❌ ERROR: {data['error']}")
                else:
                    status.update(label="Analysis complete!", state="complete", expanded=True)

                    # --- UI DISPLAY OF RESULTS ---
                    st.success(f"**Dominant Sentiment:** {data.get('dominant_vibe')}")
                    
                    # Sentiment Metrics
                    percents = data.get('sentiment_percentage', {})
                    met_col1, met_col2, met_col3 = st.columns(3)
                    met_col1.metric("🟢 Positive", f"{percents.get('Positive', 0)}%")
                    met_col2.metric("🔴 Negative", f"{percents.get('Negative', 0)}%")
                    met_col3.metric("⚪ Neutral", f"{percents.get('Neutral', 0)}%")
                    
                    st.divider()
                    
                    # Summary & Actions
                    st.subheader("📝 Executive Summary")
                    summary = data.get('deep_summary')
                    if isinstance(summary, dict):
                        summary = " ".join(summary.values())
                    st.write(summary)
                    
                    st.subheader("🚀 Strategic Actions")
                    actions = data.get('strategic_actions', [])
                    for i, action in enumerate(actions):
                        clean_action = action.get('action', action.get('text', str(action))) if isinstance(action, dict) else action
                        st.markdown(f"**{i+1}.** {clean_action}")
                    
                    st.markdown(f"**📂 Categories:** {', '.join(data.get('categories', []))}")
                    st.divider()

                    # Top 25 Words Table
                    # Top 25 Words Table
                    st.subheader("📊 Top 25 Keywords Frequency")
                    top_words = get_top_words(titles, 25)
                    if top_words:
                        df_words = pd.DataFrame(top_words, columns=["Keyword", "Frequency"])
                        
                        # ADD THIS LINE: Convert numbers to strings to force left-alignment
                        df_words["Frequency"] = df_words["Frequency"].astype(str) 
                        
                        st.dataframe(df_words, use_container_width=True, hide_index=True)

                    
                    
                    st.divider()

                    # Individual Sentiment List
                    st.subheader(f"🔍 Individual Sentiment Analysis (Top {min(15, len(titles))})")
                    for item in data.get('individual_analysis', []):
                        s = item.get('sentiment', 'Neutral')
                        icon = "🟢" if s in ["Positive", "Positif"] else "🔴" if s in ["Negative", "Negatif"] else "⚪"
                        st.markdown(f"{icon} **[{s}]** {item.get('title')}")




