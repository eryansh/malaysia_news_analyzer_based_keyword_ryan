import streamlit as st
import feedparser
from groq import Groq
import json
import urllib.parse
import matplotlib.pyplot as plt
from wordcloud import WordCloud

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
    # Get news up to the user's limit
    return [entry.title for entry in feed.entries[:limit]] 

@st.cache_data(show_spinner=False)
def analyze_with_llm(titles, target_lang):
    titles_string = "\n".join([f"{i+1}. {t}" for i, t in enumerate(titles)])
    system_prompt = "You are a Strategic Data Scientist. Output ONLY raw JSON."
    top_n = min(15, len(titles))
    
    user_prompt = f"""
    Based on these {len(titles)} news titles:
    {titles_string}

    Please provide a deep analysis in JSON format. 
    IMPORTANT: Use {target_lang} for all text fields.

    Structure:
    1. individual_analysis: [{{"id": 1, "title": "...", "sentiment": "Positive/Negative/Neutral"}}] -> ONLY list the top {top_n} most impactful news here.
    2. deep_summary: "A detailed 2-paragraph summary based on all {len(titles)} news."
    3. categories: ["List of news categories"]
    4. strategic_actions: ["3 concrete action recommendations"]
    5. dominant_vibe: "Overall main sentiment"
    6. sentiment_percentage: {{"Positive": 40, "Negative": 30, "Neutral": 30}} -> Provide integer percentages summing to 100.
    """

    try:
        completion = client.chat.completions.create(
            model="llama-3.1-8b-instant",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            response_format={"type": "json_object"}
        )
        return json.loads(completion.choices[0].message.content)
    except Exception as e:
        return {"error": str(e)}

def show_charts(titles, sentiment_percentages):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    # Pie Chart
    labels = list(sentiment_percentages.keys())
    sizes = list(sentiment_percentages.values())
    colors = ['#4CAF50', '#F44336', '#9E9E9E'] 
    ax1.pie(sizes, labels=labels, autopct='%1.1f%%', startangle=140, colors=colors)
    ax1.set_title(f'Sentiment Distribution ({len(titles)} News)')

    # Word Cloud
    all_text = " ".join(titles)
    wordcloud = WordCloud(width=500, height=400, background_color='white', colormap='ocean').generate(all_text)
    ax2.imshow(wordcloud, interpolation='bilinear')
    ax2.axis("off") 
    ax2.set_title('Top Keywords (Word Cloud)')

    plt.tight_layout()
    st.pyplot(fig)

# --- MAIN STREAMLIT APP ---

st.set_page_config(page_title="News Analysis", page_icon="📰", layout="wide")

# Header
st.markdown("<h1 style='text-align: center; color: #1a73e8;'>NEWS ANALYSIS BASED TOPIC KEYWORD</h1>", unsafe_allow_html=True)
st.divider()

# Controls (Using columns for a clean layout)
col1, col2, col3, col4 = st.columns([3, 2, 2, 2])

with col1:
    topic = st.text_input("Topik Analisis:", value="Sabah")
with col2:
    selected_lang = st.selectbox("Bahasa:", ["English", "Bahasa Malaysia", "Bahasa Indonesia"], index=1)
with col3:
    selected_limit = st.selectbox("Max Berita:", [10, 20, 50, 100], index=2)
with col4:
    st.markdown("<br>", unsafe_allow_html=True) # Push button down to align with inputs
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
                status.update(label="No news found.", state="error", expanded=False)
                st.error("❌ No news found.")
            else:
                st.write(f"🧠 AI is analyzing {len(titles)} articles to generate insights...")
                data = analyze_with_llm(titles, selected_lang)
                
                if "error" in data:
                    status.update(label="Analysis failed.", state="error", expanded=False)
                    st.error(f"❌ ERROR: {data['error']}")
                else:
                    status.update(label="Analysis complete!", state="complete", expanded=False)

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

                    # Charts
                    st.subheader("📊 Visualizations")
                    show_charts(titles, data.get('sentiment_percentage', {"Positive": 33, "Negative": 33, "Neutral": 34}))
                    
                    # Individual Sentiment List
                    with st.expander(f"🔍 View Individual Sentiment Analysis (Top {min(15, len(titles))})"):
                        for item in data.get('individual_analysis', []):
                            s = item.get('sentiment', 'Neutral')
                            icon = "🟢" if s in ["Positive", "Positif"] else "🔴" if s in ["Negative", "Negatif"] else "⚪"
                            st.markdown(f"{icon} **[{s}]** {item.get('title')}")