import nltk
nltk.download('punkt')
nltk.download('punkt_tab')

import streamlit as st
import pickle 
import string
from nltk.corpus import stopwords
import nltk
from nltk.stem.porter import PorterStemmer
import time

# Custom CSS for advanced styling
st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Poppins:wght@300;400;600&display=swap');
    
    .main {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
    }
    
    .stTextInput>div>div>input {
        background-color: rgba(255, 255, 255, 0.9);
        border-radius: 10px;
        padding: 10px;
        font-family: 'Poppins', sans-serif;
    }
    
    .stButton>button {
        background: linear-gradient(45deg, #FF6B6B, #FF8E8E);
        color: white;
        border-radius: 25px;
        padding: 12px 30px;
        font-weight: bold;
        font-family: 'Poppins', sans-serif;
        border: none;
        box-shadow: 0 4px 15px rgba(255, 107, 107, 0.4);
        transition: all 0.3s ease;
    }
    
    .stButton>button:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 20px rgba(255, 107, 107, 0.6);
    }
    
    .result-box {
        padding: 25px;
        border-radius: 15px;
        margin: 20px 0;
        text-align: center;
        animation: fadeIn 0.5s ease-in;
        box-shadow: 0 4px 15px rgba(0, 0, 0, 0.1);
    }
    
    .spam {
        background: linear-gradient(45deg, #ff416c, #ff4b2b);
        color: white;
    }
    
    .not-spam {
        background: linear-gradient(45deg, #00b09b, #96c93d);
        color: white;
    }
    
    .header {
        text-align: center;
        padding: 20px 0;
        animation: slideDown 0.5s ease-out;
    }
    
    .description {
        background: rgba(255, 255, 255, 0.1);
        padding: 20px;
        border-radius: 15px;
        margin: 20px 0;
        backdrop-filter: blur(10px);
    }
    
    @keyframes fadeIn {
        from { opacity: 0; transform: translateY(20px); }
        to { opacity: 1; transform: translateY(0); }
    }
    
    @keyframes slideDown {
        from { transform: translateY(-50px); opacity: 0; }
        to { transform: translateY(0); opacity: 1; }
    }
    
    .stats {
        display: flex;
        justify-content: space-around;
        margin: 20px 0;
        padding: 15px;
        background: rgba(255, 255, 255, 0.1);
        border-radius: 15px;
    }
    
    .stat-item {
        text-align: center;
    }
    
    .footer {
        text-align: center;
        padding: 20px;
        margin-top: 40px;
        font-size: 14px;
        color: rgba(255, 255, 255, 0.7);
    }
    </style>
    """, unsafe_allow_html=True)

def transform_text(text):
    text=text.lower()
    text=nltk.word_tokenize(text)
    y=[]
    for i in text:
        if(i.isalnum()):
            y.append(i)
    text=y[:]
    y.clear()
    for i in text:
        if(i not in stopwords.words("english") and i not in string.punctuation):
         y.append(i) 
    text=y[:]
    y.clear()
    stemmer=PorterStemmer()
    for i in text:
        y.append(stemmer.stem(i))
    return " ".join(y)

# Load the model and vectorizer
tfidf=pickle.load(open('vectorizer.pkl','rb'))
model=pickle.load(open('model.pkl','rb'))

# Main layout
st.markdown("""
    <div class="header">
        <h1 style="font-size: 2.5em; margin-bottom: 10px;">📱 SMS Spam Classifier</h1>
        <p style="font-size: 1.2em; opacity: 0.8;">Advanced Machine Learning for Message Analysis</p>
    </div>
    """, unsafe_allow_html=True)

# Add a description
st.markdown("""
    <div class="description">
        <p style="font-size: 16px; text-align: center;">
            This advanced application uses state-of-the-art machine learning algorithms to analyze and classify SMS messages.
            Enter your message below to determine if it's spam or legitimate.
        </p>
    </div>
    """, unsafe_allow_html=True)

# Stats section
st.markdown("""
    <div class="stats">
        <div class="stat-item">
            <h3>🔒 Secure</h3>
            <p>Real-time Analysis</p>
        </div>
        <div class="stat-item">
            <h3>⚡ Fast</h3>
            <p>Instant Results</p>
        </div>
        <div class="stat-item">
            <h3>🎯 Accurate</h3>
            <p>ML-Powered</p>
        </div>
    </div>
    """, unsafe_allow_html=True)

# Input section
st.markdown("### 📝 Enter your message")
input_sms = st.text_area("", height=150, placeholder="Type or paste your SMS message here...")

# Prediction button
col1, col2, col3 = st.columns([1,2,1])
with col2:
    predict_button = st.button('🔍 Analyze Message')

# Result display
if predict_button:
    if input_sms.strip() == "":
        st.warning("Please enter a message to analyze.")
    else:
        with st.spinner('Analyzing message...'):
            time.sleep(1)  # Simulate processing time
            transformed_sms = transform_text(input_sms)
            vector_input = tfidf.transform([transformed_sms])
            result = model.predict(vector_input)[0]

            if result == 1:
                st.markdown("""
                    <div class='result-box spam'>
                        <h2>⚠️ Spam Detected</h2>
                        <p>This message has been classified as spam with high confidence.</p>
                        <p style="font-size: 0.9em; opacity: 0.8;">Please exercise caution with this message.</p>
                    </div>
                    """, unsafe_allow_html=True)
            else:
                st.markdown("""
                    <div class='result-box not-spam'>
                        <h2>✅ Legitimate Message</h2>
                        <p>This message appears to be safe and legitimate.</p>
                        <p style="font-size: 0.9em; opacity: 0.8;">No suspicious content detected.</p>
                    </div>
                    """, unsafe_allow_html=True)

# Footer
st.markdown("""
    <div class="footer">
        <p>Built with ❤️ using Streamlit and Machine Learning</p>
        <p style="font-size: 0.8em;">© 2024 Advanced SMS Classifier</p>
    </div>
    """, unsafe_allow_html=True)



