import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud
import shap
import joblib
import pickle
from urllib.parse import urlparse
import re
from tld import get_tld
import plotly.express as px
from datetime import datetime
import base64
from io import BytesIO

# Custom CSS for improved UI
def local_css(file_name):
    with open(file_name) as f:
        st.markdown(f'<style>{f.read()}</style>', unsafe_allow_html=True)

# Load your pre-trained model
@st.cache_resource
def load_model():
    try:
        model = joblib.load('random_forest_model.pkl')
        return model
    except:
        st.error("Model file not found. Please ensure 'random_forest_model.pkl' is in your directory.")
        return None

# Feature extraction functions (remain unchanged)
def having_ip_address(url):
    match = re.search(
        '(([01]?\\d\\d?|2[0-4]\\d|25[0-5])\\.([01]?\\d\\d?|2[0-4]\\d|25[0-5])\\.([01]?\\d\\d?|2[0-4]\\d|25[0-5])\\.'
        '([01]?\\d\\d?|2[0-4]\\d|25[0-5])\\/)|'
        '((0x[0-9a-fA-F]{1,2})\\.(0x[0-9a-fA-F]{1,2})\\.(0x[0-9a-fA-F]{1,2})\\.(0x[0-9a-fA-F]{1,2})\\/)'
        '(?:[a-fA-F0-9]{1,4}:){7}[a-fA-F0-9]{1,4}', url)
    return 1 if match else 0

def abnormal_url(url):
    hostname = urlparse(url).hostname
    hostname = str(hostname)
    match = re.search(hostname, url)
    return 1 if match else 0

def count_dot(url):
    return url.count('.')

def count_www(url):
    return url.count('www')

def count_atrate(url):
    return url.count('@')

def no_of_dir(url):
    urldir = urlparse(url).path
    return urldir.count('/')

def no_of_embed(url):
    urldir = urlparse(url).path
    return urldir.count('//')

def shortening_service(url):
    match = re.search('bit\.ly|goo\.gl|shorte\.st|go2l\.ink|x\.co|ow\.ly|t\.co|tinyurl|tr\.im|is\.gd|cli\.gs|'
                      'yfrog\.com|migre\.me|ff\.im|tiny\.cc|url4\.eu|twit\.ac|su\.pr|twurl\.nl|snipurl\.com|'
                      'short\.to|BudURL\.com|ping\.fm|post\.ly|Just\.as|bkite\.com|snipr\.com|fic\.kr|loopt\.us|'
                      'doiop\.com|short\.ie|kl\.am|wp\.me|rubyurl\.com|om\.ly|to\.ly|bit\.do|t\.co|lnkd\.in|'
                      'db\.tt|qr\.ae|adf\.ly|goo\.gl|bitly\.com|cur\.lv|tinyurl\.com|ow\.ly|bit\.ly|ity\.im|'
                      'q\.gs|is\.gd|po\.st|bc\.vc|twitthis\.com|u\.to|j\.mp|buzurl\.com|cutt\.us|u\.bb|yourls\.org|'
                      'x\.co|prettylinkpro\.com|scrnch\.me|filoops\.info|vzturl\.com|qr\.net|1url\.com|tweez\.me|v\.gd|'
                      'tr\.im|link\.zip\.net',
                      url)
    return 1 if match else 0

def count_https(url):
    return url.count('https')

def count_http(url):
    return url.count('http')

def count_per(url):
    return url.count('%')

def count_ques(url):
    return url.count('?')

def count_hyphen(url):
    return url.count('-')

def count_equal(url):
    return url.count('=')

def url_length(url):
    return len(str(url))

def hostname_length(url):
    return len(urlparse(url).netloc)

def suspicious_words(url):
    match = re.search('PayPal|login|signin|bank|account|update|free|lucky|service|bonus|ebayisapi|webscr', url)
    return 1 if match else 0

def digit_count(url):
    return sum(1 for i in url if i.isnumeric())

def letter_count(url):
    return sum(1 for i in url if i.isalpha())

def fd_length(url):
    urlpath = urlparse(url).path
    try:
        return len(urlpath.split('/')[1])
    except:
        return 0

def tld_length(tld):
    try:
        return len(tld)
    except:
        return -1

def extract_features(url):
    features = []
    
    features.append(having_ip_address(url))
    features.append(abnormal_url(url))
    features.append(count_dot(url))
    features.append(count_www(url))
    features.append(count_atrate(url))
    features.append(no_of_dir(url))
    features.append(no_of_embed(url))
    features.append(shortening_service(url))
    features.append(count_https(url))
    features.append(count_http(url))
    features.append(count_per(url))
    features.append(count_ques(url))
    features.append(count_hyphen(url))
    features.append(count_equal(url))
    features.append(url_length(url))
    features.append(hostname_length(url))
    features.append(suspicious_words(url))
    features.append(digit_count(url))
    features.append(letter_count(url))
    features.append(fd_length(url))
    
    tld = get_tld(url, fail_silently=True)
    features.append(tld_length(tld))
    
    return features

def explain_with_shap(model, data, feature_names):
    try:
        if not isinstance(data, np.ndarray):
            data = np.array(data)
        if data.ndim == 1:
            data = data.reshape(1, -1)
        
        if len(feature_names) != data.shape[1]:
            st.error(f"Feature count mismatch: Expected {data.shape[1]} features, got {len(feature_names)}")
            return
        
        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(data)
        
        st.write(f"SHAP values shape: {np.shape(shap_values)}")
        
        st.subheader("Feature Importance Summary")
        fig, ax = plt.subplots(figsize=(10, 6))
        plt.rcParams['font.size'] = 10
        
        if isinstance(shap_values, np.ndarray) and len(shap_values.shape) == 3:
            shap_values_mean = np.abs(shap_values).mean(axis=2)
            shap.summary_plot(shap_values_mean, data, feature_names=feature_names, plot_type="bar", show=False)
        else:
            shap.summary_plot(shap_values, data, feature_names=feature_names, plot_type="bar", show=False)
        
        st.pyplot(fig)
        plt.close(fig)
        
        st.subheader("Prediction Explanation")
        shap.initjs()
        
        pred_probs = model.predict_proba(data)[0]
        class_idx = np.argmax(pred_probs)
        class_names = ['Safe', 'Defacement', 'Phishing', 'Malware']
        
        if isinstance(shap_values, np.ndarray) and len(shap_values.shape) == 3:
            shap_values_class = shap_values[:, :, class_idx]
            expected_value = explainer.expected_value[class_idx]
        else:
            shap_values_class = shap_values
            expected_value = explainer.expected_value
        
        force_plot = shap.force_plot(
            expected_value,
            shap_values_class[0],
            data[0],
            feature_names=feature_names,
            matplotlib=True,
            show=False
        )
        
        st.pyplot(force_plot)
        plt.close()
        
        st.subheader("Detailed Feature Impact")
        fig, ax = plt.subplots(figsize=(10, 6))
        if isinstance(shap_values, np.ndarray) and len(shap_values.shape) == 3:
            shap.summary_plot(shap_values[:, :, class_idx], data, feature_names=feature_names, show=False)
        else:
            shap.summary_plot(shap_values, data, feature_names=feature_names, show=False)
        st.pyplot(fig)
        plt.close(fig)
        
        st.subheader("SHAP Values for Predicted Class")
        shap_df = pd.DataFrame({
            'Feature': feature_names,
            'SHAP Value': shap_values_class[0]
        })
        shap_df = shap_df.sort_values(by='SHAP Value', key=abs, ascending=False)
        st.dataframe(shap_df.style.background_gradient(cmap='RdBu', subset=['SHAP Value']))
        
    except Exception as e:
        st.error(f"Could not generate SHAP explanation: {str(e)}")
        st.write("Debug Info:")
        st.write(f"Data shape: {data.shape}")
        st.write(f"Feature names: {feature_names}")
        st.write(f"SHAP values type: {type(shap_values)}")
        st.write(f"SHAP values shape: {np.shape(shap_values)}")

def generate_pdf_report(url, prediction, features, shap_values):
    report = f"""
    URL Threat Analysis Report
    =========================
    
    Analyzed URL: {url}
    Date: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}
    
    Prediction: {prediction}
    Confidence: {np.max(shap_values):.2f}
    
    Key Indicators:
    - Suspicious Words: {'Detected' if features[16] == 1 else 'Not detected'}
    - Uses IP Address: {'Yes' if features[0] == 1 else 'No'}
    - URL Length: {features[14]} characters
    - Number of Directories: {features[5]}
    
    Recommendations:
    - {'This URL appears malicious. Avoid visiting.' if prediction != 'SAFE' else 'This URL appears safe.'}
    - {'Enable additional security measures if you must visit this site.' if prediction != 'SAFE' else 'No special precautions needed.'}
    """
    return report

def main():
    # Custom CSS and page config
    st.set_page_config(
        page_title="URL Guardian",
        page_icon="🛡️",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    # Load custom CSS
    local_css("styles.css")  # You'll need to create this file
    
    # Load model
    model = load_model()
    if model is None:
        return
    
    # Feature names
    feature_names = ['use_of_ip', 'abnormal_url', 'count.', 'count-www', 'count@',
                    'count_dir', 'count_embed_domian', 'short_url', 'count-https',
                    'count-http', 'count%', 'count?', 'count-', 'count=', 'url_length',
                    'hostname_length', 'sus_url', 'fd_length', 'tld_length', 'count-digits',
                    'count-letters']
    
    # Sidebar with improved styling
    with st.sidebar:
        st.markdown("""
        <div style="text-align:center; margin-bottom:30px;">
            <h1 style="color:#4a8cff; font-size:28px;">URL Guardian</h1>
            <p style="color:#666; font-size:14px;">Advanced web threat detection with explainable AI</p>
        </div>
        """, unsafe_allow_html=True)
        
        st.image("https://cdn-icons-png.flaticon.com/512/1271/1271847.png", width=150, output_format="PNG")
        
        menu = st.selectbox("Navigation", ["Threat Detection", "Threat Analysis Dashboard", "Historical Reports"], 
                          format_func=lambda x: f"📌 {x}" if x == "Threat Detection" else f"📊 {x}" if x == "Threat Analysis Dashboard" else f"📜 {x}")
        
        st.markdown("---")
        st.markdown("""
        <div style="background-color:#f0f2f6; padding:10px; border-radius:5px;">
            <h3 style="color:#4a8cff; margin-top:0;">Settings</h3>
        </div>
        """, unsafe_allow_html=True)
        
        show_shap = st.checkbox("Show detailed explanations", True)
        show_technical = st.checkbox("Show technical details", False)
        
        st.markdown("---")
        st.markdown("""
        <div style="text-align:center; margin-top:30px; color:#666; font-size:12px;">
            <p>URL Guardian v1.0</p>
            <p>© 2023 Security Analytics</p>
        </div>
        """, unsafe_allow_html=True)
    
    if menu == "Threat Detection":
        st.markdown("""
        <div style="background-color:#4a8cff; padding:20px; border-radius:10px; margin-bottom:30px;">
            <h1 style="color:white; margin:0;">🛡️ URL Threat Detection</h1>
            <p style="color:white; margin:0;">Analyze any URL for potential phishing, malware, defacement, or other threats.</p>
        </div>
        """, unsafe_allow_html=True)
        
        col1, col2 = st.columns([3, 1])
        
        with col1:
            url_input = st.text_input("Enter URL to analyze:", placeholder="https://example.com",
                                     help="Enter the full URL including http:// or https://")
            
        with col2:
            st.markdown("<div style='height:28px'></div>", unsafe_allow_html=True)
            analyze_btn = st.button("Analyze URL", type="primary", use_container_width=True)
        
        if analyze_btn and url_input:
            try:
                with st.spinner("🔍 Analyzing URL features..."):
                    features = extract_features(url_input)
                    features_array = np.array(features).reshape(1, -1)
                    
                    # Predict
                    prediction = model.predict(features_array)[0]
                    pred_proba = model.predict_proba(features_array)[0]
                    
                    # Map prediction code to label
                    pred_map = {0: "SAFE", 1: "DEFACEMENT", 2: "PHISHING", 3: "MALWARE"}
                    pred_label = pred_map.get(prediction, "UNKNOWN")
                    
                    # Display results with improved styling
                    st.markdown("---")
                    st.subheader("Analysis Results")
                    
                    # Result card with better styling
                    if pred_label == "SAFE":
                        st.markdown(f"""
                        <div style="background-color:#e6f7e6; padding:20px; border-radius:10px; border-left:5px solid #4CAF50;">
                            <h2 style="color:#4CAF50; margin-top:0;">✅ This URL appears to be <strong>{pred_label}</strong></h2>
                        </div>
                        """, unsafe_allow_html=True)
                    else:
                        st.markdown(f"""
                        <div style="background-color:#ffebee; padding:20px; border-radius:10px; border-left:5px solid #F44336;">
                            <h2 style="color:#F44336; margin-top:0;">⚠️ This URL appears to be <strong>{pred_label}</strong></h2>
                        </div>
                        """, unsafe_allow_html=True)
                    
                    # Confidence meter with better styling
                    confidence = np.max(pred_proba) * 100
                    col1, col2 = st.columns([1, 4])
                    with col1:
                        st.markdown(f"""
                        <div style="background-color:#f8f9fa; padding:15px; border-radius:10px; text-align:center;">
                            <h3 style="margin:0; color:#666;">Confidence</h3>
                            <h1 style="margin:0; color:#4a8cff;">{confidence:.1f}%</h1>
                        </div>
                        """, unsafe_allow_html=True)
                    with col2:
                        st.markdown(f"""
                        <div style="padding-top:25px;">
                        """, unsafe_allow_html=True)
                        st.progress(int(confidence))
                    
                    # Feature breakdown with cards
                    st.subheader("🔑 Key Indicators")
                    
                    cols = st.columns(4)
                    with cols[0]:
                        st.markdown(f"""
                        <div style="background-color:#f8f9fa; padding:15px; border-radius:10px; text-align:center; height:120px;">
                            <h3 style="margin:0; color:#666;">Suspicious Words</h3>
                            <h1 style="margin:10px 0; color:#4a8cff;">{'⚠️' if features[16] == 1 else '✅'}</h1>
                            <p style="margin:0; color:#666;">{'Detected' if features[16] == 1 else 'Not detected'}</p>
                        </div>
                        """, unsafe_allow_html=True)
                    with cols[1]:
                        st.markdown(f"""
                        <div style="background-color:#f8f9fa; padding:15px; border-radius:10px; text-align:center; height:120px;">
                            <h3 style="margin:0; color:#666;">URL Length</h3>
                            <h1 style="margin:10px 0; color:#4a8cff;">{features[14]}</h1>
                            <p style="margin:0; color:#666;">characters</p>
                        </div>
                        """, unsafe_allow_html=True)
                    with cols[2]:
                        st.markdown(f"""
                        <div style="background-color:#f8f9fa; padding:15px; border-radius:10px; text-align:center; height:120px;">
                            <h3 style="margin:0; color:#666;">Uses IP</h3>
                            <h1 style="margin:10px 0; color:#4a8cff;">{'⚠️' if features[0] == 1 else '✅'}</h1>
                            <p style="margin:0; color:#666;">{'Yes' if features[0] == 1 else 'No'}</p>
                        </div>
                        """, unsafe_allow_html=True)
                    with cols[3]:
                        st.markdown(f"""
                        <div style="background-color:#f8f9fa; padding:15px; border-radius:10px; text-align:center; height:120px;">
                            <h3 style="margin:0; color:#666;">Shortened URL</h3>
                            <h1 style="margin:10px 0; color:#4a8cff;">{'⚠️' if features[7] == 1 else '✅'}</h1>
                            <p style="margin:0; color:#666;">{'Yes' if features[7] == 1 else 'No'}</p>
                        </div>
                        """, unsafe_allow_html=True)
                    
                    # SHAP explanation
                    if show_shap:
                        st.markdown("---")
                        explain_with_shap(model, features_array, feature_names)
                    
                    # Technical details
                    if show_technical:
                        with st.expander("🔧 Technical Details", expanded=False):
                            st.write("Feature Values:")
                            feature_df = pd.DataFrame([features], columns=feature_names)
                            st.dataframe(feature_df.style.background_gradient(cmap='Blues'))
                            
                            st.write("Prediction Probabilities:")
                            proba_df = pd.DataFrame([pred_proba], 
                                                  columns=["Safe", "Defacement", "Phishing", "Malware"])
                            st.dataframe(proba_df.style.background_gradient(cmap='Greens', axis=1))
                    
                    # Generate report
                    st.markdown("---")
                    report = generate_pdf_report(url_input, pred_label, features, pred_proba)
                    st.download_button(
                        label="📥 Download Report",
                        data=report,
                        file_name=f"url_analysis_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt",
                        mime="text/plain",
                        use_container_width=True
                    )
                    
                    # Save to history
                    if 'history' not in st.session_state:
                        st.session_state.history = []
                    
                    st.session_state.history.append({
                        'url': url_input,
                        'prediction': pred_label,
                        'confidence': confidence,
                        'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                    })
                    
            except Exception as e:
                st.error(f"❌ Error analyzing URL: {str(e)}")
    
    elif menu == "Threat Analysis Dashboard":
        st.markdown("""
        <div style="background-color:#4a8cff; padding:20px; border-radius:10px; margin-bottom:30px;">
            <h1 style="color:white; margin:0;">📊 Threat Analysis Dashboard</h1>
            <p style="color:white; margin:0;">Visual analytics of URL threat patterns and trends</p>
        </div>
        """, unsafe_allow_html=True)
        
        # Sample data with more entries
        sample_data = {
            'type': ['SAFE', 'PHISHING', 'MALWARE', 'DEFACEMENT', 'SAFE', 'PHISHING', 'MALWARE', 'SAFE', 'PHISHING'],
            'url_length': [20, 85, 120, 65, 25, 90, 110, 30, 95],
            'count_dir': [1, 5, 8, 3, 2, 6, 7, 1, 5],
            'sus_words': [0, 1, 1, 0, 0, 1, 1, 0, 1],
            'timestamp': pd.date_range('2023-01-01', periods=9).tolist()
        }
        df = pd.DataFrame(sample_data)
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("""
            <div style="background-color:#f8f9fa; padding:15px; border-radius:10px;">
                <h3 style="margin-top:0; color:#4a8cff;">Threat Distribution</h3>
            </div>
            """, unsafe_allow_html=True)
            fig = px.pie(df, names='type', hole=0.3, color='type',
                        color_discrete_map={'SAFE':'#4CAF50','PHISHING':'#F44336',
                                           'MALWARE':'#FF9800','DEFACEMENT':'#9C27B0'})
            fig.update_traces(textposition='inside', textinfo='percent+label')
            st.plotly_chart(fig, use_container_width=True)
            
        with col2:
            st.markdown("""
            <div style="background-color:#f8f9fa; padding:15px; border-radius:10px;">
                <h3 style="margin-top:0; color:#4a8cff;">URL Length by Threat Type</h3>
            </div>
            """, unsafe_allow_html=True)
            fig = px.box(df, x='type', y='url_length', color='type',
                        color_discrete_map={'SAFE':'#4CAF50','PHISHING':'#F44336',
                                           'MALWARE':'#FF9800','DEFACEMENT':'#9C27B0'})
            fig.update_layout(showlegend=False)
            st.plotly_chart(fig, use_container_width=True)
        
        st.markdown("""
        <div style="background-color:#f8f9fa; padding:15px; border-radius:10px; margin-bottom:20px;">
            <h3 style="margin-top:0; color:#4a8cff;">Recent Threat Patterns</h3>
        </div>
        """, unsafe_allow_html=True)
        fig = px.scatter(df, x='timestamp', y='url_length', color='type', size='count_dir',
                         hover_name='type', size_max=20,
                         color_discrete_map={'SAFE':'#4CAF50','PHISHING':'#F44336',
                                           'MALWARE':'#FF9800','DEFACEMENT':'#9C27B0'})
        fig.update_layout(xaxis_title='Date', yaxis_title='URL Length')
        st.plotly_chart(fig, use_container_width=True)
    
    elif menu == "Historical Reports":
        st.markdown("""
        <div style="background-color:#4a8cff; padding:20px; border-radius:10px; margin-bottom:30px;">
            <h1 style="color:white; margin:0;">📜 Historical Reports</h1>
            <p style="color:white; margin:0;">Review previously analyzed URLs and their results</p>
        </div>
        """, unsafe_allow_html=True)
        
        if 'history' not in st.session_state or len(st.session_state.history) == 0:
            st.warning("No analysis history found. Analyze some URLs first.")
        else:
            history_df = pd.DataFrame(st.session_state.history)
            
            # Apply color coding to the dataframe
            def color_row(row):
                if row['prediction'] == 'SAFE':
                    return ['background-color: #e6f7e6'] * len(row)
                else:
                    return ['background-color: #ffebee'] * len(row)
            
            st.dataframe(
                history_df.style.apply(color_row, axis=1),
                use_container_width=True,
                height=min(400, 45 * len(history_df) + 45)
            )
            
            st.markdown("""
            <div style="background-color:#f8f9fa; padding:15px; border-radius:10px; margin-bottom:20px;">
                <h3 style="margin-top:0; color:#4a8cff;">History Analysis</h3>
            </div>
            """, unsafe_allow_html=True)
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown(f"""
                <div style="background-color:#f0f2f6; padding:20px; border-radius:10px; text-align:center;">
                    <h3 style="margin:0; color:#666;">Total URLs Analyzed</h3>
                    <h1 style="margin:10px 0; color:#4a8cff;">{len(history_df)}</h1>
                </div>
                """, unsafe_allow_html=True)
                
            with col2:
                threats = len(history_df[history_df['prediction'] != 'SAFE'])
                st.markdown(f"""
                <div style="background-color:#f0f2f6; padding:20px; border-radius:10px; text-align:center;">
                    <h3 style="margin:0; color:#666;">Potential Threats Detected</h3>
                    <h1 style="margin:10px 0; color:#F44336;">{threats}</h1>
                </div>
                """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()