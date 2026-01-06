import streamlit as st
from transformers import BertTokenizer, BertForSequenceClassification
import torch
from PIL import Image

# --- Page Configuration ---
# This must be the first Streamlit command in your script
st.set_page_config(
    page_title="SafeGuard AI",
    page_icon="🛡️",
    layout="centered",
    initial_sidebar_state="auto",
)

# --- Configuration ---
MODEL_PATH = "BERT_Bullying_Detector_Model"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# --- Load Model and Tokenizer (with caching) ---
@st.cache_resource
def load_model():
    try:
        model = BertForSequenceClassification.from_pretrained(MODEL_PATH)
        tokenizer = BertTokenizer.from_pretrained(MODEL_PATH)
        model.to(device)
        model.eval()
        return model, tokenizer
    except Exception as e:
        st.error(f"Error loading model: {e}. Make sure the model files are in the 'BERT_Bullying_Detector_Model' folder.")
        return None, None

model, tokenizer = load_model()

# --- Streamlit App Interface ---
st.title("🛡️ SafeGuard AI")
st.subheader("A Real-Time Bullying and Toxicity Detector")
st.write(
    "Enter any text below to check if it contains harmful content. "
    "This tool is built on a fine-tuned BERT model to help promote safer online interactions."
)

user_input = st.text_area("Enter your text here:", "", height=150, placeholder="e.g., 'You are so smart!' or 'I hate this, it's terrible.'")

if st.button("Analyze Text"):
    if model and tokenizer and user_input:
        # --- Prediction Logic ---
        encoding = tokenizer.encode_plus(
            user_input,
            add_special_tokens=True,
            max_length=128,
            return_token_type_ids=False,
            padding='max_length',
            return_tensors='pt',
            truncation=True
        )
        
        input_ids = encoding['input_ids'].to(device)
        attention_mask = encoding['attention_mask'].to(device)

        with torch.no_grad():
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            _, prediction = torch.max(outputs.logits, dim=1)
            
        confidence = torch.softmax(outputs.logits, dim=1).max().item()
        label = "Bullying" if prediction.item() == 1 else "Not Bullying"

        # --- Display Results ---
        st.write("### Prediction:")
        if label == "Bullying":
            st.error(f"**{label}** (Confidence: {confidence:.2f})")
        else:
            st.success(f"**{label}** (Confidence: {confidence:.2f})")
    elif not user_input:
        st.warning("Please enter some text to analyze.")

# --- Performance Section ---
with st.expander("Learn More About The Model's Performance"):
    st.write("""
    This model is a fine-tuned version of `bert-base-uncased`, a powerful language model developed by Google. 
    It was trained on a balanced dataset of over 115,000 text samples to learn the nuances of toxic and non-toxic language.
    """)
    
    st.write("#### Key Performance Metrics:")
    st.text("""
    - Accuracy: 93%
    - Precision (Bullying): 90%
    - Recall (Bullying): 96%
    - F1-Score (Bullying): 93%
    """)
    
    st.write("#### Confusion Matrix:")
    st.write("The matrix below shows how the model performed on the test data. It correctly identified 5455 bullying cases while misclassifying only 225.")
    
    try:
        # Make sure the confusion matrix image is in the same folder as app.py
        # or provide the correct path.
        image = Image.open('confusion_matrix.png')
        st.image(image, caption='Confusion Matrix for the BERT model')
    except FileNotFoundError:
        st.warning("`confusion_matrix.png` not found. Please add it to your repository to display the matrix.")

st.markdown("<br><hr><center>Developed with ❤️ by a student passionate about safe AI.</center>", unsafe_allow_html=True)