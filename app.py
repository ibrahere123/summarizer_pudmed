import streamlit as st
from transformers import T5Tokenizer, T5ForConditionalGeneration

# Set the page config
st.set_page_config(page_title="PubMed Article Summarizer", layout="wide")

# Custom CSS for styling
st.markdown(
    """
    <style>
    .main {
        background-color: grey;  /* Light grey background */
        padding: 20px;
    }
    .stButton>button {
        background-color: #4CAF50;  /* Green button */
        color: #333333;  /* Dark grey button text */
        border-radius: 5px;
        padding: 8px 16px;
        font-size: 14px;
    }
    .stTextInput>div>div>input {
        border: 1px solid #ccc;
        border-radius: 5px;
        padding: 8px;
        font-size: 14px;
    }
    .stFileUploader>div>div>input {
        border: 1px solid #ccc;
        border-radius: 5px;
        padding: 8px;
        font-size: 14px;
    }
    .stSelectbox>div>div>div {
        border: 1px solid #ccc;
        border-radius: 5px;
        padding: 8px;
        font-size: 14px;
    }
    .stText {
        color: #333333;  /* Dark grey text */
    }
    .summarized-title {
        background-color: #4CAF50;  /* Green title background */
        color: white;  /* White title text */
        padding: 10px;
        margin-bottom: 10px;
        border-radius: 5px;
    }
    .summarized-text {
        background-color: white;  /* White background */
        color: #333333;  /* Dark grey text */
        padding: 10px;
        border-radius: 5px;
    }
    .title-text {
        color: grey;  /* Grey title text color */
        font-size: 24px;  /* Adjust title font size */
        font-weight: bold;  /* Bold title */
        margin-bottom: 20px;  /* Add space below the title */
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# Function to load model and tokenizer
def load_model_and_tokenizer(model_name):
    try:
        tokenizer = T5Tokenizer.from_pretrained(model_name)
        model = T5ForConditionalGeneration.from_pretrained(model_name)
        return tokenizer, model
    except Exception as e:
        st.error(f"Error loading model or tokenizer: {e}")
        return None, None

# Initialize tokenizer and model
model_name = "./models/fine_tuned_t5_document"  # Adjust the path as necessary
tokenizer, model = load_model_and_tokenizer(model_name)

# Main application code
if tokenizer is not None and model is not None:
    def summarize_text(text, summary_type):
        try:
            num_tokens = len(tokenizer.encode(text, return_tensors="pt")[0])
            if summary_type == 'brief':
                max_length = 150 if num_tokens <= 500 else 300
                min_length = 40 if num_tokens <= 500 else 100
            elif summary_type == 'detailed':
                max_length = 300 if num_tokens <= 500 else 500
                min_length = 100 if num_tokens <= 500 else 200
            else:
                st.error("Invalid summary type selected.")
                return None
            
            inputs = tokenizer.encode("summarize: " + text, return_tensors="pt", max_length=512, truncation=True)
            summary_ids = model.generate(inputs, max_length=max_length, min_length=min_length, length_penalty=2.0, num_beams=4, early_stopping=True)
            summary = tokenizer.decode(summary_ids[0], skip_special_tokens=True)
            return summary
        except Exception as e:
            st.error(f"Error during summarization: {e}")
            return None

    st.title("PubMed Article Summarizer")

    # Sidebar with instructions and input options
    st.sidebar.markdown(
        """
        ### Instructions
        Upload a file (TXT, MD, PDF) or paste the article text into the text area. 
        Select the summary type (brief or detailed) and click **Summarize**.
        """
    )

    # Input method to upload or input article text
    st.sidebar.subheader("Upload or input article")
    uploaded_file = st.sidebar.file_uploader("Choose a file", type=["txt", "md", "pdf"])
    if uploaded_file is not None:
        try:
            article_text = uploaded_file.read().decode("utf-8")
        except Exception as e:
            st.error(f"Error reading file: {e}")
            article_text = ""
    else:
        article_text = st.sidebar.text_area("Or paste the article text here", height=200)

    summary_type = st.sidebar.selectbox("Select summary type", ["brief", "detailed"])

    if st.sidebar.button("Summarize"):
        if article_text:
            with st.spinner("Summarizing..."):
                summary = summarize_text(article_text, summary_type)
            if summary:
                st.markdown('<div class="summarized-title">Summarized Article</div>', unsafe_allow_html=True)
                st.markdown(f'<div class="summarized-text">{summary}</div>', unsafe_allow_html=True)
        else:
            st.warning("Please provide an article text or upload a file.")

else:
    st.error("Model or tokenizer could not be loaded. Please check the paths and try again.")
