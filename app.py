# app.py (Unified with Summarization and ASR - FIX FOR DUPLICATE KEYS)

import streamlit as st
import google.generativeai as genai
import json
import os
from dotenv import load_dotenv
from PIL import Image
from io import BytesIO
import pypdfium2 as pdfium
import torch # For ASR
from transformers import pipeline # For ASR
# import soundfile as sf # Not directly used in the final transcription call
# import librosa # Not directly used in the final transcription call

# --- 0. Streamlit Page Configuration ---
st.set_page_config(
    page_title="questionii AI",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- Custom CSS for Better Design (Keeping your original CSS) ---
st.markdown("""
<style>
    /* Main container styling */
    .main {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    }
    
    /* Card-like containers */
    .stTextArea, .stNumberInput, .stSelectbox {
        background-color: white;
        border-radius: 10px;
        padding: 10px;
    }
    
    /* Buttons */
    .stButton>button {
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        color: white;
        border-radius: 20px;
        padding: 10px 30px;
        font-weight: bold;
        border: none;
        transition: all 0.3s ease;
    }
    
    .stButton>button:hover {
        transform: translateY(-2px);
        box-shadow: 0 5px 15px rgba(0,0,0,0.3);
    }
    
    /* Tab styling */
    .stTabs [data-baseweb="tab-list"] {
        gap: 10px;
    }
    
    .stTabs [data-baseweb="tab"] {
        background-color: rgba(255,255,255,0.1);
        border-radius: 10px;
        padding: 10px 20px;
        color: white;
    }
    
    .stTabs [aria-selected="true"] {
        background-color: white;
        color: #667eea;
    }
    
    /* Header styling */
    h1 {
        color: white;
        text-align: center;
        font-size: 3rem;
        margin-bottom: 0;
    }
    
    h2, h3 {
        color: white;
    }
    
    /* Info boxes */
    .stInfo, .stSuccess, .stError {
        border-radius: 10px;
    }
    
    /* JSON display */
    .stJson {
        background-color: white;
        border-radius: 10px;
        padding: 15px;
    }
</style>
""", unsafe_allow_html=True)

# ------------------------
# 1. Configuration and Model Initialization
# ------------------------
load_dotenv()
api_key = os.getenv("GEMINI_API_KEY")

if api_key:
    genai.configure(api_key=api_key)
else:
    st.error("❌ ERROR: GEMINI_API_KEY not found. Please check your .env file.")
    st.stop()
    
GEMINI_MODEL = 'gemini-2.5-flash'

# Initialize Whisper ASR Pipeline (Caching resource)
@st.cache_resource
def load_asr_pipeline():
    # Use CPU (-1) for Streamlit Cloud to avoid GPU issues unless running locally with GPU
    device = 0 if torch.cuda.is_available() and 'st-cpu' not in os.environ else -1 
    MODEL_NAME = "openai/whisper-small"
    try:
        asr_pipeline = pipeline(
            "automatic-speech-recognition",
            model=MODEL_NAME,
            device=device,
            torch_dtype=torch.float16 if device == 0 else None
        )
        return asr_pipeline
    except Exception as e:
        st.warning(f"⚠️ ASR Error: Whisper model couldn't load. ASR tab will not function fully. ({e})")
        return None

asr_pipeline = load_asr_pipeline()

# ------------------------
# 2. Question Generator Class
# ------------------------
class QuestionGeneratorAI:
    def __init__(self):
        self.model = genai.GenerativeModel(GEMINI_MODEL)
        self.feedback_history = []
    
    def _is_content_sufficient(self, text):
        return text and len(text.split()) >= 50
    
    def _clean_json(self, txt):
        return txt.replace("```json", "").replace("```", "").strip()
    
    def generate_questions(self, text_content, num_questions, difficulty):
        if not self._is_content_sufficient(text_content):
            return {
                "status": "error", 
                "message": " The text is too short. Please add more content (at least 50 words)."
            }
        
        # Include feedback history in prompt if available
        feedback_context = ""
        if self.feedback_history:
            feedback_context = "\n\nPrevious Feedback to Consider:\n"
            for fb in self.feedback_history[-3:]: 
                feedback_context += f"- {fb}\n"
        
        prompt = f"""
Generate {num_questions} high-quality multiple choice questions based on the following text:

{text_content}

Difficulty Level: {difficulty}
{feedback_context}

Requirements:
- Create questions that test understanding, not just memorization
- Provide 4 options (A, B, C, D) for each question
- Include a clear explanation for the correct answer
- Make distractors plausible but clearly incorrect

Format: Return ONLY a valid JSON array with this exact structure:
[
    {{
        "id": 1,
        "question": "Question text here?",
        "options": ["Option A", "Option B", "Option C", "Option D"],
        "correct_answer": "Option A",
        "explanation": "Detailed explanation of why this is correct"
    }}
]
"""
        
        try:
            response = self.model.generate_content(prompt)
            cleaned = self._clean_json(response.text)
            data = json.loads(cleaned)
            return {"status": "success", "data": data}
        except json.JSONDecodeError as e:
            return {"status": "error", "message": f"Invalid JSON format: {str(e)}"}
        except Exception as e:
            return {"status": "error", "message": f"Error generating questions: {str(e)}"}
    
    def improve_questions_with_feedback(self, questions, feedback_text, original_content):
        """Regenerate questions based on user feedback"""
        self.feedback_history.append(feedback_text)
        
        prompt = f"""
The following questions were generated but received this feedback:
"{feedback_text}"

Original Content:
{original_content}

Original Questions:
{json.dumps(questions, indent=2)}

Please improve or regenerate the questions based on the feedback while maintaining the same format.

Return ONLY a valid JSON array with this structure:
[
    {{
        "id": 1,
        "question": "Improved question text here?",
        "options": ["Option A", "Option B", "Option C", "Option D"],
        "correct_answer": "Option A",
        "explanation": "Detailed explanation of why this is correct"
    }}
]
"""
        
        try:
            response = self.model.generate_content(prompt)
            cleaned = self._clean_json(response.text)
            data = json.loads(cleaned)
            return {"status": "success", "data": data}
        except Exception as e:
            return {"status": "error", "message": f"Error improving questions: {str(e)}"}

# ------------------------
# 3. Summarization Function
# ------------------------
def summarize_text_with_gemini(text_to_summarize):
    """Uses Gemini API to summarize text into a clear bulleted list."""
    
    prompt = f"""
    You are an expert summarizer. Summarize the following English text into clear, concise, and structured bullet points.
    For the list format, strictly use hyphens (-) or numbers (1., 2., 3.).
    Do not use any Markdown formatting in the output, including asterisks (*), double asterisks (**), or any form of bolding or italics.
    Focus only on the main ideas.

    TEXT TO SUMMARIZE:
    ---
    {text_to_summarize}
    ---
    """
    try:
        model = genai.GenerativeModel(GEMINI_MODEL)
        response = model.generate_content(prompt)
        return response.text
    except Exception as e:
        st.error(f"❌ Gemini API Error during Summarization: {e}")
        return "Error during summarization."

# ------------------------
# 4. PDF Text Extraction
# ------------------------
def extract_text_from_pdf(file_bytes):
    try:
        pdf = pdfium.PdfDocument(BytesIO(file_bytes))
        all_text = ""
        for i in range(len(pdf)):
            page = pdf[i]
            textpage = page.get_textpage()
            all_text += textpage.get_text_range() + "\n"
        return all_text
    except Exception as e:
        return f"Error extracting text: {str(e)}"

# ------------------------
# 5. Display Questions Function (FIXED: Added key_prefix)
# ------------------------
def display_questions(questions_data, key_prefix, show_rating=True):
    """
    Displays questions with feedback options. 
    Requires a unique key_prefix to avoid StreamlitDuplicateElementKey error across tabs.
    """
    question_ratings = {}
    
    for i, q in enumerate(questions_data, 1):
        # Ensure the expander key is also unique
        expander_key = f"{key_prefix}_q_expander_{i}" 
        
        with st.expander(f" Question {i}: {q['question']}", expanded=True):
            st.markdown(f"**Question:** {q['question']}")
            st.markdown("**Options:**")
            for opt in q['options']:
                st.write(f"- {opt}")
            st.success(f" **Correct Answer:** {q['correct_answer']}")
            st.info(f" **Explanation:** {q['explanation']}")
            
            if show_rating:
                st.markdown("---")
                col1, col2 = st.columns([3, 1])
                with col1:
                    rating = st.select_slider(
                        "Rate this question:",
                        options=[" Poor", " Fair", " Good", " Very Good", " Excellent"],
                        value=" Good",
                        # --- FIX: Use unique key_prefix ---
                        key=f"{key_prefix}_rating_{i}" 
                    )
                    question_ratings[i] = rating
                with col2:
                    # --- FIX: Use unique key_prefix ---
                    if st.button(" Remove", key=f"{key_prefix}_remove_{i}"):
                        st.warning(f"Question {i} marked for removal (Regenerate to see changes)")
                
            st.divider()
    
    return question_ratings


# ------------------------
# 6. Main Application
# ------------------------
def main():
    # Header
    st.markdown("<h1>questionii </h1>", unsafe_allow_html=True)
    st.markdown("<p style='text-align: center; color: white; font-size: 1.2rem;'>Transform any content into exam questions, summaries, or transcribed audio</p>", unsafe_allow_html=True)
    st.markdown("---")
    
    # Initialize model
    if 'model_ai' not in st.session_state:
        st.session_state.model_ai = QuestionGeneratorAI()
    
    model = st.session_state.model_ai
    
    # Sidebar
    with st.sidebar:
        st.header(" Settings")
        st.markdown("### About")
        st.info("""
    
        **Features:**
        - Question Generation (Text/PDF)
        - **Text Summarization**
        - **Speech-to-Text (ASR)**
        """)
        
        st.markdown("### Tips")
        st.success("""
        Use at least 50 words for questions/summaries.
        For ASR, upload clear audio (mp3/wav).
        Review and provide feedback for better results.
        """)
    
    # Main tabs
    tab1, tab2, tab3, tab4 = st.tabs([
        "❓ Questions from Text", 
        "📄 Questions from PDF", 
        "📝 Text Summarization", 
        "🎤 Speech-to-Text (ASR)"
    ])
    
    # ------------------------
    # TAB 1: TEXT MODE
    # ------------------------
    with tab1:
        st.subheader(" Generate Questions from Text")
        
        col1, col2 = st.columns([2, 1])
        
        with col1:
            user_text = st.text_area(
                "Write or paste your content here:",
                height=300,
                placeholder="Enter the text you want to generate questions from...",
                help="Minimum 50 words recommended for best results",
                key="text_input_q"
            )
            word_count = len(user_text.split())
            st.caption(f"Word count: {word_count}")
        
        with col2:
            st.subheader(" Configuration")
            num_q = st.number_input(
                "Number of Questions",
                min_value=1,
                max_value=50,
                value=5,
                help="How many questions to generate",
                key="num_q_text"
            )
            difficulty = st.selectbox(
                "Difficulty Level",
                ["Easy", "Medium", "Hard"],
                index=1,
                help="Choose the difficulty level of questions",
                key="diff_text"
            )
            st.markdown("")
            generate_btn = st.button(" Generate Questions", key="gen_text_btn", use_container_width=True)
        
        if 'text_questions' not in st.session_state:
            st.session_state.text_questions = None
        if 'text_content' not in st.session_state:
            st.session_state.text_content = None
        
        if generate_btn:
            if not user_text.strip():
                st.warning(" Please enter some text first!")
            else:
                with st.spinner(" AI is generating questions..."):
                    result = model.generate_questions(user_text, num_q, difficulty)
                
                if result["status"] == "success":
                    st.session_state.text_questions = result["data"]
                    st.session_state.text_content = user_text
                    st.success(" Questions generated successfully!")
                else:
                    st.error(result["message"])
        
        if st.session_state.text_questions:
            st.markdown("---")
            # --- FIX APPLIED HERE ---
            display_questions(st.session_state.text_questions, key_prefix="text_q", show_rating=True)
            
            # Download button
            json_str = json.dumps(st.session_state.text_questions, indent=2)
            st.download_button(
                label=" Download Questions (JSON)",
                data=json_str,
                file_name="generated_questions.json",
                mime="application/json",
                key="download_text_q"
            )
            
            # Feedback Section
            st.markdown("---")
            st.subheader(" Provide Feedback to Improve Questions")
            
            col1_fb, col2_fb = st.columns([3, 1])
            with col1_fb:
                feedback_options = st.multiselect(
                    "What would you like to improve?",
                    [
                        "Make questions more challenging",
                        "Make questions easier",
                        "Focus more on specific concepts",
                        "Add more detail to explanations",
                        "Make options less obvious",
                        "Change question types"
                    ],
                    key="text_feedback_options"
                )
                
                custom_feedback = st.text_area(
                    "Additional feedback or specific instructions:",
                    height=100,
                    placeholder="E.g., 'Focus more on chapter 3', 'Add more application-based questions', etc.",
                    key="text_custom_feedback"
                )
            
            with col2_fb:
                st.markdown("### Quick Actions")
                if st.button(" Regenerate All", key="text_regen", use_container_width=True):
                    feedback_text = " ".join(feedback_options)
                    if custom_feedback:
                        feedback_text += f". {custom_feedback}"
                    
                    if feedback_text.strip():
                        with st.spinner(" Regenerating questions based on your feedback..."):
                            result = model.improve_questions_with_feedback(
                                st.session_state.text_questions,
                                feedback_text,
                                st.session_state.text_content
                            )
                        
                        if result["status"] == "success":
                            st.session_state.text_questions = result["data"]
                            st.success(" Questions regenerated!")
                            st.rerun()
                        else:
                            st.error(result["message"])
                    else:
                        st.warning("Please select or write feedback first")
            
            if model.feedback_history:
                with st.expander(" Feedback History"):
                    for idx, fb in enumerate(model.feedback_history, 1):
                        st.write(f"{idx}. {fb}")

    
    # ------------------------
    # TAB 2: PDF MODE
    # ------------------------
    with tab2:
        st.subheader(" Generate Questions from PDF")
        
        col1_pdf, col2_pdf = st.columns([2, 1])
        
        with col1_pdf:
            pdf_file = st.file_uploader(
                "Choose a PDF file",
                type=["pdf"],
                help="Upload a PDF document to extract text and generate questions",
                key="pdf_uploader"
            )
            
            extracted_text = ""
            if pdf_file:
                st.success(f" File uploaded: {pdf_file.name}")
                
                with st.spinner(" Extracting text from PDF..."):
                    pdf_bytes = pdf_file.read()
                    extracted_text = extract_text_from_pdf(pdf_bytes)
                
                if extracted_text.startswith("Error"):
                    st.error(extracted_text)
                else:
                    st.subheader(" Extracted Text")
                    with st.expander("View extracted text", expanded=False):
                        st.text_area(
                            "Extracted Content:",
                            extracted_text,
                            height=300,
                            disabled=True,
                            key="extracted_text_pdf"
                        )
                    word_count_pdf = len(extracted_text.split())
                    st.caption(f"Word count: {word_count_pdf}")
        
        if 'pdf_questions' not in st.session_state:
            st.session_state.pdf_questions = None
        if 'pdf_content' not in st.session_state:
            st.session_state.pdf_content = None
        
        with col2_pdf:
            if pdf_file:
                st.subheader(" Configuration")
                num_q_pdf = st.number_input(
                    "Number of Questions",
                    min_value=1,
                    max_value=50,
                    value=5,
                    key="pdf_num",
                    help="How many questions to generate"
                )
                difficulty_pdf = st.selectbox(
                    "Difficulty Level",
                    ["Easy", "Medium", "Hard"],
                    index=1,
                    key="pdf_diff",
                    help="Choose the difficulty level of questions"
                )
                st.markdown("")
                generate_pdf_btn = st.button(" Generate Questions", key="pdf_gen", use_container_width=True, disabled=extracted_text.startswith("Error"))
                
                if generate_pdf_btn:
                    with st.spinner(" AI is generating questions from PDF..."):
                        result = model.generate_questions(extracted_text, num_q_pdf, difficulty_pdf)
                    
                    if result["status"] == "success":
                        st.session_state.pdf_questions = result["data"]
                        st.session_state.pdf_content = extracted_text
                        st.success(" Questions generated successfully!")
                    else:
                        st.error(result["message"])
        
        if st.session_state.pdf_questions:
            st.markdown("---")
            # --- FIX APPLIED HERE ---
            display_questions(st.session_state.pdf_questions, key_prefix="pdf_q", show_rating=True)
            
            # Download and Feedback logic for PDF (similar to text mode, using different keys)
            json_str = json.dumps(st.session_state.pdf_questions, indent=2)
            st.download_button(
                label=" Download Questions (JSON)",
                data=json_str,
                file_name="generated_questions_pdf.json",
                mime="application/json",
                key="pdf_download"
            )

            # Feedback Section for PDF
            st.markdown("---")
            st.subheader(" Provide Feedback to Improve Questions")
            
            col1_pdf_fb, col2_pdf_fb = st.columns([3, 1])
            with col1_pdf_fb:
                feedback_options_pdf = st.multiselect(
                    "What would you like to improve?",
                    [
                        "Make questions more challenging",
                        "Make questions easier",
                        "Focus more on specific concepts",
                        "Add more detail to explanations",
                        "Make options less obvious",
                        "Change question types"
                    ],
                    key="pdf_feedback_options"
                )
                
                custom_feedback_pdf = st.text_area(
                    "Additional feedback or specific instructions:",
                    height=100,
                    placeholder="E.g., 'Focus more on chapter 3', 'Add more application-based questions', etc.",
                    key="pdf_custom_feedback"
                )
            
            with col2_pdf_fb:
                st.markdown("### Quick Actions")
                if st.button(" Regenerate All", key="pdf_regen", use_container_width=True):
                    feedback_text = " ".join(feedback_options_pdf)
                    if custom_feedback_pdf:
                        feedback_text += f". {custom_feedback_pdf}"
                    
                    if feedback_text.strip():
                        with st.spinner(" Regenerating questions based on your feedback..."):
                            result = model.improve_questions_with_feedback(
                                st.session_state.pdf_questions,
                                feedback_text,
                                st.session_state.pdf_content
                            )
                        
                        if result["status"] == "success":
                            st.session_state.pdf_questions = result["data"]
                            st.success(" Questions regenerated!")
                            st.rerun()
                        else:
                            st.error(result["message"])
                    else:
                        st.warning("Please select or write feedback first")
            
            if model.feedback_history:
                with st.expander(" Feedback History"):
                    for idx, fb in enumerate(model.feedback_history, 1):
                        st.write(f"{idx}. {fb}")

    # ------------------------
    # TAB 3: SUMMARIZATION
    # ------------------------
    with tab3:
        st.subheader("Text Summarization ")
        
        sum_text = st.text_area(
            "Paste the English text you want to summarize (50 words recommended):", 
            height=300, 
            key="sum_text_input"
        )
        
        if st.button("Summarize Text", key="summarize_btn", use_container_width=True):
            if not sum_text:
                st.warning("Please enter some text first!")
            elif len(sum_text.split()) < 50:
                st.warning("Please provide at least 50 words for a meaningful summary.")
            else:
                with st.spinner(" Summarizing ..."):
                    summary = summarize_text_with_gemini(sum_text)
                    st.success(" Summary Complete:")
                    st.markdown(summary)

    # ------------------------
    # TAB 4: ASR
    # ------------------------
    with tab4:
        st.subheader("🎤 Speech-to-Text (ASR) using Whisper")
        
        if asr_pipeline is None:
            st.error("Cannot run ASR. Whisper model failed to load or is unavailable.")
        else:
            uploaded_audio = st.file_uploader(
                "Choose an Audio file (e.g., mp3, wav, flac):", 
                type=["mp3", "wav", "flac"],
                key="audio_uploader"
            )
            
            if uploaded_audio is not None:
                st.info(f"File uploaded: {uploaded_audio.name}")
                
                if st.button("Transcribe Audio", key="asr_btn", use_container_width=True):
                    # Transcribe logic
                    audio_bytes = uploaded_audio.read()
                    
                    with st.spinner("🎧 Transcribing audio using Whisper..."):
                        try:
                            # Using temp file approach for robust compatibility with Whisper pipeline
                            temp_file = os.path.join(os.getcwd(), uploaded_audio.name)
                            with open(temp_file, "wb") as f:
                                f.write(audio_bytes)
                                
                            result = asr_pipeline(
                                temp_file,
                                chunk_length_s=30,
                                stride_length_s=(4, 2),
                                generate_kwargs={"language": "english"} 
                            )
                            transcribed_text = result["text"]
                            
                            st.success("✅ Transcription Complete!")
                            st.text_area("Transcribed Text:", transcribed_text, height=300)
                            
                        except Exception as e:
                            st.error(f"❌ Error during transcription: {e}")
                        finally:
                            # Clean up the temporary file
                            if os.path.exists(temp_file):
                                os.remove(temp_file)

if __name__ == "__main__":
    main()
