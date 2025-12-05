import streamlit as st
import json
from dotenv import load_dotenv
from questionii_model import QuestioniiAI
import sys

# Load environment variables
load_dotenv()

# Page configuration
st.set_page_config(
    page_title="Questionii AI",
    page_icon="",   
    layout="wide"
)

# Initialize AI model with caching
@st.cache_resource
def initialize_ai_model():
    """Initialize the QuestioniiAI model with error handling for missing API key."""
    # Check if the model can be initialized successfully
    try:
        return QuestioniiAI()
    except ValueError as e:
        st.error(f"Configuration Error: {str(e)}")
        st.error("Please ensure your GEMINI_API_KEY is set in the .env file.")
        # Halt execution gracefully
        sys.exit() 

# Initialize the model
ai_model = initialize_ai_model()

# Sidebar - Configuration and Feedback
with st.sidebar:
    st.header("Configuration")
    
    # Generation controls
    num_questions = st.slider(
        "Number of Questions",
        min_value=1,
        max_value=10,
        value=3,
        help="Select how many questions to generate"
    )
    
    difficulty = st.selectbox(
        "Difficulty Level",
        options=["Easy", "Medium", "Hard"],
        index=1,
        help="Choose the difficulty level for questions"
    )
    
    st.divider()
    
    # Feedback/Memory section
    with st.expander("**Model Learning History**", expanded=False):
        st.markdown("### Previous Feedback")
        
        # Display existing feedback
        if ai_model.feedback_history:
            for idx, feedback in enumerate(ai_model.feedback_history, 1):
                st.info(f"**{idx}.** {feedback}")
        else:
            st.caption("No feedback recorded yet. Add your first feedback below!")
        
        st.divider()
        
        # Add new feedback
        st.markdown("### Add New Feedback")
        new_feedback = st.text_input(
            "Enter feedback for the model",
            placeholder="e.g., Focus more on practical applications...",
            key="feedback_input"
        )
        
        if st.button("Save Feedback & Rerun", type="primary"):
            if new_feedback.strip():
                ai_model.handle_feedback(new_feedback.strip())
                st.success("✅ Feedback saved successfully!")
                st.rerun()
            else:
                st.warning("Please enter some feedback before saving.")

# Main area
st.title(" Questionii AI")
st.markdown("Generate intelligent multiple-choice questions from any text content.")

# Input area
text_content = st.text_area(
    " Enter Your Text Content",
    height=350,
    placeholder="Paste your study material, article, or any text here...\n\n"
                " Tip: For best results, provide at least 50 words of content. "
                "The more detailed your input, the better the generated questions!",
    help="Enter the text you want to generate questions from"
)

# Generate button
col1, col2, col3 = st.columns([1, 2, 1])
with col2:
    generate_button = st.button(
        " Generate Questions",
        type="primary",
        use_container_width=True
    )

# Generation logic
if generate_button:
    # Check if content is empty
    if not text_content.strip():
        st.warning("Please enter some text content before generating questions.")
    else:
        # Check word count (soft warning)
        word_count = len(text_content.split())
        if word_count < 50:
            st.warning(
                f"Your input contains only {word_count} words. "
                f"For optimal results, we recommend at least 50 words. "
                f"Proceeding with generation..."
            )
        
        # Generate questions with spinner
        with st.spinner(" AI is analyzing your content and generating questions..."):
            # NOTE: The parameter name for content in questionii_model is 'text_content'
            result = ai_model.generate_questions(
                text_content=text_content, 
                num_questions=num_questions,
                difficulty=difficulty
            )
        
        # Display results
        if result["status"] == "success":
            st.success("✅ Generation Complete!")
            st.markdown(f"**Generated {len(result['data'])} questions:**")
            st.divider()
            
            # Display each question
            for idx, question_data in enumerate(result['data'], 1):
                # Ensure options is a list before processing
                options_list = question_data.get('options', [])
                
                with st.expander(f"**Question {idx}:** {question_data['question']}", expanded=True):
                    # Display options - CORRECTED LOOP: iterates over a list
                    st.markdown("**Options:**")
                    
                    # Assuming the list contains the options strings, we assign letters A, B, C...
                    for i, option_value in enumerate(options_list):
                        # Convert index (0, 1, 2...) to letter (A, B, C...)
                        option_key = chr(65 + i) 
                        st.markdown(f"- **{option_key}:** {option_value}")
                    
                    st.divider()
                    
                    # Display correct answer
                    st.success(f"**✓ Correct Answer:** {question_data['correct_answer']}")
                    
                    # Display explanation
                    st.info(f"**Explanation:** {question_data['explanation']}")
        
        else:
            # Display error
            st.error(f"Error: {result['message']}")
            st.error("Please try again with different content or check your API configuration.")

