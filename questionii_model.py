import google.generativeai as genai
import json
import os
from dotenv import load_dotenv
# Load environment variables from .env file
load_dotenv()
class QuestioniiAI:
    def __init__(self):
        # 1. Setup Gemini
        api_key = os.getenv("GEMINI_API_KEY")
        
        if not api_key:
            raise ValueError("GEMINI_API_KEY not found. Make sure you have a .env file with the key.")

        genai.configure(api_key=api_key)
        self.model = genai.GenerativeModel('gemini-2.5-flash')
        
        # 2. Memory for Feedback
        self.feedback_history = []

    def _is_content_sufficient(self, text):
        """
        Check if the content is enough to generate questions.
        Rule: If text is less than 50 words, it's probably not enough.
        """
        if not text or len(text.split()) < 50:
            return False
        return True

    def _clean_json_response(self, response_text):
        """
        Helper function to clean markdown from JSON string (e.g. ```json ... ```)
        """
        return response_text.replace('```json', '').replace('```', '').strip()

    def generate_questions(self, text_content, num_questions, difficulty):
        """
        Main function to generate questions.
        """
        # Step 1: Check Capability (Validation)
        if not self._is_content_sufficient(text_content):
            return {
                "status": "error",
                "message": "The provided content is not capable/enough to generate high-quality questions. Please provide more text."
            }

        # Step 2: Construct the Prompt
        feedback_instruction = ""
        if self.feedback_history:
            feedback_instruction = f"IMPORTANT: Avoid the following based on previous user feedback: {'; '.join(self.feedback_history)}."

        prompt = f"""
        Act as an expert exam creator for a university student.
        
        Context Material:
        "{text_content}"

        Task:
        Generate {num_questions} questions based ONLY on the text above.
        Difficulty Level: {difficulty} (Choose between Easy, Medium, Hard).
        
        {feedback_instruction}

        Output Format:
        Return a strict JSON array of objects. Each object must have:
        - "id": number
        - "question": string
        - "options": list of strings (if MCQ)
        - "correct_answer": string
        - "explanation": string (Why is this correct?)

        Do not wrap the output in markdown code blocks. Just return the raw JSON.
        """

        try:
            # Step 3: Call Gemini API
            response = self.model.generate_content(prompt)
            
            # Step 4: Parse Result
            cleaned_text = self._clean_json_response(response.text)
            questions_json = json.loads(cleaned_text)
            
            return {
                "status": "success",
                "data": questions_json
            }

        except Exception as e:
            return {"status": "error", "message": str(e)}

    def handle_feedback(self, feedback_text):
        """
        Store user feedback to influence future generations.
        Ex: User says "Don't ask about dates", model remembers this.
        """
        print(f"--> System: Learning from feedback: '{feedback_text}'")
        self.feedback_history.append(feedback_text)
        return True