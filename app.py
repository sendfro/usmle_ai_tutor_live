import streamlit as st
import json
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_community.vectorstores import Chroma

# --- 1. PAGE CONFIGURATION ---
st.set_page_config(page_title="The Interrogation Engine", layout="wide")

# --- 2. DUAL-ENGINE INITIALIZATION ---
@st.cache_resource
def load_engine():
    # ENGINE A: The STRICT Grader (Zero Creativity Lock)
    evaluator_llm = ChatOpenAI(
        api_key=st.secrets["OPENAI_API_KEY"],
        model="gpt-4o-mini", 
        temperature=0.0  
    )
    
    # ENGINE B: The CREATIVE Question Writer (For dynamic generation)
    generator_llm = ChatOpenAI(
        api_key=st.secrets["OPENAI_API_KEY"],
        model="gpt-4o-mini", 
        temperature=0.7  
    )
    
    # Connect to the V12 local vector database
    embeddings = OpenAIEmbeddings(api_key=st.secrets["OPENAI_API_KEY"])
    db = Chroma(persist_directory="chroma_db", embedding_function=embeddings)
    
    return evaluator_llm, generator_llm, db

evaluator_llm, generator_llm, vector_db = load_engine()

# --- 3. SESSION STATE MANAGEMENT ---
if "initial_answer" not in st.session_state:
    st.session_state.initial_answer = None
if "justification_provided" not in st.session_state:
    st.session_state.justification_provided = False
if "current_question" not in st.session_state:
    st.session_state.current_question = None

# --- 4. THE API QUESTION GENERATOR ---
def generate_new_question():
    prompt = """
    You are an expert USMLE medical board examiner. 
    Generate a brand new, challenging clinical vignette multiple-choice question.
    
    You MUST output ONLY a valid JSON object with this exact structure. Do not include markdown formatting, code blocks, or any other text:
    {
        "clinical_presentation": "A 34-year-old male presents with...",
        "options": ["A. Option 1", "B. Option 2", "C. Option 3", "D. Option 4"],
        "correct_answer": "C. Option 3"
    }
    """
    response = generator_llm.invoke(prompt)
    
    # Clean the response to ensure perfect JSON parsing
    clean_text = response.content.replace("```json", "").replace("```", "").strip()
    return json.loads(clean_text)

# --- 5. USER INTERFACE ---
st.title("The Interrogation Engine")
st.radio("Select Operating Mode:", ["Shift Mode", "Marathon Mode"], horizontal=True)
st.markdown("---")

# Generate a new question if the memory bank is empty
if st.session_state.current_question is None:
    with st.spinner("Forging a new clinical vignette via API..."):
        st.session_state.current_question = generate_new_question()

# Extract the current question data
clinical_presentation = st.session_state.current_question["clinical_presentation"]
options = st.session_state.current_question["options"]

st.markdown(f"**Clinical Presentation:** {clinical_presentation}")
user_choice = st.radio("Select best action:", options, index=None)

# Lock in the answer when selected
if user_choice and st.session_state.initial_answer != user_choice:
    st.session_state.initial_answer = user_choice
    st.session_state.justification_provided = False

# --- 6. THE EVALUATION LOGIC ---
if st.session_state.initial_answer:
    st.success(f"Answer '{st.session_state.initial_answer}' locked in. Processing strict evaluation...")
    
    if not st.session_state.justification_provided:
        with st.spinner("Consulting vector database & verifying facts..."):
            
            # Retrieve the medical facts from the 229MB database
            search_results = vector_db.similarity_search(clinical_presentation, k=10)
            context = "\n\n".join([doc.page_content for doc in search_results])
            
            # BULLETPROOF PYTHON GRADING (Bypassing AI logic)
            user_ans = st.session_state.initial_answer.strip()
            correct_ans = st.session_state.current_question["correct_answer"].strip()
            
            # Checking the exact string (or at least the first letter, e.g., 'A' vs 'A')
            if user_ans[0] == correct_ans[0]:
                grade_header = "✅ Correct!"
            else:
                grade_header = f"❌ Incorrect. The correct answer is {correct_ans}."
            
            # THE SOCRATIC PROMPT (Titanium Guardrails Installed)
            prompt = f"""
            You are an expert medical board examiner tutoring a student. 
            
            The student was given a clinical vignette. 
            They selected: '{user_ans}'.
            The actual correct answer is: '{correct_ans}'.
            
            Using ONLY the retrieved medical textbook context below, provide a Socratic explanation of WHY the correct answer is right, and why the student's choice (if they were wrong) is incorrect.
            
            Retrieved Textbook Context:
            {context} 

            CRITICAL GUARDRAILS:
            1. You must base your reasoning EXCLUSIVELY on the textbook context provided above.
            2. DO NOT use outside knowledge, extrapolate, deduce, or guess.
            3. Do not re-state whether the student is correct or incorrect, just provide the medical reasoning.
            4. If the retrieved context does not explicitly contain the clinical mechanics to explain the answer, you must strictly reply with: 'I cannot evaluate this. The retrieved context does not contain the necessary information.' Do not attempt to answer it yourself.
            """
            
            # Fire the zero-temperature engine
            actual_response = evaluator_llm.invoke(prompt)
            
            # Display the hardcoded grade AND the AI's strictly-bounded reasoning
            st.markdown(f"### {grade_header}")
            st.info(f"**Socratic Evaluation:**\n\n{actual_response.content}")
            
            st.session_state.justification_provided = True

st.markdown("---")

# --- 7. SYSTEM RESET ---
if st.button("Next Question"):
    # Wiping the memory forces the app to generate a fresh question from the API at the top
    st.session_state.current_question = None 
    st.session_state.initial_answer = None
    st.session_state.justification_provided = False
    st.rerun()