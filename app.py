import streamlit as st
import json
import random
from langchain_community.vectorstores import Chroma
# CHANGED: We now import OpenAIEmbeddings instead of Ollama
from langchain_openai import ChatOpenAI, OpenAIEmbeddings

# --- 1. SESSION STATE INITIALIZATION ---
if "initial_answer" not in st.session_state:
    st.session_state.initial_answer = None
if "justification_provided" not in st.session_state:
    st.session_state.justification_provided = False
if "current_question" not in st.session_state:
    st.session_state.current_question = None

# --- 2. CLOUD-READY DUAL-ENGINE SETUP ---
@st.cache_resource
def load_database():
    # CHANGED: Using OpenAI embeddings and pulling the key securely from st.secrets
    embeddings = OpenAIEmbeddings(
        api_key=st.secrets["OPENAI_API_KEY"], 
        model="text-embedding-3-small"
    )
    db = Chroma(persist_directory="./chroma_db", embedding_function=embeddings)
    return db

vector_db = load_database()

# Engine A: Creative (Locked to JSON output)
# CHANGED: API key is now secured via st.secrets
generator_llm = ChatOpenAI(
    api_key=st.secrets["OPENAI_API_KEY"], 
    model="gpt-4o-mini",
    temperature=0.7,
    model_kwargs={"response_format": {"type": "json_object"}}
)

# Engine B: Strict (For grading safely)
# CHANGED: API key is now secured via st.secrets
evaluator_llm = ChatOpenAI(
    api_key=st.secrets["OPENAI_API_KEY"], 
    model="gpt-4o-mini",
    temperature=0.0 
)

# --- 3. TRUE REAL-TIME QUESTION GENERATOR ---
def generate_new_question():
    topics = ["Cardiology", "Neurology", "Gastroenterology", "Renal", "Endocrine", "Hematology", "Infectious Disease", "Pulmonology"]
    chosen_topic = random.choice(topics)
    
    docs = vector_db.similarity_search(f"USMLE high yield {chosen_topic}", k=3)
    context = "\n".join([doc.page_content for doc in docs])
    
    prompt = f"""
    You are an expert medical board exam writer. 
    Read this textbook context: {context}
    
    Generate ONE multiple-choice clinical vignette question based strictly on this text.
    Return ONLY a valid JSON object in this exact format:
    {{
        "text": "A [age]-year-old [gender] presents with...",
        "options": ["A. [option]", "B. [option]", "C. [option]", "D. [option]"],
        "answer": "B. [option]"
    }}
    """
    
    response = generator_llm.invoke(prompt)
    raw_text = response.content.strip().replace('```json', '').replace('```', '')
    return json.loads(raw_text)

# Generate or load the current question
if st.session_state.current_question is None:
    with st.spinner("Generating a unique clinical scenario from your textbooks..."):
        try:
            st.session_state.current_question = generate_new_question()
        except Exception as e:
            st.error(f"Engine Misfire: {e}") 
            st.warning("Click 'Next Question' to spin up a new scenario.")
            st.stop()

# Extract question details from memory
question_data = st.session_state.current_question
question_text = question_data["text"]
options = question_data["options"]
correct_answer = question_data["answer"]

# --- 4. FRONTEND UI ---
st.title("The Interrogation Engine")
st.radio("Select Operating Mode:", ["Shift Mode", "Marathon Mode"], horizontal=True)
st.markdown("---")

st.write(f"**Clinical Presentation:** {question_text}")

# Handle selected answer UI state
current_index = options.index(st.session_state.initial_answer) if st.session_state.initial_answer in options else None
selected_answer = st.radio("Select best action:", options, index=current_index)

# --- 5. STRICT EVALUATION ---
if selected_answer and st.session_state.initial_answer is None:
    st.session_state.initial_answer = selected_answer
    st.rerun()

if st.session_state.initial_answer:
    st.success(f"Answer '{st.session_state.initial_answer}' locked in. Processing strict evaluation...")
    
    if not st.session_state.justification_provided:
        with st.spinner("Consulting vector database & verifying facts..."):
            
            user_answer = st.session_state.initial_answer
            docs = vector_db.similarity_search(question_text, k=6)
            medical_context = "\n".join([doc.page_content for doc in docs])
            
            prompt = f"""
            You are a strict, clinical USMLE tutor.
            
            Clinical Case: {question_text}
            Student's Chosen Answer: {user_answer}
            The Actual Correct Answer is: {correct_answer}

            Medical Textbook Context: 
            {medical_context}

            CRITICAL SAFETY INSTRUCTIONS:
            1. You must explain the correct answer using ONLY the facts present in the Medical Textbook Context above.
            2. If the student is INCORRECT, explicitly state: "Incorrect. The correct answer is {correct_answer}."
            3. If the provided context does not explicitly verify the mechanism of action or clinical link for these options, say EXACTLY: "I can confirm the correct answer is {correct_answer}, but my local medical database context is missing the specific text required to fully explain this mechanism."
            4. DO NOT extrapolate, speculate, or bring in outside medical assumptions not written in the context.
            """
            
            actual_response = evaluator_llm.invoke(prompt)
            st.info(f"**Socratic Evaluation:**\n\n{actual_response.content}")
            st.session_state.justification_provided = True

st.markdown("---")

# --- 6. SYSTEM RESET ---
if st.button("Next Question"):
    st.session_state.initial_answer = None
    st.session_state.justification_provided = False
    st.session_state.current_question = None 
    st.rerun()