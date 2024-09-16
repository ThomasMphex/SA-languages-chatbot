import streamlit as st
from streamlit_chat import message
import numpy as np
import tensorflow as tf
import pickle
import speech_recognition as sr
from langchain import hub
import os
from deep_translator import GoogleTranslator
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import Chroma
from langchain.chains import RetrievalQA
from langchain.llms import Ollama
from langchain.embeddings import OllamaEmbeddings
from tensorflow.keras.preprocessing.sequence import pad_sequences

os.environ['LANGSMITH_API_KEY'] = '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'

# Load the language detection model and tokenizers
def load_language_detection_model(model_path, label_tokenizer_path, text_tokenizer_path):
    model = tf.keras.models.load_model(model_path)
    with open(label_tokenizer_path, 'rb') as f:
        label_tokenizer = pickle.load(f)
    with open(text_tokenizer_path, 'rb') as f:
        text_tokenizer = pickle.load(f)
    return model, label_tokenizer, text_tokenizer

language_detection_model, label_tokenizer, text_tokenizer = load_language_detection_model(
    'language_detection_model.h5',
    'label_tokenizer.pkl',
    'text_tokenizer.pkl'
)

# Mapping from the predicted labels to Google Translator language codes
language_code_map = {
    'xho': 'xh',
    'eng': 'en',
    'nso': 'nso',
    'ven': 've',
    'sot': 'st',
    'tsn': 'tn',
    'afr': 'af',
    'zul': 'zu',
    'ssw': 'ss',
    'tso': 'ts',
}

# Function to tokenize and predict the language of a given text
def predict_language(text):
    sequence = text_tokenizer.texts_to_sequences([text])
    padded_sequence = pad_sequences(sequence, maxlen=100)
    prediction = language_detection_model.predict(padded_sequence)
    label = label_tokenizer.index_word[np.argmax(prediction) + 1]
    google_lang_code = language_code_map.get(label, 'en')  # Default to 'en' if label not found
    return google_lang_code

# Function to translate the text
def translate_text(text, target_language='en'):
    try:
        translated_text = GoogleTranslator(source='auto', target=target_language).translate(text)
        if not translated_text:
            raise ValueError("Received None or incomplete result from the translation service.")
        return translated_text
    except Exception as e:
        print(f"An error occurred during translation: {e}")
        return None

# Function to modify the query with Absa-related context
def build_absa_related_query(user_query):
    absa_context = "Answer the query considering only information related to Absa South Africa."
    full_query = f"{absa_context} {user_query}"
    return full_query

# Function to process the text with Ollama
def process_text(text):
    try:
        lang_id = predict_language(text)

        # Translate the text to English if it's not already in English
        if lang_id != 'en':
            translated_text = translate_text(text, target_language='en')
            if translated_text is not None:
                text = translated_text
            else:
                return "Failed to translate text."

        # Modify the query to include Absa-related context
        full_query = build_absa_related_query(text)

        # Set up Ollama model and embeddings
        llm = Ollama(model="llama3.1", base_url="http://127.0.0.1:11434")
        embed_model = OllamaEmbeddings(model="llama3.1", base_url='http://127.0.0.1:11434')

        # Use a curated database or document set of Absa-related content
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=512, chunk_overlap=128)
        chunks = text_splitter.split_text(full_query)

        vector_store = Chroma.from_texts(chunks, embed_model)
        retriever = vector_store.as_retriever()

        # Set up RetrievalQA chain with the query limited to Absa content
        retrieval_chain = RetrievalQA.from_chain_type(
            llm=llm,
            retriever=retriever,
            chain_type="stuff"
        )

        response = retrieval_chain({"query": full_query})
        answer = response.get('result', 'No relevant Absa-related information found.')

        # Translate the response back to the original language (if needed)
        if lang_id != 'en':
            response_translation = translate_text(answer, target_language=lang_id)
            if response_translation is not None:
                return response_translation
            else:
                return "Failed to translate the response."
        else:
            return answer

    except Exception as e:
        return f"Processing error: {e}"

# Speech recognition function with automatic language detection
def recognize_and_detect_language():
    r = sr.Recognizer()

    with sr.Microphone() as source:
        st.write("Say something...")
        audio = r.listen(source)

    try:
        # Recognize speech using Google Web Speech API (English by default for speech recognition)
        text = r.recognize_google(audio)
        st.write(f"Recognized Speech: {text}")

        # Automatically detect the language using the language detection model
        detected_language_code = predict_language(text)
        st.write(f"Detected Language: {detected_language_code}")

        return text, detected_language_code
    except sr.UnknownValueError:
        st.write("Google Speech Recognition could not understand the audio.")
        return None, None
    except sr.RequestError as e:
        st.write(f"Could not request results from Google Speech Recognition service; {e}")
        return None, None

# Streamlit chatbot interface
st.title("Abby - Lingual")
st.markdown("Chat with Abby to retrieve information about Absa South Africa.")

# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Add initial bot message asking how it can help the user
if not st.session_state.messages:
    st.session_state.messages.append({"role": "bot", "text": "Hello! I'm Abby, a multilingual chatbot here to assist you with information about Absa South Africa. You are welcome to chat with me in any language. How can I help you today?", "key": "initial_bot_message"})

# Display chat history
for message_data in st.session_state.messages:
    message_key = message_data.get("key", f"{message_data['role']}_{len(st.session_state.messages)}")
    if message_data["role"] == "user":
        message(message_data["text"], is_user=True, key=message_key)
    else:
        message(message_data["text"], is_user=False, key=message_key)

# Create a column layout to position the record button
col1, col2 = st.columns([100,1])
with col1:
    st.write("")
with col2:
    if st.button("🎤", key="record_button"):
        # Get the recognized speech and detected language automatically
        speech_query, detected_language = recognize_and_detect_language()

        if speech_query and detected_language:
            # Add user message (via voice) to chat history
            st.session_state.messages.append({"role": "user", "text": speech_query, "key": f"user_{len(st.session_state.messages)}"})

            # Process the user input
            bot_response = process_text(speech_query)

            # Add bot response to chat history
            st.session_state.messages.append({"role": "bot", "text": bot_response, "key": f"bot_{len(st.session_state.messages)}"})

            # Rerun to update chat history
            st.experimental_rerun()

# Handle chat input
user_input = st.chat_input("What is up?", key="chat_input")
if user_input:
    # Add user message (via chat input) to chat history
    st.session_state.messages.append({"role": "user", "text": user_input, "key": f"user_{len(st.session_state.messages)}"})

    # Process the user input
    bot_response = process_text(user_input)

    # Add bot response to chat history
    st.session_state.messages.append({"role": "bot", "text": bot_response, "key": f"bot_{len(st.session_state.messages)}"})

    # Rerun to update chat history
    st.experimental_rerun()
