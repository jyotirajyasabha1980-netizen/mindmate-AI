
import streamlit as st
import librosa
import numpy as np
import io
import torch

from transformers import pipeline

# -------------------------------
# PAGE CONFIGURATION
# -------------------------------

st.set_page_config(
    page_title="MindMate AI",
    page_icon="🧠",
    layout="centered"
)

st.title("🧠 MindMate AI")
st.subheader("Your AI-powered study and academic support assistant")

st.write(
    "Share what's on your mind through your voice. "
    "MindMate will transcribe your speech, estimate vocal emotion "
    "and suggest academic support."
)

st.info(
    "This is an educational prototype. "
    "It does not diagnose mental health conditions."
)

# -------------------------------
# LOAD PRE-TRAINED MODELS
# -------------------------------

@st.cache_resource(show_spinner="Loading MindMate AI models...")
def load_models():

    speech_model = pipeline(
        "automatic-speech-recognition",
        model="openai/whisper-tiny.en",
        device=-1
    )

    emotion_model = pipeline(
        "audio-classification",
        model="superb/wav2vec2-base-superb-er",
        device=-1
    )

    return speech_model, emotion_model


# -------------------------------
# TOPIC DETECTION
# -------------------------------

def detect_topic(text):

    text = text.lower()

    topics = {
        "examination stress": [
            "exam", "examination", "test", "marks",
            "result", "board", "paper", "revision"
        ],

        "study planning": [
            "timetable", "schedule", "planning",
            "manage time", "organise", "organize"
        ],

        "academic pressure": [
            "pressure", "expectations", "burden",
            "overwhelmed", "stress", "competition"
        ],

        "subject difficulty": [
            "difficult", "understand", "physics",
            "chemistry", "biology", "maths",
            "subject", "chapter", "concept"
        ],

        "motivation": [
            "motivation", "lazy", "procrastination",
            "concentrate", "focus", "tired"
        ]
    }

    for topic, keywords in topics.items():
        if any(word in text for word in keywords):
            return topic

    return "general academic concern"


# -------------------------------
# MINDMATE RESPONSE SYSTEM
# -------------------------------

responses = {

    "examination stress":
        "Examinations can feel demanding. Try dividing "
        "your syllabus into smaller sections and revising "
        "one manageable topic at a time.",

    "study planning":
        "A realistic study schedule can make your workload "
        "easier to manage. Prioritise important topics and "
        "include short breaks.",

    "academic pressure":
        "Academic expectations can sometimes feel overwhelming. "
        "Focus on the progress you can make today rather than "
        "trying to solve everything at once.",

    "subject difficulty":
        "When a subject feels difficult, start with its basic "
        "concepts. Work through one example at a time and "
        "identify exactly where you need help.",

    "motivation":
        "Try beginning with a small study task. Completing "
        "one achievable goal can help you build momentum.",

    "general academic concern":
        "Thank you for sharing your concern. Try identifying "
        "the main academic challenge and breaking it into "
        "smaller steps. You can work through it gradually."
}


# -------------------------------
# AUDIO ANALYSIS
# -------------------------------

def analyse_audio(audio_bytes):

    speech_model, emotion_model = load_models()

    audio, sample_rate = librosa.load(
        io.BytesIO(audio_bytes),
        sr=16000,
        mono=True
    )

    if len(audio) == 0:
        raise ValueError("The audio recording is empty.")

    # Speech transcription
    transcription = speech_model({
        "raw": audio,
        "sampling_rate": sample_rate
    })

    text = transcription["text"].strip()

    if not text:
        raise ValueError("No speech could be recognised.")

    # Vocal emotion classification
    emotion_results = emotion_model({
        "raw": audio,
        "sampling_rate": sample_rate
    })

    emotion = emotion_results[0]["label"]

    # Academic topic detection
    topic = detect_topic(text)

    # Generate predefined response
    response = responses[topic]

    return text, emotion, topic, response


# -------------------------------
# USER AUDIO INPUT
# -------------------------------

st.divider()

st.header("🎙️ Talk to MindMate")

st.write("Record your voice or upload an audio recording.")

recording = st.audio_input("Record your voice")

uploaded_file = st.file_uploader(
    "Or upload an audio file",
    type=["wav", "mp3", "ogg", "flac", "m4a"]
)

audio_source = recording if recording is not None else uploaded_file

# -------------------------------
# DISPLAY RESULTS
# -------------------------------

if audio_source is not None:

    st.audio(audio_source)

    if st.button("Analyse my recording", type="primary"):

        try:
            with st.spinner(
                "MindMate is analysing your recording..."
            ):

                audio_bytes = audio_source.getvalue()

                text, emotion, topic, response = analyse_audio(
                    audio_bytes
                )

            st.success("Analysis completed!")

            st.subheader("📝 Speech Transcription")
            st.write(text)

            st.subheader("🎭 Detected Vocal Emotion")
            st.write(emotion)

            st.caption(
                "This is an estimated vocal emotion label, "
                "not a definitive assessment of how someone feels."
            )

            st.subheader("📚 Identified Academic Topic")
            st.write(topic.title())

            st.subheader("💡 MindMate's Response")
            st.write(response)

        except Exception as error:
            st.error(
                "Something went wrong while processing the audio."
            )
            st.caption(str(error))

st.divider()

st.caption(
    "MindMate AI | Class XII Artificial Intelligence Project"
)
