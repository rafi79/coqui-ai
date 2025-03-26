import streamlit as st
import torch
import os
import tempfile
import base64
from TTS.api import TTS

st.set_page_config(page_title="Coqui TTS Voice Generator", layout="wide")

st.title("🐸 Coqui TTS Voice Generator")
st.markdown("Generate speech with male and female voices using Coqui TTS")

# Get device
device = "cuda" if torch.cuda.is_available() else "cpu"
st.sidebar.info(f"Using device: {device}")

@st.cache_resource
def load_tts_model(model_name):
    try:
        return TTS(model_name).to(device)
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None

def get_available_models():
    try:
        # Initialize TTS to get model list
        all_models = TTS().list_models()
        # Filter for models that can likely be used without a reference file
        single_speaker_models = [model for model in all_models if "tts_models" in model 
                                and not any(x in model for x in ["your_tts", "xtts"])]
        # Get multilingual models for voice cloning
        cloning_models = [model for model in all_models if "tts_models" in model 
                         and any(x in model for x in ["your_tts", "xtts", "multi-dataset"])]
        
        return {
            "single_speaker": single_speaker_models,
            "voice_cloning": cloning_models
        }
    except Exception as e:
        st.error(f"Error getting models: {e}")
        return {"single_speaker": [], "voice_cloning": []}

def autoplay_audio(audio_data):
    b64 = base64.b64encode(audio_data).decode()
    md = f"""
        <audio autoplay controls>
        <source src="data:audio/wav;base64,{b64}" type="audio/wav">
        </audio>
    """
    st.markdown(md, unsafe_allow_html=True)

def get_binary_file_downloader_html(bin_data, file_label='File'):
    b64 = base64.b64encode(bin_data).decode()
    button_uuid = f'download_button_{file_label}'
    custom_css = f"""
        <style>
            #{button_uuid} {{
                background-color: rgb(255, 255, 255);
                color: rgb(38, 39, 48);
                padding: 0.5em 0.7em;
                position: relative;
                text-decoration: none;
                border-radius: 4px;
                border-width: 1px;
                border-style: solid;
                border-color: rgb(230, 234, 241);
                border-image: initial;
            }}
            #{button_uuid}:hover {{
                border-color: rgb(246, 51, 102);
                color: rgb(246, 51, 102);
            }}
        </style>
    """
    dl_link = custom_css + f'<a href="data:application/octet-stream;base64,{b64}" download="{file_label}" id="{button_uuid}">Download {file_label}</a><br></br>'
    return dl_link

# Get available models
models = get_available_models()

# Sidebar for model selection
st.sidebar.header("Model Selection")
mode = st.sidebar.radio("Select mode", ["Pre-trained voices", "Voice cloning"])

if mode == "Pre-trained voices":
    model_name = st.sidebar.selectbox("Select a model", models["single_speaker"])
    
    # Initialize model
    if model_name:
        with st.spinner("Loading model..."):
            tts_model = load_tts_model(model_name)
            
        if tts_model:
            st.success(f"Model loaded: {model_name}")
            
            # Check if model has multiple speakers
            speakers = []
            try:
                if hasattr(tts_model, "speakers") and tts_model.speakers:
                    speakers = tts_model.speakers
                    st.sidebar.subheader("Available Speakers")
                    speaker_idx = st.sidebar.selectbox("Select speaker", speakers)
            except:
                pass
            
            # User input
            text_input = st.text_area("Enter text to synthesize", "Hello! This is Coqui TTS. I can speak in different voices!")
            
            if st.button("Generate Speech"):
                with st.spinner("Generating audio..."):
                    try:
                        # Create a temporary file
                        with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as fp:
                            temp_filename = fp.name
                        
                        # Generate speech
                        if speakers and speaker_idx:
                            tts_model.tts_to_file(text=text_input, file_path=temp_filename, speaker=speaker_idx)
                        else:
                            tts_model.tts_to_file(text=text_input, file_path=temp_filename)
                        
                        # Load and display audio
                        with open(temp_filename, "rb") as f:
                            audio_bytes = f.read()
                        
                        st.subheader("Generated Audio")
                        st.audio(audio_bytes, format="audio/wav")
                        
                        # Add download button
                        st.markdown(get_binary_file_downloader_html(audio_bytes, 'generated_speech.wav'), unsafe_allow_html=True)
                        
                        # Clean up
                        os.unlink(temp_filename)
                    
                    except Exception as e:
                        st.error(f"Error generating speech: {e}")

else:  # Voice cloning mode
    model_name = st.sidebar.selectbox("Select a voice cloning model", models["voice_cloning"])
    languages = ["en", "es", "fr", "de", "it", "pt", "pl", "tr", "ru", "nl", "cs", "ar", "zh-cn", "ja"]
    selected_language = st.sidebar.selectbox("Select language", languages)
    
    # Initialize model
    if model_name:
        with st.spinner("Loading model..."):
            tts_model = load_tts_model(model_name)
            
        if tts_model:
            st.success(f"Model loaded: {model_name}")
            
            st.subheader("Upload Reference Voice")
            st.write("Upload audio samples for male and female voices to clone")
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("### Male Voice")
                male_file = st.file_uploader("Upload male voice sample (WAV format)", type=["wav"])
                male_sample_path = None
                
                if male_file:
                    # Save uploaded file temporarily
                    with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as m_file:
                        m_file.write(male_file.getvalue())
                        male_sample_path = m_file.name
                    st.audio(male_file, format="audio/wav")
            
            with col2:
                st.markdown("### Female Voice")
                female_file = st.file_uploader("Upload female voice sample (WAV format)", type=["wav"])
                female_sample_path = None
                
                if female_file:
                    # Save uploaded file temporarily
                    with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as f_file:
                        f_file.write(female_file.getvalue())
                        female_sample_path = f_file.name
                    st.audio(female_file, format="audio/wav")
            
            # Text input
            text_input = st.text_area("Enter text to synthesize", "Hello! This is cloned voice speaking. How do I sound?")
            
            # Select voice to use
            voice_type = st.radio("Select voice to use", ["Male", "Female"])
            
            if st.button("Generate Speech"):
                voice_sample_path = male_sample_path if voice_type == "Male" else female_sample_path
                
                if voice_sample_path:
                    with st.spinner("Generating audio with cloned voice..."):
                        try:
                            # Create a temporary file for output
                            with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as fp:
                                temp_filename = fp.name
                            
                            # Generate speech with cloned voice
                            tts_model.tts_to_file(
                                text=text_input,
                                file_path=temp_filename,
                                speaker_wav=voice_sample_path,
                                language=selected_language
                            )
                            
                            # Load and display audio
                            with open(temp_filename, "rb") as f:
                                audio_bytes = f.read()
                            
                            st.subheader("Generated Audio")
                            st.audio(audio_bytes, format="audio/wav")
                            
                            # Add download button
                            st.markdown(get_binary_file_downloader_html(audio_bytes, f'{voice_type.lower()}_cloned_speech.wav'), unsafe_allow_html=True)
                            
                            # Clean up
                            os.unlink(temp_filename)
                            os.unlink(voice_sample_path)
                        
                        except Exception as e:
                            st.error(f"Error generating speech: {e}")
                else:
                    st.error(f"Please upload a {voice_type.lower()} voice sample first")

st.sidebar.markdown("---")
st.sidebar.subheader("About")
st.sidebar.info(
    """
    This app uses 🐸 Coqui TTS, an open-source text-to-speech system.
    
    You can use pre-trained models for various languages or clone your own voice by uploading a sample.
    """
)
