import streamlit as st
import moviepy.editor as mp 
import speech_recognition as sr 
import requests
import tempfile
from config import HF_API_KEY, HF_SUMMARIZATION_MODEL

def upload_video():
    uploaded_file = st.file_uploader("Upload Interview Video", type=["mp4", "mov"])
    return uploaded_file


def process_video(video_file):
    """
    Extract audio from video and return the audio text.
    """
    
    with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as temp_file:
        temp_file.write(video_file.read())
        temp_file_path = temp_file.name
    if temp_file_path is not None:
        video_clip = mp.VideoFileClip(temp_file_path)
        audio_path = tempfile.NamedTemporaryFile(delete=False, suffix=".wav").name
        video_clip.audio.write_audiofile(audio_path, codec='pcm_s16le')
        # Initialize recognizer
        video_clip.close() 
        r = sr.Recognizer() 

        # Load the audio file 
        with sr.AudioFile(audio_path) as source: 
            data = r.record(source) 

        # Convert speech to text 
        text = r.recognize_google(data) 
        print(text)
        return text
    return None

def summarize_interview(transcript):
    """Generate a summary of the interview using the Hugging Face model."""
    headers = {"Authorization": f"Bearer {HF_API_KEY}"}
    prompt = (
    f"Please summarize the following interview transcript in the categories provided below. Do not include any part of the prompt in your response.\n\n"
    f"**Transcript:**\n{transcript}\n\n"
    f"**Please provide your analysis as follows:**\n"
    f"- **Communication Style**:\n"
    f"- **Active Listening**:\n"
    f"- **Engagement with the Interviewer**:\n"
    f"**End of instructions.**"
)
    # since the summary model is not par with other LLMs. It also includes the prompt in its response which is a bug that can be fixed using a better model like GPT 4
    # Set up the JSON payload for the POST request
    data = {
        "inputs": prompt,
        "parameters": {
            "max_tokens": 500
        }
    }

    # Send the POST request to the Hugging Face API
    response = requests.post(HF_SUMMARIZATION_MODEL, headers=headers, json=data)

    # Check if the request was successful and parse the response
    if response.status_code == 200:
        # Parse JSON response
        result = response.json()
        # print("API Response:", json.dumps(result, indent=2))  # Print the response for inspection

        summary = result[0].get("generated_text", "No summary generated.") if isinstance(result, list) else result.get("generated_text", "No summary generated.")
        
        # Print the formatted summary
        print("**Interview Analysis Summary**")
        print(summary)
    else:
        print(f"Error: {response.status_code} - {response.text}")
    return summary

def analyze_interview():
    """
    Main function for analyzing the interview.
    """

    st.title("Interview Analysis Module")

    # Video Upload
    video_file = upload_video()

    if video_file is not None:
        st.video(video_file)
        
        # Process the video to extract audio
        transcript = process_video(video_file)
        st.success("Video processed successfully.")

        # Display the transcript
        # st.subheader("Transcript:")
        # st.write(transcript)

        summary = summarize_interview(transcript)
        if summary:
            st.subheader("Summary of Candidate's Responses:")
            st.write(summary)