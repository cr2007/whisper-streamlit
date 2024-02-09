import os
import pathlib
import streamlit as st
import whisper

# Choose model to use by uncommenting
# modelName = "tiny.en"
modelName: str = "base.en"
model: whisper.Whisper = None
# modelName = "small.en"
# modelName = "medium.en"
# modelName = "large-v2"

# Other Variables
exportTimestampData = True  # (bool) Whether to export the segment data to a json file. Will include word level timestamps if word_timestamps is True.
outputFolder = "output"

#  ----- Select variables for transcribe method  -----
# audio: path to audio file
verbose = True  # (bool): Whether to display the text being decoded to the console. If True, displays all the details, If False, displays minimal details. If None, does not display anything
language = "english"  # Language of audio file
word_timestamps = True  # (bool): Extract word-level timestamps using the cross-attention pattern and dynamic time warping, and include the timestamps for each word in each segment.
# initial_prompt="" # (optional str): Optional text to provide as a prompt for the first window. This can be used to provide, or "prompt-engineer" a context for transcription, e.g. custom vocabularies or proper nouns to make it more likely to predict those word correctly.


@st.cache_resource(show_spinner=False)
def load_whisper_model(modelName: str) -> whisper.Whisper:
    try:
        global model
        model = whisper.load_model(modelName)
        print("Model Loaded")
        print("-------------------------")
        return model
    except Exception as e:
        st.error(f"Failed to load Whisper model: {e}", icon="❌")

def main():
    st.header("Whisper Tool")

    global modelName
    result = ""

    modelName = st.selectbox(
        label="Select Model",
        options=["base.en", "small.en", "medium.en", "large-v2"],
        # index=1,
        help="Select the model to use for transcription",
    )

    audio_file = st.file_uploader(
        "Upload a file",
        accept_multiple_files=False,
        type=["mp3", "mp4", "mpeg", "mpga", "m4a", "wav", "webm"],
        key="audio_uploader",
    )

    if audio_file:
        st.success("File uploaded successfully", icon="✅")
        st.audio(audio_file, format="audio/mp3")
        if st.button("Transcribe", type="primary"):
            # Get filename stem using pathlib (filename without extension)
            fileNameStem = pathlib.Path(audio_file.name).stem

            resultFileName = f"{fileNameStem}.txt"

            with st.spinner("Loading model..."):
                model = load_whisper_model(modelName)
            st.success(f"Model {modelName} loaded successfully", icon="✅")

            with st.spinner("Saving file..."):
                if not os.path.exists(outputFolder):
                    os.makedirs(outputFolder)
                with open(os.path.join(outputFolder, audio_file.name), "wb") as file:
                    file.write(audio_file.getbuffer())
            st.success("File saved successfully", icon="✅")

            with st.spinner("Transcribing..."):
                result = model.transcribe(
                    audio=f"{outputFolder}/{audio_file.name}",
                    language=language,
                    word_timestamps=word_timestamps,
                    verbose=verbose,
                )
            st.success("Transcription complete", icon="✅")

            # remove audio file after transcription
            os.remove(f"{outputFolder}/{audio_file.name}")

    if result:
        with open(
            os.path.join(outputFolder, resultFileName), "w", encoding="utf-8"
        ) as file:
            file.write(result["text"])

        st.download_button(
            label="Export Transcription",
            type="primary",
            file_name=resultFileName,
            data=result["text"],
            mime="text/plain",
            use_container_width=True
        )

        st.markdown(f"Transcription:\n {result['text']}")



if __name__ == "__main__":
    main()
