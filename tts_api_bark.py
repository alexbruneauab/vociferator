#1. Using the Bark library directly
#https://github.com/suno-ai/bark
from bark import SAMPLE_RATE, generate_audio, preload_models
from scipy.io.wavfile import write as write_wav

# download and load all models
preload_models()

# generate audio from text
text_prompt = """
     Hello, my name is Suno. And, uh — and I like pizza. [laughs] 
     But I also have other interests such as playing tic tac toe.
"""
audio_array = generate_audio(text_prompt)

# save audio to disk
write_wav("/home/py-projects/vociferator/data/output/output.wav", SAMPLE_RATE, audio_array)

##2. Using the Coqui TTS library
##https://docs.coqui.ai/en/latest/models/bark.html
# from TTS.api import TTS

# # Init TTS
# tts = TTS("tts_models/multilingual/multi-dataset/bark", gpu=False)

# # Run TTS

# # # Cloning a new speaker
# # # This expects to find a mp3 or wav file like `bark_voices/new_speaker/speaker.wav`
# # # It computes the cloning values and stores in `bark_voices/new_speaker/speaker.npz`
# # tts.tts_to_file(text="Hello, my name is Manmay , how are you?",
# #                 file_path="output.wav",
# #                 voice_dir="bark_voices/",
# #                 speaker="ljspeech")


# # # When you run it again it uses the stored values to generate the voice.
# # tts.tts_to_file(text="Hello, my name is Manmay , how are you?",
# #                 file_path="output.wav",
# #                 voice_dir="bark_voices/",
# #                 speaker="ljspeech")

# # random speaker
# tts.tts_to_file("Hello, my name is Manmay , how are you?", file_path="output1.wav")

