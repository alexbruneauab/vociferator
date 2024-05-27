https://medium.com/@zahrizhalali/crafting-your-custom-text-to-speech-model-f151b0c9cba2


2024-05-27: Needs to replace lines in the library files to prevent decimal error while training
env/lib/python3.10/site-packages/TTS/tts/models/vits.py
- in function def _log, replace sample_voice = y_hat[0].squeeze(0).detach().cpu().numpy() for sample_voice = y_hat[0].squeeze(0).detach().cpu().float().numpy()

env/lib/python3.10/site-packages/TTS/vocoder/utils/generic_utils.py
- in function def plot_results, replace y_hat = y_hat[0].squeeze().detach().cpu().numpy() for y_hat = y_hat[0].squeeze().detach().cpu().float().numpy()