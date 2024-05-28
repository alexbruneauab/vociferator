Source: https://docs.coqui.ai/en/latest/training_a_model.html

The trainer generates a model in the data folder. To use the generated model, hit this command in the terminal and change the run-{generated_Date} folder by the real name:
tts --text "hello world! this is a demo" --model_path /home/py-projects/vociferator/data/run-{generated_Date}/best_model.pth --config_path /home/py-projects/vociferator/data/run-{generated_Date}/config.json --out_path /home/py-projects/vociferator/data/outp
ut/test.wav