import torch
from pytorch_lightning import LightningModule, Trainer
import torchaudio
from torchaudio.models import Tacotron2
from torch.utils.data import DataLoader
from torch.nn.utils.rnn import pad_sequence

#Tacotron2TTS class: This class inherits from the LightningModule class, which provides a number of helper functions for training and evaluating deep learning models. 
#Initialize the model and the configuration in constructor.
class Tacotron2TTS(LightningModule):

    def __init__(self, cfg):
        super().__init__()
        self.model = Tacotron2(cfg)
        self.dataset = torchaudio.datasets.LJSPEECH(root="/home/py-projects/vociferator/data/",download = True)

    #Forward Pass: This function takes a text input and returns a mel spectrogram.
    def forward(self, text):
        mel_spectrogram = self.model(text)
        return mel_spectrogram

    #Training step: This function takes a batch of data and calculates the loss. The loss is then logged to the console and TensorBoard.
    def training_step(self, batch, batch_idx):
        text, mel_spectrogram = batch
        mel_spectrogram_pred = self(text)
        loss = torch.nn.functional.mse_loss(mel_spectrogram_pred, mel_spectrogram)
        self.log("train_loss", loss)
        return loss

    #Validation step: This function is similar to the training step, but it uses the validation dataset
    def validation_step(self, batch, batch_idx):
        text, mel_spectrogram = batch
        mel_spectrogram_pred = self(text)
        loss = torch.nn.functional.mse_loss(mel_spectrogram_pred, mel_spectrogram)
        self.log("val_loss", loss)

    #Test step: This function is similar to the validation step, but it uses the test dataset
    def test_step(self, batch, batch_idx):
        text, mel_spectrogram = batch
        mel_spectrogram_pred = self(text)
        loss = torch.nn.functional.mse_loss(mel_spectrogram_pred, mel_spectrogram)
        self.log("test_loss", loss)

    #optimizer: It is used to update the model’s parameters after each training step.
    def configure_optimizers(self):
        return torch.optim.Adam(self.model.parameters(), lr=1e-3)

    def collate_fn(self, batch):
        # separate the text and mel_spectrogram in the batch
        texts, mel_spectrograms = zip(*batch)

        # pad the sequences in the batch to be the same length
        texts = pad_sequence([torch.tensor(t) for t in texts], batch_first=True)
        mel_spectrograms = pad_sequence([torch.tensor(m) for m in mel_spectrograms], batch_first=True)

        return texts, mel_spectrograms

    def train_dataloader(self):
        return DataLoader(self.dataset, batch_size=32, shuffle=True, collate_fn=self.collate_fn)

#create the configuration, model and trainer object
if __name__ == "__main__":
    cfg = {
        "model": {
            "num_mels": 80,
            "hidden_channels": 128,
            "attention_dim": 128,
        },
        "trainer": {
            "max_epochs": 100,
        },
    }

    model = Tacotron2TTS(cfg)
    trainer = Trainer()
    trainer.fit(model)