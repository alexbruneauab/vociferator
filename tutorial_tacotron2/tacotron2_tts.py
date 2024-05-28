import torch
from pytorch_lightning import LightningModule, Trainer
import torchaudio
from torchaudio.models import Tacotron2
from torch.utils.data import DataLoader

dataset = torchaudio.datasets.LJSPEECH(root="/home/py-projects/vociferator/data/",download = False)

class Tacotron2TTS(LightningModule):

    def __init__(self, cfg):
        super().__init__()
        self.model = Tacotron2(cfg)
        self.dataset = dataset

    def forward(self, text):
        t = torch.tensor([ord(c) for c in text])
        mel_spectrogram = self.model(t)
        return mel_spectrogram

    def training_step(self, batch, batch_idx):
        tensors, sample_rates, strings1, strings2 = batch
        mel_spectrogram_pred = self(strings1[0])
        loss = torch.nn.functional.mse_loss(mel_spectrogram_pred, tensors[0])
        self.log("train_loss", loss)
        return loss

    def validation_step(self, batch, batch_idx):
        tensors, sample_rates, strings1, strings2 = batch
        mel_spectrogram_pred = self(strings1[0])
        loss = torch.nn.functional.mse_loss(mel_spectrogram_pred, tensors[0])
        self.log("val_loss", loss)

    def test_step(self, batch, batch_idx):
        tensors, sample_rates, strings1, strings2 = batch
        mel_spectrogram_pred = self(strings1[0])
        loss = torch.nn.functional.mse_loss(mel_spectrogram_pred, tensors[0])
        self.log("test_loss", loss)

    def configure_optimizers(self):
        return torch.optim.Adam(self.model.parameters(), lr=1e-3)

if __name__ == "__main__":
    cfg = {
        "model": {
            "num_mels": 80,
            "hidden_channels": 128,
            "attention_dim": 128,
        },
        "trainer": {
            "max_epochs": 10,
        },
    }

    dataloader = DataLoader(dataset)
    model = Tacotron2TTS(cfg)
    trainer = Trainer(accelerator="cpu")
    trainer.fit(model=model, train_dataloaders=dataloader)