import torch
from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor


class ModelWrapper:
    """Helper class used to wrap Wav2Vec2 models."""

    def __init__(self, model_uri):
        self.processor = Wav2Vec2Processor.from_pretrained(model_uri)
        self.model = Wav2Vec2ForCTC.from_pretrained(model_uri)

    @staticmethod
    def split_phonemes(sentence):
        return sentence[1:-1].replace("][", " ")

    def predict(self, speech, sampling_rate):
        input_values = self.processor(
            speech, sampling_rate=sampling_rate, return_tensors="pt"
        ).input_values

        with torch.no_grad():
            logits = self.model(input_values).logits

        pred_ids = torch.argmax(logits, dim=-1)
        pred_str = ModelWrapper.split_phonemes(
            self.processor.batch_decode(pred_ids)[0]
        )
        return pred_str
