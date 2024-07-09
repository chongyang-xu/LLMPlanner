from datasets import load_dataset

from .dataset import ConversationalDataset


class Dolly15kOAI(ConversationalDataset):

    def __init__(self):
        self.dataset = load_dataset("philschmid/dolly-15k-oai-style",
                                    split="train")
        self.split = self.dataset.train_test_split(test_size=0.001)

    def train(self):
        self.format_checking(self.split["train"])
        return self.split["train"]

    def val(self):
        self.format_checking(self.split["test"])
        return self.split["test"]

    def test(self):
        assert False

    def get_data_collator(self):
        pass
