from datasets import Dataset

from .dataset import MiscDataset


class DummyDataset(MiscDataset):

    def __init__(self):
        data = {
            "input_ids": [
                [22244, 374, 279, 1176, 3187, 10684, 720],
                [1, 8586, 374, 2500, 3187, 10560, 1],
                [22244, 10550, 5727, 3892, 23719, 10560, 1],
                [1, 26021, 264, 10550, 449, 832, 3330],
                [
                    46639,
                    36368,
                    19109,
                    596,
                    30525,
                    6875,
                    374,
                ],
            ],
            "label": [
                [22244, 374, 279, 1176, 3187, 10684, 720],
                [1, 8586, 374, 2500, 3187, 10560, 1],
                [22244, 10550, 5727, 3892, 23719, 10560, 1],
                [1, 26021, 264, 10550, 449, 832, 3330],
                [46639, 36368, 19109, 596, 30525, 6875, 374],
            ],
            "text": [
                "This is the first example.", "Here is another example.",
                "This dataset contains several sentences.",
                "Creating a dataset with one column.",
                "Hugging Face's datasets library is convenient."
            ]
        }

        # Create a Dataset object
        dataset = Dataset.from_dict(data)
        self.split = dataset.train_test_split(test_size=0.0001)

    def train(self):
        return self.split["train"]

    def val(self):
        return self.split["test"]

    def test(self):
        assert False

    def get_data_collator(self):
        pass
