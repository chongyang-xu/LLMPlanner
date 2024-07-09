from abc import ABC, abstractmethod


class Dataset(ABC):

    @abstractmethod
    def train(self):
        pass

    @abstractmethod
    def val(self):
        pass

    @abstractmethod
    def test(self):
        pass

    @abstractmethod
    def get_data_collator(self):
        pass


# https://huggingface.co/docs/trl/en/sft_trainer#dataset-format-support
class ConversationalDataset(Dataset):

    def format_checking(self, dataset, check_all=False):
        check_n = len(dataset) if check_all else 10
        counter = 0
        while counter < check_n:
            counter += 1
            row = dataset[counter]
            assert 'messages' in row
            for content in row['messages']:
                assert 'content' in content
                assert 'role' in content


# https://huggingface.co/docs/trl/en/sft_trainer#dataset-format-support
class InstructionDataset(Dataset):

    def format_checking(self, dataset, check_all=False):
        check_n = len(dataset) if check_all else 10
        counter = 0
        while counter < check_n:
            counter += 1
            row = dataset[counter]
            assert 'prompt' in row
            assert 'completion' in row


class MiscDataset(Dataset):
    pass
