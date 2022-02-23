import os  # file paths
import pandas as pd  # read_csv
import spacy
import torch
from torch.utils.data import DataLoader, Dataset
from torch.nn.utils.rnn import pad_sequence  # padding batches
import torchvision.transforms as T

spacy_eng = spacy.load("en")

class Vocabulary:
    def __init__(self, freq_threshold):
        self.itos = {0:"<PAD>", 1:"<SOS>", 2:"<EOS>", 3:"<UNK>"}
        self.stoi = {"<PAD>":0, "<SOS>":1, "<EOS>":2, "<UNK>":3}
        self.freq_threshold = freq_threshold

    def __len__(self):
        return len(self.itos)

    @staticmethod
    def tokenizer_en(text):
        return [tok.text.lower() for tok in spacy_eng.tokenizer(text)]

    def build_vocabulary(self, sentence_list):
        frequencies = {}
        idx = 4
        for sentence in sentence_list:
            for word in self.tokenizer_en(sentence):
                if word not in frequencies:
                    frequencies[word] = 1

                else:
                    frequencies[word] += 1

                if frequencies[word] == self.freq_threshold:
                    self.stoi[word] = idx
                    self.itos[idx] = word
                    idx += 1

    def numericalize(self, text):
        tokenized_text = self.tokenizer_en(text)

        return[
            self.stoi[token] if token in self.stoi else self.stoi["<UNK>"]
            for token in tokenized_text
        ]

class FPBdataset(Dataset):
    def __init__(self, root_dir, file_path, transform=None, freq_threshold=5):
        self.root_dir = root_dir
        self.df = pd.read_csv(file_path, delimiter=',', encoding='latin-1')
        self.transform = transform

        # small changes to input file:
        # rename first row to column titles
        # could add a row above but lazy
        self.df = self.df.rename(columns={'neutral': 'sentiment',
                                'According to Gran , the company has no plans to move all production to Russia , although that is where the company is growing .': 'message'})

        # change string sentiments to numeric
        sentiment = {'positive': 0, 'neutral': 1, 'negative': 2}
        self.df.sentiment = [sentiment[item] for item in self.df.sentiment]

        # sanity
        print("Head:")
        print(self.df[0:5])

        self.label = self.df["sentiment"]
        self.data = self.df["message"]

        self.vocab = Vocabulary(freq_threshold)
        self.vocab.build_vocabulary(self.data.tolist())

    def __len__(self):
        return len(self.df)

    def __getitem__(self, index):
        label = self.label[index]
        data = self.data[index]

        if self.transform is not None:
            data = self.transform(data)

        numericalized_data = [self.vocab.stoi["<SOS>"]]
        numericalized_data += self.vocab.numericalize(data)
        numericalized_data.append(self.vocab.stoi["<EOS>"])

        return label, torch.tensor(numericalized_data)

class myCollate:
    def __init__(self, pad_idx):
        self.pad_idx = pad_idx

    def __call__(self, batch):
        # labels = [item[0].unsqueeze(0) for item in batch]  # unsqueeze for batch dim
        # labels = torch.cat(labels, dim=0)
        labels = [item[0] for item in batch]
        targets = [item[1] for item in batch]
        targets = pad_sequence(targets, batch_first=True, padding_value=self.pad_idx)

        return labels, targets

def get_loader(
        root_dir,
        file_path,
        transform,
        batch_size=32,
        num_workers=1,
        shuffle=True,
        pin_memory=True,
):
    dataset = FPBdataset(root_dir, file_path, transform=transform)

    pad_idx = dataset.vocab.stoi["<PAD>"]

    loader = DataLoader(
        dataset=dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        shuffle=shuffle,
        pin_memory=pin_memory,
        collate_fn = myCollate(pad_idx=pad_idx),
    )

    return loader

def main():
    transform = T.Compose([T.ToTensor()])
    dataloader = get_loader("./", file_path="data/all-data.csv", transform=None)

    for idx, (labels, data) in enumerate(dataloader):
        print(labels)
        print(data.shape)

if __name__ == "__main__":
    main()
