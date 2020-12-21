import config
import torch

class BERTDataset:
    def __init__(self, description, points):
        self.description = description
        self.points = points
        self.tokenizer = config.TOKENIZER
        self.max_len = config.MAX_LEN

    def __len__(self):
        return len(self.description)

    def __getitem__(self, item):
        description = str(self.description[item])
        description = " ".join(description.split())

        inputs = self.tokenizer.encode_plus(
            description,
            None,
            add_special_tokens=True,
            max_length=self.max_len,
            padding="max_length",
            truncation=True,
        )
        ids = inputs["input_ids"]
        mask = inputs["attention_mask"]

        return {
            "ids": torch.tensor(ids, dtype=torch.long),
            "mask": torch.tensor(mask, dtype=torch.long),
            "points": torch.tensor(self.points[item], dtype=torch.long),
        }
