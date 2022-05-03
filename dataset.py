class NewsSummaryDataset(Dataset):
    def __init__(self, data, tokenizer: T5Tokenizer, title_max_len: int = 64, maintext_max_len:int = 2048):
        self.data = data
        self.tokenizer = tokenizer
        self.title_max_len = title_max_len
        self.maintext_max_len = maintext_max_len

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index: int):
        sample = self.data[index]
        title = sample["title"]
        title_encoding = self.tokenizer(
            title,
            max_length=self.title_max_len,
            padding="max_length",
            truncation=True,
            return_attention_mask=True,
            add_special_tokens=True,
            return_tensors="pt"
        )
        labels = title_encoding["input_ids"]
        # Change padding element 0 to -100
        #   In addition, we must make sure that padding token id’s of the labels are not taken into account by the loss function. 
        #   In PyTorch and Tensorflow, this can be done by replacing them with -100, which is the ignore_index of the CrossEntropyLoss.
        labels[labels == 0] = -100 

        maintext = sample["maintext"]
        maintext_encoding = self.tokenizer(
            maintext,
            max_length=self.title_max_len,
            padding="max_length",
            truncation=True,
            return_attention_mask=True,
            add_special_tokens=True,
            return_tensors="pt"
        )

        return {
            #"maintext": maintext,
            "title": title,
            "input_ids": maintext_encoding["input_ids"].flatten(),
            "attention_mask": maintext_encoding["attention_mask"].flatten(),
            "labels": labels.flatten(),
            "labels_attention_mask": title_encoding["attention_mask"].flatten(),
        }