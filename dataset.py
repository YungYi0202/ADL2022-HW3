from torch.utils.data import Dataset
from utils import *
class NewsSummaryDataset(Dataset):
    def __init__(self, data, tokenizer, title_max_len: int = 64, maintext_max_len:int = 2048, add_prefix: bool = False):
        self.data = data
        self.tokenizer = tokenizer
        self.title_max_len = title_max_len
        self.maintext_max_len = maintext_max_len
        self.add_prefix = add_prefix

    def __len__(self):
        return len(self.data)
    
    def clean(self, text):
        text = text.replace("\n", "")
        text = text.replace("\r", "")
        #text = text.replace(" ", "")
        #text = text.replace("`", "")
        return text

    def __getitem__(self, index: int):
        sample = self.data[index]
        ret = dict()
        if TITLE in sample:
            title = sample[TITLE]
            title_encoding = self.tokenizer(
                self.clean(title),
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
        
            ret["title"] = title
            ret["labels"] = labels.flatten()
            ret["labels_attention_mask"] = title_encoding["attention_mask"].flatten()

        if MAINTEXT in sample:
            maintext = sample[MAINTEXT]
            if add_prefix:
                maintext = "summarize: " + maintext
            maintext_encoding = self.tokenizer(
                self.clean(maintext),
                max_length=self.title_max_len,
                padding="max_length",
                truncation=True,
                return_attention_mask=True,
                add_special_tokens=True,
                return_tensors="pt"
            )
            ret["input_ids"] = maintext_encoding["input_ids"].flatten()
            ret["attention_mask"] = maintext_encoding["attention_mask"].flatten()
        
        if ID in sample:
            ret[ID] = sample[ID]

        return ret