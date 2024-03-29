import json

# Global varirables 
TRAIN = "train"
DEV = "dev"
TEST = "test"

TITLE = "title"
MAINTEXT = "maintext"
ID = "id"
KEYS = {
    TRAIN: [TITLE, MAINTEXT],
    DEV: [TITLE, MAINTEXT],
    TEST: [MAINTEXT, ID],
}

def read_data(filename, keys):
    data = []
    with open(filename, 'r') as json_file:
        train_json_list = list(json_file)
    for json_str in train_json_list:
        result = json.loads(json_str)
        new_result = {key: result[key] for key in keys}
        data.append(new_result)
    return data