import json

def load_data(data_path):
    examples = []
    with open(data_path) as file:
        for k, line in enumerate(file):
            if line:
                obj = json.loads(line)
                examples.append(obj)
    return examples
