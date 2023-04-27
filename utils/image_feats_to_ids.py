import argparse
import datasets
import json

parser = argparse.ArgumentParser()

parser.add_argument("--image_feats_addr", required = True)
parser.add_argument("--output", required = True)

if __name__ == "__main__":
    opts = parser.parse_args()

    dataset = datasets.Dataset.from_file(opts.image_feats_addr)

    id2index = dict()
    for i, row in enumerate(dataset):
        id2index[row['img_id']] = i
    
    with open(opts.output, "w") as file:
        json.dump(id2index, file)