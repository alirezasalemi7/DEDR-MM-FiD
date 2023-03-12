from torch.utils.data import Dataset
import json
import glob
import os
from PIL import Image

class PassageDataset(Dataset):
    
    def __init__(self, data_addr, shard_id = 0, num_shards = 1) -> None:
        super().__init__()

        self.passages = []
        self.ids = []
        with open(data_addr) as file:
            for line in file:
                if line:
                    obj = json.loads(line)
                    self.passages.append(obj['text'])
                    self.ids.append(obj['id'])
        shard_size = len(self.passages) // num_shards
        start_idx = shard_id * shard_size
        end_idx = start_idx + shard_size
        if shard_id == num_shards-1:
            end_idx = len(self.passages)
        
        self.passages = self.passages[start_idx:end_idx]
        self.ids = self.ids[start_idx:end_idx]

    def __len__(self):
        return len(self.passages)

    def __getitem__(self, index):
        passage = self.passages[index]
        pid = self.ids[index]

        return {
            "text" : [passage],
            "pid" : [pid]
        }


class OKVQATrainingDataset(Dataset):

    def __init__(self, data) -> None:
        super().__init__()

        self.image_ids = []
        self.qids = []
        self.answers = []
        self.questions = []
        self.positive_passages = []
        self.negative_passages = []

        for obj in data:
            image_id = obj['image_id']
            question = obj['question']
            qid = obj['question_id']
            answers = obj['answers']
            pos_passage = obj['pos_passage']['passage']
            neg_passage = obj['neg_passage']['passage']

            self.image_ids.append(image_id)
            self.answers.append(answers)
            self.questions.append(question)
            self.positive_passages.append(pos_passage)
            self.negative_passages.append(neg_passage)
            self.qids.append(qid)

    def __len__(self):
        return len(self.questions)

    def __getitem__(self, index):
        q = self.questions[index]
        pos_p = self.positive_passages[index]
        neg_p = self.negative_passages[index]
        img_id = self.image_ids[index]
        answers = self.answers[index]
        qid = self.qids[index]
        return {
            "question" : [q],
            "pos_passage" : [pos_p],
            "neg_passage" : [neg_p],
            "img_id" : [img_id],
            "answers" : [answers],
            "qid" : [qid]
        }

class RetrievalDataset(Dataset):

    def __init__(self, data) -> None:
        super().__init__()

        self.image_ids = []
        self.qids = []
        self.questions = []
        
        for obj in data:
            image_id = obj['image_id']
            question = obj['question']
            qid = obj['question_id']
            
            self.image_ids.append(image_id)
            self.questions.append(question)
            self.qids.append(qid)

    def __len__(self):
        return len(self.questions)

    def __getitem__(self, index):
        q = self.questions[index]
        img_id = self.image_ids[index]
        qid = self.qids[index]
        return {
            "question" : [q],
            "img_id" : [img_id],
            "qid" : [qid]
        }