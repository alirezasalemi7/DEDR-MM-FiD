from torch.utils.data import Dataset
import random
import json

class OKVQADataset(Dataset):

    def __init__(self, samples, n_context, test = False, gpt3_answers_addr = "") -> None:
        super().__init__()

        self.questions = []
        self.question_ids = []
        self.answers = []
        self.image_ids = []
        self.ctxs = []
        self.test = test

        self.use_gpt3 = gpt3_answers_addr != ""
        
        if self.use_gpt3:
            with open(gpt3_answers_addr) as file:
                self.gpt3_answers = json.load(file)

        for sample in samples:
            self.question_ids.append(sample['question_id'])
            self.questions.append(sample['question'])
            self.answers.append(sample['answers'])
            self.image_ids.append(sample["image_id"])
            self.ctxs.append(sample["ctxs"][:n_context] if n_context > 0 else [''])
    
    def __getitem__(self, index: int):
        ctxts = self.ctxs[index]
        img_id = self.image_ids[index]
        if self.use_gpt3:
            ctxts = [self.gpt3_answers[str(img_id)]] + ctxts
        
        return {
            "question" : self.questions[index],
            "question_id" : self.question_ids[index],
            "answers" : random.choice(self.answers[index]) if not self.test else self.answers[index],
            "image_id" : img_id,
            "ctx" : ctxts,
            "test" : self.test
        }

    def __len__(self) -> int:
        return len(self.questions)