import numpy as np
import torch
import json

import datasets

# CaptBERTLXMERTQueryCollator
class TrainingQueryCollator(object):
    
    def __init__(self, tokenizer_bert, tokenizer_lxmert, max_length, image_feats_addr, image_mapping_addr, caption_file_addr) -> None:
        
        self.tokenizer_bert = tokenizer_bert
        self.tokenizer_lxmert = tokenizer_lxmert
        self.max_len = max_length
        
        with open(image_mapping_addr) as file:
            self.image_id_to_dataset_mapping = json.load(file)

        self.image_features = datasets.Dataset.from_file(image_feats_addr)
        self.captions = dict()

        with open(caption_file_addr) as file:
            caps = json.load(file)
            for k, v in caps.items():
                self.captions[k] = " ".join(v).strip()
    
    def __call__(self, batch):
        query_lxmert = self.tokenizer_lxmert.batch_encode_plus(
            [ex['question'][0] for ex in batch],
            max_length = self.max_len,
            padding='max_length',
            return_tensors = 'pt',
            truncation = True
        )

        pos_passages_lxmert = self.tokenizer_lxmert.batch_encode_plus(
            [ex['pos_passage'][0] for ex in batch],
            max_length = self.max_len,
            padding='max_length',
            return_tensors = 'pt',
            truncation = True
        )
        
        neg_passages_lxmert = self.tokenizer_lxmert.batch_encode_plus(
            [ex['neg_passage'][0] for ex in batch],
            max_length = self.max_len,
            padding='max_length',
            return_tensors = 'pt',
            truncation = True
        )

        visual_feats = torch.tensor([self.image_features[self.image_id_to_dataset_mapping[str(ex['img_id'][0])]]['roi_features'] for ex in batch])
        visual_pos = torch.tensor([self.image_features[self.image_id_to_dataset_mapping[str(ex['img_id'][0])]]['boxes'] for ex in batch])
        
        query_bert = self.tokenizer_bert.batch_encode_plus(
            [ex['question'][0] + " " + self.tokenizer_bert.sep_token + " " + self.captions[str(ex['img_id'][0])] for ex in batch],
            max_length = self.max_len,
            padding='max_length',
            return_tensors = 'pt',
            truncation = True
        )

        pos_passages_bert = self.tokenizer_bert.batch_encode_plus(
            [ex['pos_passage'][0] for ex in batch],
            max_length = self.max_len,
            padding='max_length',
            return_tensors = 'pt',
            truncation = True
        )
        
        neg_passages_bert = self.tokenizer_bert.batch_encode_plus(
            [ex['neg_passage'][0] for ex in batch],
            max_length = self.max_len,
            padding='max_length',
            return_tensors = 'pt',
            truncation = True
        )

        return {
            "query_input_ids_lxmert" : query_lxmert['input_ids'],
            "query_token_type_ids_lxmert" : query_lxmert['token_type_ids'],
            "query_attention_mask_lxmert" : query_lxmert['attention_mask'],
            "pos_input_ids_lxmert" : pos_passages_lxmert['input_ids'],
            "pos_token_type_ids_lxmert" : pos_passages_lxmert['token_type_ids'],
            "pos_attention_mask_lxmert" : pos_passages_lxmert['attention_mask'],
            "visual_feats" : visual_feats,
            "visual_pos" : visual_pos,
            "neg_input_ids_lxmert" : neg_passages_lxmert['input_ids'],
            "neg_token_type_ids_lxmert" : neg_passages_lxmert['token_type_ids'],
            "neg_attention_mask_lxmert" : neg_passages_lxmert['attention_mask'],
            "query_ids" : [ex["qid"][0] for ex in batch],
            "query_input_ids_bert" : query_bert['input_ids'],
            "query_token_type_ids_bert" : query_bert['token_type_ids'],
            "query_attention_mask_bert" : query_bert['attention_mask'],
            "pos_input_ids_bert" : pos_passages_bert['input_ids'],
            "pos_token_type_ids_bert" : pos_passages_bert['token_type_ids'],
            "pos_attention_mask_bert" : pos_passages_bert['attention_mask'],
            "neg_input_ids_bert" : neg_passages_bert['input_ids'],
            "neg_token_type_ids_bert" : neg_passages_bert['token_type_ids'],
            "neg_attention_mask_bert" : neg_passages_bert['attention_mask'],
        }
# CaptBERTLXMERTQueryForRetrievalCollator
class RetrievalCollator(object):
    
    def __init__(self, tokenizer_bert, tokenizer_lxmert, max_length, image_feats_addr, image_mapping_addr, caption_file_addr) -> None:
        
        self.tokenizer_bert = tokenizer_bert
        self.tokenizer_lxmert = tokenizer_lxmert
        self.max_len = max_length
        
        with open(image_mapping_addr) as file:
            self.image_id_to_dataset_mapping = json.load(file)

        self.image_features = datasets.Dataset.from_file(image_feats_addr)
        self.captions = dict()

        with open(caption_file_addr) as file:
            caps = json.load(file)
            for k, v in caps.items():
                self.captions[k] = " ".join(v).strip()
    
    def __call__(self, batch):
        query_lxmert = self.tokenizer_lxmert.batch_encode_plus(
            [ex['question'][0] for ex in batch],
            max_length = self.max_len,
            padding='max_length',
            return_tensors = 'pt',
            truncation = True
        )

        visual_feats = torch.tensor([self.image_features[self.image_id_to_dataset_mapping[str(ex['img_id'][0])]]['roi_features'] for ex in batch])
        visual_pos = torch.tensor([self.image_features[self.image_id_to_dataset_mapping[str(ex['img_id'][0])]]['boxes'] for ex in batch])
        
        query_bert = self.tokenizer_bert.batch_encode_plus(
            [ex['question'][0] + " " + self.tokenizer_bert.sep_token + " " + self.captions[str(ex['img_id'][0])] for ex in batch],
            max_length = self.max_len,
            padding='max_length',
            return_tensors = 'pt',
            truncation = True
        )

        return {
            "query_input_ids_lxmert" : query_lxmert['input_ids'],
            "query_token_type_ids_lxmert" : query_lxmert['token_type_ids'],
            "query_attention_mask_lxmert" : query_lxmert['attention_mask'],
            "visual_feats" : visual_feats,
            "visual_pos" : visual_pos,
            "query_ids" : [ex["qid"][0] for ex in batch],
            "query_input_ids_bert" : query_bert['input_ids'],
            "query_token_type_ids_bert" : query_bert['token_type_ids'],
            "query_attention_mask_bert" : query_bert['attention_mask'],
        }

class PassageRepresentationCollator(object):
    
    def __init__(self, tokenizer, max_length) -> None:
        
        self.tokenizer = tokenizer
        self.max_len = max_length
    
    def __call__(self, batch):

        passages = self.tokenizer.batch_encode_plus(
            [ex['text'][0] for ex in batch],
            max_length = self.max_len,
            padding='max_length',
            return_tensors = 'pt',
            truncation = True
        )

        return {
            "input_ids" : passages['input_ids'],
            "token_type_ids" : passages['token_type_ids'],
            "attention_mask" : passages['attention_mask'],
            "pid" : [ex['pid'][0] for ex in batch]
        }