import json
import datasets
import torch

class OKVQAGenerationCollator(object):

    def __init__(self, tokenizer, max_length_input, max_length_output, image_feats_addr, image_mapping_addr, use_prefix = False) -> None:
        
        self.tokenizer = tokenizer
        with open(image_mapping_addr) as file:
            self.image_id_to_dataset_mapping = json.load(file)

        self.image_features = datasets.Dataset.from_file(image_feats_addr)
        self.max_len_in = max_length_input
        self.max_len_out = max_length_output
        self.use_prefix = use_prefix

    def __call__(self, batch):
        test = any([ex['test'] for ex in batch])
        visual_feats = torch.tensor([self.image_features[self.image_id_to_dataset_mapping[str(ex['image_id'])]]['roi_features'] for ex in batch])
        visual_pos = torch.tensor([self.image_features[self.image_id_to_dataset_mapping[str(ex['image_id'])]]['boxes'] for ex in batch])

        if not self.use_prefix:
            inps = [[ex['question'] + ' ' + ctx for ctx in ex['ctx']] for ex in batch]
        else:
            inps = [["question: " + ex['question'] + ' context: ' + ctx for ctx in ex['ctx']] for ex in batch]
        inp_input_ids = []
        inp_attention_masks = []
        for x in inps:
            inp = self.tokenizer.batch_encode_plus(
                x,
                max_length=self.max_len_in,
                padding='max_length',
                return_tensors='pt',
                truncation=True,
            )
            inp_input_ids.append(inp['input_ids'])
            inp_attention_masks.append(inp['attention_mask'])
        
        inp_input_ids = torch.stack(inp_input_ids)
        inp_attention_masks = torch.stack(inp_attention_masks)

        if test:
            return {
                "input_ids" : inp_input_ids,
                "attention_mask" : inp_attention_masks,
                "vis_inputs" : (visual_feats, visual_pos),
                "golds" : [ex['answers'] for ex in batch],
                "question_ids" : [ex['question_id'] for ex in batch]
            }
        else:
            target = [ex['answers'] for ex in batch]
            target = self.tokenizer.batch_encode_plus(
                target,
                max_length=self.max_len_out,
                padding='max_length',
                return_tensors='pt',
                truncation=True,
            )
            target_ids = target["input_ids"]
            target_mask = target["attention_mask"].bool()
            target_ids = target_ids.masked_fill(~target_mask, -100)

            return {
                "input_ids" : inp_input_ids,
                "attention_mask" : inp_attention_masks,
                "vis_inputs" : (visual_feats, visual_pos),
                "labels" : target_ids,
                "labels_mask" : target_mask,
                "question_ids" : [ex['question_id'] for ex in batch]
            }