import argparse
from torch.utils.data import DataLoader, RandomSampler, DistributedSampler, SequentialSampler
from data.datasets import PassageDataset
from data.collators import PassageRepresentationCollator
from modeling.models import PretrainedConfig, E_MM, E_T, DEDRConfig, DEDRJointTraining, DEDR
from transformers.models.bert import BertTokenizer
import pickle
import torch
import tqdm


def gen_passage_rep(opts, model, collator):
    dataset = PassageDataset(opts.passages, opts.shard_id, opts.num_shards)
    sampler = SequentialSampler(dataset)
    dataloader = DataLoader(
        dataset,
        sampler = sampler,
        batch_size = opts.batch_size,
        drop_last = False,
        num_workers = 10,
        collate_fn = collator,
        shuffle = False
    )

    reps = []
    ids = []
    for i, batch in enumerate(tqdm.tqdm(dataloader)):
        if opts.model_type in ["E_MM"]:
            outputs = model(
                input_ids = batch['input_ids'].to(opts.device),
                token_type_ids = batch['token_type_ids'].to(opts.device),
                attention_mask = batch['attention_mask'].to(opts.device),
                visual_pos = None,
                visual_feats = None,
            )
        elif opts.model_type in ["E_T"]:
            outputs = model(
                input_ids = batch['input_ids'].to(opts.device),
                token_type_ids = batch['token_type_ids'].to(opts.device),
                attention_mask = batch['attention_mask'].to(opts.device),
            )
        elif opts.model_type in ["DEDR", "DEDR_joint"]:
            outputs = model(
                input_ids_bert = batch['input_ids'].to(opts.device),
                token_type_ids_bert = batch['token_type_ids'].to(opts.device),
                attention_mask_bert = batch['attention_mask'].to(opts.device),
                input_ids_lxmert = batch['input_ids'].to(opts.device),
                token_type_ids_lxmert = batch['token_type_ids'].to(opts.device),
                attention_mask_lxmert = batch['attention_mask'].to(opts.device),
                visual_feats = None,
                visual_pos = None
            )
        
        ids.extend(batch['pid'])
        reps.extend(outputs.detach().cpu().tolist())
    
    return ids, reps

parser = argparse.ArgumentParser()
parser.add_argument("--model_path", required=True, help="model checkpoint for indexing")
parser.add_argument("--model_type", required=True, help="model type for indexing: E_T, E_MM, DEDR, DEDR_joint")
parser.add_argument("--passages", required=True, help="the address of corpus or passage source")
parser.add_argument("--output", required=True, help="address to the output file")
parser.add_argument("--batch_size", type=int, default=4, help="batch size")
parser.add_argument("--max_length", type=int, default=512, help="maximum input length")
parser.add_argument("--shard_id", type=int, default=0, help="shard id")
parser.add_argument("--num_shards", type=int, default=1, help="shards count")
parser.add_argument("--model2_path", default="", help="address to the second checkpoint only used when model_type is DEDR")
parser.add_argument("--device", default="cpu", help="device")


if __name__ == "__main__":
    
    opts = parser.parse_args()

    tokenizer_bert = BertTokenizer.from_pretrained("bert-base-uncased")
    collator_passage = PassageRepresentationCollator(tokenizer = tokenizer_bert, max_length = opts.max_length)

    if opts.model_type == "E_MM":
        model = E_MM.from_pretrained(opts.model_path)
    elif opts.model_type == "E_T":
        model = E_T.from_pretrained(opts.model_path)
    elif opts.model_type == "DEDR_joint":
        config = PretrainedConfig.from_pretrained(opts.model_path)
        model = DEDRJointTraining.from_pretrained(opts.model_path, config = config)
    elif opts.model_type == "DEDR":
        model1 = E_MM.from_pretrained(opts.model_path)
        model2 = E_T.from_pretrained(opts.model2_path)
        model = DEDR(model2, model1)
    else:
        raise RuntimeError("Unsupported model type")
    
    model.eval()
    model = model.to(torch.device(opts.device))
    print("Generating passages representations")
    ids, reps = gen_passage_rep(opts, model, collator_passage)

    print("Writing results")
    with open(opts.output+f"_{opts.shard_id}", "wb") as f:
        pickle.dump((ids, reps), f)