import faiss
import argparse
from torch.utils.data import DataLoader, RandomSampler, DistributedSampler, SequentialSampler
from data.datasets import RetrievalDataset, PassageDataset
from data.collators import RetrievalCollator, PassageRepresentationCollator
from modeling.models import PretrainedConfig, E_MM, E_T, DEDR, DEDRJointTraining
from transformers.models.bert import BertTokenizer
from transformers.models.lxmert import LxmertTokenizer
import pickle
import json
import torch
import numpy as np
import os
import glob
import tqdm

def gen_passage_rep(opts, model, collator):
    dataset = PassageDataset(opts.passages)
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
        if opts.model_type == "E_MM":
            outputs = model(
                input_ids = batch['input_ids'].to(opts.device),
                token_type_ids = batch['token_type_ids'].to(opts.device),
                attention_mask = batch['attention_mask'].to(opts.device),
                visual_pos = None,
                visual_feats = None,
            )
        elif opts.model_type == "E_T":
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

def gen_query_rep(opts, model, dataset, collator):
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
        if opts.model_type == "E_MM":
            outputs = model(
                input_ids = batch['query_input_ids_lxmert'].to(opts.device),
                token_type_ids = batch['query_token_type_ids_lxmert'].to(opts.device),
                attention_mask = batch['query_attention_mask_lxmert'].to(opts.device),
                visual_pos = batch['visual_pos'].to(opts.device),
                visual_feats = batch['visual_feats'].to(opts.device),
            )
        elif opts.model_type == "E_T":
            outputs = model(
                input_ids = batch['query_input_ids_bert'].to(opts.device),
                token_type_ids = batch['query_token_type_ids_bert'].to(opts.device),
                attention_mask = batch['query_attention_mask_bert'].to(opts.device),
            )
        elif opts.model_type in ["DEDR", "DEDR_joint"]:
            outputs = model(
                input_ids_bert = batch['query_input_ids_bert'].to(opts.device),
                token_type_ids_bert = batch['query_token_type_ids_bert'].to(opts.device),
                attention_mask_bert = batch['query_attention_mask_bert'].to(opts.device),
                input_ids_lxmert = batch['query_input_ids_lxmert'].to(opts.device),
                token_type_ids_lxmert = batch['query_token_type_ids_lxmert'].to(opts.device),
                attention_mask_lxmert = batch['query_attention_mask_lxmert'].to(opts.device),
                visual_feats = batch['visual_feats'].to(opts.device),
                visual_pos = batch['visual_pos'].to(opts.device)
            )

        ids.extend(batch['query_ids'])
        reps.extend(outputs.detach().cpu().tolist())
    
    return ids, reps

parser = argparse.ArgumentParser()

parser.add_argument("--model_path", required=True, help="model checkpoint for indexing")
parser.add_argument("--model_type", required=True, help="model type for indexing: E_T, E_MM, DEDR, DEDR_joint")
parser.add_argument("--passages", required=True, help="the address of corpus or passage source")
parser.add_argument("--output_file", required=True, help="address to the output file")
parser.add_argument("--batch_size", type=int, default=4, help="batch size")
parser.add_argument("--max_length", type=int, default=512, help="maximum input length")
parser.add_argument("--top_k_retrieve", type=int, required=True, help="number of retrieved docs")
parser.add_argument("--input_queries", required=True, help="address to the input queries")
parser.add_argument("--image_feats_addr", required=True, help="address to the dataset that contains image features")
parser.add_argument("--image_feats_mapping_addr", required=True, help="address to the json file that map each image id to the index in the feature dataset")
parser.add_argument("--caption_address", required=True, help="address to the caption file")
parser.add_argument("--index_addr", default="", help="address to the directory that contains index files")
parser.add_argument("--model2_path", default="", help="address to the second checkpoint only used when model_type is DEDR")
parser.add_argument("--device", default="cpu", help="device")


if __name__ == "__main__":
    
    opts = parser.parse_args()
    queries_saved = dict()

    queries = []
    with open(opts.input_queries) as file:
        for line in file:
            if line.strip():
                queries.append(json.loads(line.strip()))
                obj = queries[-1]
                queries_saved[obj['question_id']] = (obj['question'], obj['image_id'], obj['answers'] if "answers" in obj.keys() else [])
    
    dataset = RetrievalDataset(queries)
    tokenizer_lxmert = LxmertTokenizer.from_pretrained("unc-nlp/lxmert-base-uncased")
    tokenizer_bert = BertTokenizer.from_pretrained("bert-base-uncased")
    collator = RetrievalCollator(tokenizer_bert, tokenizer_lxmert, opts.max_length, opts.image_feats_addr, opts.image_feats_mapping_addr, opts.caption_address)
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
    
    if opts.index_addr:
        print("Loading index")
        addrs = glob.glob(os.path.join(opts.index_addr, "*"))
        addrs = sorted(addrs)
        passage_ids, passage_reps = [], None
        for addr in addrs:
            with open(addr, "rb") as file:
                pids, reps = pickle.load(file)
                if passage_reps is None:
                    passage_reps = np.array(reps, dtype='float32')
                else:
                    passage_reps = np.append(passage_reps, reps, axis = 0)
                passage_ids.extend(pids)
    else:
        print("Generating passages representations")
        passage_ids, passage_reps = gen_passage_rep(opts, model, collator_passage)
        passage_reps = np.asarray(passage_reps, dtype='float32')
    
    index = faiss.IndexFlatIP(model.config.hidden_size)
    index.add(passage_reps)
    
    print("Generating queries representations")
    qids, qreps = gen_query_rep(opts, model, dataset, collator)
    qreps = np.asarray(qreps, dtype='float32')

    D, I = index.search(qreps, opts.top_k_retrieve)
    
    all_ids = set([passage_ids[i] for x in I for i in x])
    selected_passages = dict()

    with open(opts.passages) as file:
        for line in file:
            if line.strip():
                obj = json.loads(line.strip())
                if obj['id'] in all_ids:
                    selected_passages[obj['id']] = obj['text']

    print("Writing the results")
    with open(opts.output_file, 'w') as file:
        for qid, ret_ids in zip(qids, I):
            ret_passages = []
            for ret_id in ret_ids:
                # ret_passages.append({"id" : passage_ids[ret_id], "passage" : selected_passages[passage_ids[ret_id]]})
                ret_passages.append(selected_passages[passage_ids[ret_id]])
            json.dump({"question_id" : qid, "question" : queries_saved[qid][0], "image_id" : queries_saved[qid][1], "ctxs" : ret_passages, "answers" : queries_saved[qid][2]}, file)
            file.write("\n")

    