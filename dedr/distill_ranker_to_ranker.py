import argparse
from data.datasets import PassageDataset, OKVQATrainingDataset, RetrievalDataset
from transformers.models.lxmert import LxmertConfig, LxmertTokenizer
from transformers.models.bert import BertTokenizer, BertConfig
from modeling.models import RankerToRankerDistilationPipeline, PretrainedConfig, E_MM, E_T, DEDR, DEDRJointTraining
from utils.log import init_logger
from data.utils import load_data
from data.collators import PassageRepresentationCollator, TrainingQueryCollator
from pathlib import Path
from utils.distributed import init_distributed_mode, init_signal_handler
import torch
from eval.dynamic_eval import DynamicEval
from eval.static_eval import StaticEval
import numpy as np
import os
from modeling import optim
from modeling.utils import load_checkpoint, average_main, save_checkpoint
from torch.utils.data import DataLoader, RandomSampler, DistributedSampler, SequentialSampler
import faiss
import tqdm
import pytrec_eval
import glob
import pickle

def gen_passage_rep(opts, model, collator):
    dataset = PassageDataset(opts.val_passages)
    sampler = SequentialSampler(dataset)
    dataloader = DataLoader(
        dataset,
        sampler = sampler,
        batch_size = opts.val_rep_batch_size,
        drop_last = False,
        num_workers = 10,
        collate_fn = collator,
        shuffle = False
    )

    reps = []
    ids = []
    for i, batch in enumerate(tqdm.tqdm(dataloader)):
        if opts.student_model_type == "E_MM":
            outputs = model(
                input_ids = batch['input_ids'].cuda(),
                token_type_ids = batch['token_type_ids'].cuda(),
                attention_mask = batch['attention_mask'].cuda(),
                visual_pos = None,
                visual_feats = None,
            )
        elif opts.student_model_type == "E_T":
            outputs = model(
                input_ids = batch['input_ids'].cuda(),
                token_type_ids = batch['token_type_ids'].cuda(),
                attention_mask = batch['attention_mask'].cuda(),
            )
        elif opts.student_model_type == "DEDR":
            outputs = model(
                input_ids_bert = batch['input_ids'].cuda(),
                token_type_ids_bert = batch['token_type_ids'].cuda(),
                attention_mask_bert = batch['attention_mask'].cuda(),
                input_ids_lxmert = batch['input_ids'].cuda(),
                token_type_ids_lxmert = batch['token_type_ids'].cuda(),
                attention_mask_lxmert = batch['attention_mask'].cuda(),
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
        batch_size = opts.val_rep_batch_size,
        drop_last = False,
        num_workers = 10,
        collate_fn = collator,
        shuffle = False
    )

    reps = []
    ids = []
    for i, batch in enumerate(tqdm.tqdm(dataloader)):
        if opts.student_model_type == "E_MM":
            outputs = model(
                input_ids = batch['query_input_ids_lxmert'].cuda(),
                token_type_ids = batch['query_token_type_ids_lxmert'].cuda(),
                attention_mask = batch['query_attention_mask_lxmert'].cuda(),
                visual_pos = batch['visual_pos'].cuda(),
                visual_feats = batch['visual_feats'].cuda(),
            )
        elif opts.student_model_type == "E_T":
            outputs = model(
                input_ids = batch['query_input_ids_bert'].cuda(),
                token_type_ids = batch['query_token_type_ids_bert'].cuda(),
                attention_mask = batch['query_attention_mask_bert'].cuda(),
            )
        elif opts.student_model_type == "DEDR":
            outputs = model(
                input_ids_bert = batch['query_input_ids_bert'].cuda(),
                token_type_ids_bert = batch['query_token_type_ids_bert'].cuda(),
                attention_mask_bert = batch['query_attention_mask_bert'].cuda(),
                input_ids_lxmert = batch['query_input_ids_lxmert'].cuda(),
                token_type_ids_lxmert = batch['query_token_type_ids_lxmert'].cuda(),
                attention_mask_lxmert = batch['query_attention_mask_lxmert'].cuda(),
                visual_feats = batch['visual_feats'].cuda(),
                visual_pos = batch['visual_pos'].cuda()
            )

        ids.extend(batch['query_ids'])
        reps.extend(outputs.detach().cpu().tolist())
    
    return ids, reps

def train(opts, model, optimizer, scheduler, step, dataset, collator, collator_passage, checkpoint_path, test_dataset, dynamic_eval):
    
    if opts.is_main:
        try:
            tb_logger = torch.utils.tensorboard.SummaryWriter(Path(opts.checkpoint_dir)/opts.name)
        except:
           tb_logger = None
           logger.warning('Tensorboard is not available.')

    torch.manual_seed(opts.global_rank + opts.seed)
    bar = tqdm.tqdm(total=opts.total_steps)
    train_sampler = DistributedSampler(dataset, num_replicas=opts.n_gpu_per_node, rank=opts.local_rank)
    train_dataloader = DataLoader(
        dataset,
        sampler = train_sampler,
        batch_size = opts.per_gpu_batch_size,
        drop_last = True,
        num_workers = 10,
        collate_fn = collator,
    )

    loss, curr_loss = 0.0, 0.0
    epoch = 1
    model.train()
    while step < opts.total_steps:
        epoch += 1
        for i, batch in enumerate(train_dataloader):
            step += 1
            bar.update(1)
            batch = {k:v.cuda() for k, v in batch.items() if type(v) != list}
            train_loss = model(**batch)[0]

            train_loss.backward()

            if step % opts.accumulation_steps == 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), opts.clip)
                optimizer.step()
                scheduler.step()
                model.zero_grad()

            train_loss = average_main(train_loss, opts)
            curr_loss += train_loss.item()
            
            if step % opts.eval_freq == 0 or step == opts.total_steps:
                if opts.is_main:
                    model.eval()
                    metric = evaluate(model.module.model if opts.is_distributed else model.model, test_dataset, collator, collator_passage, opts, step, dynamic_eval, logger)
                    log = f"{step} / {opts.total_steps} |"
                    log += f"train: {curr_loss/opts.eval_freq:.3f} |"
                    log += f"MRR@5: {metric['MRR@5']:.3f} |"
                    log += f"precision@1: {metric['Precision@1']:.3f} |"
                    log += f"precision@5: {metric['Precision@5']:.3f} |"
                    logger.info(log)
                    if tb_logger is not None:
                        tb_logger.add_scalar("Training", curr_loss / (opts.eval_freq), step)
                    curr_loss = 0.
                    save_checkpoint(model, optimizer, scheduler, step, opts, checkpoint_path, f"step-{step}")
                    model.train()

            if opts.is_main and step % opts.save_freq == 0:
                save_checkpoint(model, optimizer, scheduler, step, opts, checkpoint_path, f"step-{step}")
            if step > opts.total_steps:
                break
    save_checkpoint(model, optimizer, scheduler, step, opts, checkpoint_path, f"step-{step}")

def evaluate(model, dataset, collator_test, collator_passage, opts, step, dynamic_eval, logger):
    logger.info("Evaluation started")
    if not os.path.exists(opts.output_dir):
        os.makedirs(opts.output_dir)
    predict_dir = os.path.join(opts.output_dir, 'predictions')
    if not os.path.exists(predict_dir):
        os.makedirs(predict_dir)

    if opts.index_addr:
        logger.info("Loading index")
        addrs = glob.glob(os.path.join(opts.index_addr, "*"))
        addrs = sorted(addrs)
        passage_ids, passage_reps = [], None
        for addr in addrs:
            print(addr)
            with open(addr, "rb") as file:
                pids, reps = pickle.load(file)
                if passage_reps is None:
                    passage_reps = np.array(reps, dtype='float32')
                else:
                    passage_reps = np.append(passage_reps, reps, axis = 0)
                passage_ids.extend(pids)
    else:
        logger.info("Generating passages representations")
        passage_ids, passage_reps = gen_passage_rep(opts, model, collator_passage)
        passage_reps = np.asarray(passage_reps, dtype='float32')

    index = faiss.IndexFlatIP(model.config.hidden_size)
    index.add(passage_reps)

    logger.info("Generating queries representations")
    qids, query_reps = gen_query_rep(opts, model, dataset, collator_test)
    query_reps = np.asarray(query_reps, dtype='float32')

    D, I = index.search(query_reps, opts.top_k_retrieve)

    with open(os.path.join(predict_dir, f'results_{step}.txt'), 'w') as fout:
        run = {}
        for qid, retrieved_ids, scores in zip(qids, I, D):
            run[str(qid)] = {passage_ids[retrieved_id]: float(
                score) for retrieved_id, score in zip(retrieved_ids, scores)}
            for i, (retrieved_id, score) in enumerate(zip(retrieved_ids, scores)):
                fout.write(f'{qid} Q0 {passage_ids[retrieved_id]} {i + 1} {score} DENSE\n')

    qrels = dynamic_eval.gen_qrels(qids, I, passage_ids)
    
    with open(os.path.join(predict_dir, f'qrels_{step}.txt'), 'w') as qrels_fout:
        for qid, pids in qrels.items():
            for pid, relevance in pids.items():
                qrels_fout.write(f'{qid} 0 {pid} {relevance}\n')

    assert len(qrels) == len(qids) == len(run), f'lengths of qrels, qids, and run do not match {len(qrels)}, {len(qids)}, {len(run)}'
    num_passages_in_qrels = len([pid for l in qrels.values() for pid in l])
    num_pos_passages = len([pid for l in qrels.values() for pid in l if pid != 'placeholder'])
    num_placeholder_passages = len([pid for l in qrels.values() for pid in l if pid == 'placeholder'])
    num_questions_with_pos_passages = len([ps for ps in qrels.values() if ps != {'placeholder': 0}])
    assert num_pos_passages + num_placeholder_passages == num_passages_in_qrels
    assert num_placeholder_passages == len(qrels)
    logger.info(f'len(qrels): {len(qrels)}')
    logger.info(f'num_passages_in_qrels: {num_passages_in_qrels}')
    logger.info(f'num_pos_passages: {num_pos_passages}')
    logger.info(f'num_placeholder_passages: {num_placeholder_passages}')
    logger.info(f'num_questions_with_pos_passages: {num_questions_with_pos_passages}')
    
    evaluator = pytrec_eval.RelevanceEvaluator(
        qrels, {'recip_rank_1', 'recip_rank_5', 'P_5', 'P_1'})
    metrics = evaluator.evaluate(run)
    mrr_list = [v['recip_rank'] for v in metrics.values()]
    p_1_list = [v['P_1'] for v in metrics.values()]
    p_5_list = [v['P_5'] for v in metrics.values()]
    eval_metrics = {'MRR@5': np.average(
        mrr_list), 'Precision@1': np.average(p_1_list), 'Precision@5': np.average(p_5_list)}
    
    for k, v in eval_metrics.items():
        logger.info(f'{k}: {v}')
        
    return eval_metrics


parser = argparse.ArgumentParser()

parser.add_argument("--train_data", required = True, help="training data")
parser.add_argument("--val_data", required = True, help="validation data")
parser.add_argument("--val_passages", required = True, help="validation corpus or passage source")
parser.add_argument("--val_rep_batch_size", type = int, default = 128, help="batch size of generating passage representation")

parser.add_argument("--ann_file", required = False, help="the address of the ann file for dynamic validation")
parser.add_argument("--ques_file", required = False, help="the address of the qs file for dynamic validation")
parser.add_argument("--passage_id_to_line_id_file", required = False, help="the address of a file that shows each passage id is in which line of corpus or passage source for dynamic validation")
parser.add_argument("--all_blocks_file", required = False, help="the address of corpus or passage source for dynamic validation")

parser.add_argument("--do_train", action='store_true', help="perform training")
parser.add_argument("--do_validation", action='store_true', help="perform validation")
parser.add_argument("--max_length", type = int, default = 400, help="maximum input length")

parser.add_argument('--name', type=str, default='experiment_name', help='name of the experiment')
parser.add_argument('--checkpoint_dir', type=str, default='./checkpoint/', help='models are saved here')
parser.add_argument("--per_gpu_batch_size", default=1, type=int, 
           help="Batch size per GPU/CPU for training.")
parser.add_argument("--local_rank", type=int, default=-1,
           help="For distributed training: local_rank")
parser.add_argument("--main_port", type=int, default=-1,
           help="Main port (for multi-node SLURM jobs)")
parser.add_argument('--seed', type=int, default=0, help="random seed for initialization")
parser.add_argument('--eval_freq', type=int, default=500,
           help='evaluate model every <eval_freq> steps during training')
parser.add_argument('--save_freq', type=int, default=5000,
           help='save model every <save_freq> steps during training')
parser.add_argument('--eval_print_freq', type=int, default=1000,
           help='print intermdiate results of evaluation every <eval_print_freq> steps')
parser.add_argument('--warmup_steps', type=int, default=1000, help="number of warmup steps")
parser.add_argument('--total_steps', type=int, default=1000, help="number of training steps")
parser.add_argument('--scheduler_steps', type=int, default=None, 
           help='total number of step for the scheduler, if None then scheduler_total_step = total_step')
parser.add_argument('--accumulation_steps', type=int, default=1, help="number of gradient accumulation steps")
parser.add_argument('--dropout', type=float, default=0.1, help='dropout rate')
parser.add_argument('--lr', type=float, default=1e-5, help='learning rate')
parser.add_argument('--clip', type=float, default=1., help='gradient clipping')
parser.add_argument('--optim', type=str, default='adam', help="optimizer which is used for training")
parser.add_argument('--scheduler', type=str, default='fixed', help="scheduler which is used for training")
parser.add_argument('--weight_decay', type=float, default=0.0, help="weight decay rate")
parser.add_argument('--fixed_lr', action='store_true', help="use a fixed lr")
parser.add_argument("--image_feats_addr", required=True, help="address to the dataset that contains image features")
parser.add_argument("--image_feats_mapping_addr", required=True, help="address to the json file that map each image id to the index in the feature dataset")
parser.add_argument('--top_k_retrieve', type=int, default=20, help="number of retrieved docs")
parser.add_argument('--neg_type', default="other_pos+all_neg", 
                    help="type of in batch negative: neg: just the provided negative, all_neg: all in batch negativies as negative, other_pos+neg: all positives and the provided negative as negative, other_pos+all_neg: all positives and all provided negatives as negative")
parser.add_argument("--caption_address", required=True, help="address to the caption file")
parser.add_argument("--teacher_model_type", required=True, help="teacher model type: E_T, E_MM")
parser.add_argument("--student_model_type", required=True, help="student model type: E_T, E_MM")
parser.add_argument("--teacher_model_path", required=True, help="teacher model path")
parser.add_argument("--student_model_path", required=False, help="student model path")
parser.add_argument("--qrels_addr", default="", help="address to the qrels file for static validation")
parser.add_argument("--index_addr", default="", help="address to the directory that contains index files")


if __name__ == "__main__":
    opts = parser.parse_args()

    torch.manual_seed(opts.seed)
    init_distributed_mode(opts)
    init_signal_handler()

    checkpoint_path = Path(opts.checkpoint_dir)/opts.name
    checkpoint_exists = checkpoint_path.exists()
    if opts.is_distributed:
        torch.distributed.barrier()
    
    checkpoint_path.mkdir(parents = True, exist_ok = True)
    opts.output_dir = checkpoint_path

    logger = init_logger(
        opts.is_main,
        opts.is_distributed,
        checkpoint_path / 'run.log'
    )

    logger.info(opts)

    train_examples = load_data(opts.train_data)
    test_examples = load_data(opts.val_data)

    tokenizer_lxmert = LxmertTokenizer.from_pretrained("unc-nlp/lxmert-base-uncased")
    tokenizer_bert = BertTokenizer.from_pretrained("bert-base-uncased")
    collator = TrainingQueryCollator(tokenizer_bert, tokenizer_lxmert, opts.max_length, opts.image_feats_addr, opts.image_feats_mapping_addr, opts.caption_address)
    collator_passage = PassageRepresentationCollator(tokenizer = tokenizer_bert, max_length = opts.max_length)
    
    if opts.teacher_model_type == "E_MM":
        teacher = E_MM.from_pretrained(opts.teacher_model_path)
    elif opts.teacher_model_type == "E_T":
        teacher = E_T.from_pretrained(opts.teacher_model_path)

    student_class = None
    if opts.student_model_type == "E_MM":
        student_class = E_MM
        if opts.student_model_path:
            student = E_MM.from_pretrained(opts.student_model_path)
        else:
            config = LxmertConfig.from_pretrained("unc-nlp/lxmert-base-uncased")
            student = E_MM(config, True)
    elif opts.student_model_type == "E_T":
        student_class = E_T
        if opts.student_model_path:
            student = E_T.from_pretrained(opts.student_model_path)
        else:
            config = BertConfig.from_pretrained("bert-base-uncased")
            student = E_T(config)

    if not checkpoint_exists:
        model = RankerToRankerDistilationPipeline(student, opts.student_model_type, teacher, opts.teacher_model_type, opts.neg_type)
        optimizer, scheduler = optim.set_optim(opts, model)
        step = 0
    else:
        student, optimizer, scheduler, opt_checkpoint, step = load_checkpoint(student_class, os.path.join(checkpoint_path, "checkpoint", "latest"), opts)
        model = RankerToRankerDistilationPipeline(student, opts.student_model_type, teacher, opts.teacher_model_type, opts.neg_type)
    
    
    model = model.to(opts.local_rank)

    if opts.is_distributed:
        model = torch.nn.parallel.DistributedDataParallel(
            model,
            device_ids=[opts.local_rank],
            output_device=opts.local_rank,
            find_unused_parameters=True,
        )

    train_dataset = OKVQATrainingDataset(train_examples)
    val_dataset = OKVQATrainingDataset(test_examples)


    if opts.qrels_addr:
        dynamic_eval = StaticEval(opts.qrels_addr)
    else:
        dynamic_eval = DynamicEval(opts.ann_file, opts.ques_file, opts.passage_id_to_line_id_file, opts.all_blocks_file)

    if opts.do_train:
        train(opts, model, optimizer, scheduler, step, train_dataset, collator, collator_passage, checkpoint_path, val_dataset, dynamic_eval)
    
    if opts.do_validation and opts.is_main:
        evaluate(model.model, val_dataset, collator, collator_passage, opts, step, dynamic_eval, logger)