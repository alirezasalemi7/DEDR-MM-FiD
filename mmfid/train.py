import argparse
import torch
import json
import os
from pathlib import Path
from vlt5.tokenization import VLT5Tokenizer
from utils.log import init_logger
from pathlib import Path
from utils.distributed import init_distributed_mode, init_signal_handler
import torch
from modeling import optim
from data.dataset import OKVQADataset
from data.collators import OKVQAGenerationCollator
from modeling.modeling_mmfid import MMFiD, MMFiDConfig
from modeling.utils import load_checkpoint, average_main, save_checkpoint
from eval.metrics import ems
from torch.utils.data import DataLoader, RandomSampler, DistributedSampler, SequentialSampler
from eval.vqa import VQA
from eval.vqa_eval import VQAEval
import tqdm

parser = argparse.ArgumentParser()

parser.add_argument("--train_data", required = True, help="the training data")
parser.add_argument("--val_data", required = True, help="the validation data")
parser.add_argument("--ann_file", required = False, help="the address of the ann file for OK-VQA validation")
parser.add_argument("--qs_file", required = False, help="the address of the qs file for OK-VQA validation")
parser.add_argument("--n_context", type = int, required = True, help="number of supporting documnets for each question")
parser.add_argument("--do_train", action='store_true', help="perform training")
parser.add_argument("--do_validation", action='store_true', help="perform validation")
parser.add_argument("--max_length_input", type = int, default = 400, help="maximum input length")
parser.add_argument("--max_length_output", type = int, default = 16, help="maximum output length")
parser.add_argument('--name', type=str, default='experiment_name', help='name of the experiment')
parser.add_argument('--checkpoint_dir', type=str, default='./checkpoint/', help='models are saved here')
parser.add_argument('--model_path', type=str, default='none', help='path for a checkpoint to start training from that')
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
parser.add_argument("--use_prefix", action='store_true', help="whether to use a prefix for input or not")
parser.add_argument("--num_beams", type=int, default=2, help="number of beams for text generation")
parser.add_argument("--gpt3_answers", default="", help="address to a file containing gpt3 answers for each question")
parser.add_argument("--okvqa_acc", action='store_true', help="report okvqa accuracy metric")

def train(opts, model, optimizer, scheduler, step, dataset, collator, checkpoint_path, test_dataset, logger):
    
    if opts.is_main:
        try:
            tb_logger = torch.utils.tensorboard.SummaryWriter(Path(opts.checkpoint_dir)/opts.name)
        except:
           tb_logger = None
           logger.warning('Tensorboard is not available.')

    torch.manual_seed(opts.global_rank + opts.seed) #different seed for different sampling depending on global_rank
    train_sampler = DistributedSampler(dataset, num_replicas=opts.n_gpu_per_node, rank=opts.local_rank)
    train_dataloader = DataLoader(
        dataset,
        sampler = train_sampler,
        batch_size = opts.per_gpu_batch_size,
        drop_last = True,
        num_workers = 10,
        collate_fn = collator
    )

    loss, curr_loss = 0.0, 0.0
    epoch = 1
    model.train()
    bar = tqdm.tqdm(total=opts.total_steps)
    while step < opts.total_steps:
        epoch += 1
        for i, batch in enumerate(train_dataloader):
            step += 1
            bar.update(1)
            train_loss = model(
                input_ids = batch['input_ids'].cuda(),
                attention_mask = batch['attention_mask'].cuda(),
                vis_inputs = (batch['vis_inputs'][0].cuda(), batch['vis_inputs'][1].cuda()),
                labels = batch['labels'].cuda()
            )['loss']
            
            labels_mask = batch['labels_mask'].cuda()
            B, _ = labels_mask.shape
            train_loss = train_loss.view(B, -1) * labels_mask
            train_loss = train_loss.sum(dim=1) / labels_mask.sum(dim=1).clamp(min=1)
            train_loss = train_loss.mean()
            train_loss.backward()

            if step % opts.accumulation_steps == 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), opts.clip)
                optimizer.step()
                scheduler.step()
                model.zero_grad()

            train_loss = average_main(train_loss, opts)
            curr_loss += train_loss.item()
            
            if step % opts.eval_freq == 0:
                if opts.is_main:
                    em, acc = evaluate(model, test_dataset, collator, opts, step, logger)
                    log = f"{step} / {opts.total_steps} |"
                    log += f"train: {curr_loss/opts.eval_freq:.4f} |"
                    log += f"EM: {em:.4f} |"
                    log += f"Accuracy: {acc:.4f}"
                    logger.info(log)
                    if tb_logger is not None:
                        tb_logger.add_scalar("Training", curr_loss / (opts.eval_freq), step)
                    curr_loss = 0.
                    save_checkpoint(model, optimizer, scheduler, step, opts, checkpoint_path, f"step-{step}")

            if opts.is_main and step % opts.save_freq == 0:
                save_checkpoint(model, optimizer, scheduler, step, opts, checkpoint_path, f"step-{step}")
            if step > opts.total_steps:
                break
    save_checkpoint(model, optimizer, scheduler, step, opts, checkpoint_path, f"step-{step}")

def evaluate(model, dataset, collator, opt, step, logger):
    sampler = SequentialSampler(dataset) if opts.is_distributed else RandomSampler(dataset)
    dataloader = DataLoader(dataset,
        sampler=sampler,
        batch_size=opt.per_gpu_batch_size,
        drop_last=False,
        num_workers=10,
        collate_fn=collator
    )
    model.eval()
    with torch.no_grad():
        preds = []
        golds = []
        probs = []
        ids = []
        logger.info("Evaluation Started")
        for i, batch in enumerate(tqdm.tqdm(dataloader)):
            ids.extend(batch['question_ids'])
            outputs = model.module.generate(input_ids=batch['input_ids'].cuda(), attention_mask = batch['attention_mask'].cuda(), vis_inputs = (batch['vis_inputs'][0].cuda(), batch['vis_inputs'][1].cuda()), num_beams = opt.num_beams, output_scores=True, return_dict_in_generate=True)
            output = collator.tokenizer.batch_decode(outputs.sequences, skip_special_tokens = True)
            probs.extend(outputs.sequences_scores.cpu().tolist())
            preds.extend(output)
            golds.extend(batch['golds'])
    scores = []
    for pred, gold in zip(preds, golds):
        scores.append(ems(pred, gold))
    
    em = sum(scores) / len(scores)
    
    checkpoint_path = Path(opts.checkpoint_dir)/opts.name/"predictions"
    res_file = os.path.join(checkpoint_path, f"results_{step}.json")
    checkpoint_path.mkdir(parents = True, exist_ok = True)

    with open(res_file, "w") as file:
        res = [{"answer" : pred, "question_id" : qid, "likelihood" : prob} for pred, qid, prob in zip(preds, ids, probs)]
        json.dump(res, file)

    model.train()

    if opts.okvqa_acc:
        vqa = VQA(opts.ann_file, opts.qs_file)
        vqares = vqa.loadRes(res_file, opts.qs_file)
        eval_vqa = VQAEval(vqa, vqares)
        eval_vqa.evaluate()
        return em, eval_vqa.accuracy['overall']
    
    return em, em

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


    logger = init_logger(
        opts.is_main,
        opts.is_distributed,
        checkpoint_path / 'run.log'
    )

    logger.info(opts)

    with open(opts.train_data) as file:
        train_examples = [json.loads(line.strip()) for line in file if line.strip()]
    
    with open(opts.val_data) as file:
        test_examples = [json.loads(line.strip()) for line in file if line.strip()]
    
    tokenizer = VLT5Tokenizer.from_pretrained('t5-base')

    if checkpoint_exists and opts.do_train:
        model, optimizer, scheduler, checkpoint_opts, step = load_checkpoint(MMFiD, os.path.join(checkpoint_path, "checkpoint", "latest"), opts)
    elif opts.do_train:
        config = MMFiDConfig.from_pretrained('t5-base')
        state_dict = torch.load(opts.model_path, map_location='cpu')
        model = MMFiD(config=config)
        model.load_VLT5(state_dict)
        optimizer, scheduler = optim.set_optim(opts, model)
        step = 0
    elif opts.do_validation:
        model = MMFiD.from_pretrained(opts.model_path)
    
    model = model.to(opts.local_rank)
    
    if opts.is_distributed:
        model = torch.nn.parallel.DistributedDataParallel(
            model,
            device_ids=[opts.local_rank],
            output_device=opts.local_rank,
            find_unused_parameters=False,
        )

    train_dataset = OKVQADataset(train_examples, n_context = opts.n_context, gpt3_answers_addr = opts.gpt3_answers)
    val_dataset = OKVQADataset(test_examples, n_context = opts.n_context, test = True, gpt3_answers_addr = opts.gpt3_answers)

    collator = OKVQAGenerationCollator(tokenizer, opts.max_length_input, opts.max_length_output, opts.image_feats_addr, opts.image_feats_mapping_addr, opts.use_prefix)
    
    if opts.do_train:
        train(opts, model, optimizer, scheduler, step, train_dataset, collator, checkpoint_path, val_dataset, logger)
    
    if opts.do_validation and opts.is_main:
        em, acc = evaluate(model, val_dataset, collator, opts, "validation", logger)
        log = f"EM: {em:.4f} |"
        log += f"Accuracy: {acc:.4f}"
        logger.info(log)

