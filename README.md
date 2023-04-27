# A Symmetric Dual Encoding Dense Retrieval Framework for Knowledge-Intensive Visual Question Answering

This repository contains the codes for the SIGIR23 paper: [A Symmetric Dual Encoding Dense Retrieval Framework for Knowledge-Intensive Visual Question Answering](https://arxiv.org/abs/2304.13649).

<table>
<tr>
<td><img src="images/example.png?raw=True" width="400"/></td>
<td>An example KI-VQA question; answering it requires external knowledge.<br>
<sup><sub><a href="https://www.flickr.com/photos/zrimshots/2788695458">Image copyright zrim [https://www.flickr.com/photos/zrimshots/2788695458]</a></sub></sup></td>
</tr>
</table>

This is the replication package for the paper:

> Alireza Salemi, Juan Altmayer Pizzorno, and Hamed Zamani. A Symmetric Dual Encoding Dense Retrieval Framework for Knowledge-Intensive Visual Question Answering. In Proceedings of the 46th Int’l ACM SIGIR Conference on Research and Development in Information Retrieval (SIGIR ’23). Taipei, Taiwan, July 2023. [[BibTeX]](paper.bib)

[![License](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](LICENSE)

Knowledge-Intensive Visual Question Answering (KI-VQA) refers to answering a question about an image whose answer does not lie in the image. This paper presents a new pipeline for KI-VQA tasks, consisting of a retriever and a reader. First, we introduce DEDR, a symmetric dual encoding dense retrieval framework in which documents and queries are encoded into a shared embedding space using uni-modal (textual) and multi-modal encoders. We introduce an iterative knowledge distillation approach that bridges the gap between the representation spaces in these two encoders. Extensive evaluation on two well-established KI-VQA datasets, i.e., OK-VQA and FVQA, suggests that DEDR outperforms state-of-the-art baselines by 11.6% and 30.9% on OK-VQA and FVQA, respectively. Utilizing the passages retrieved by DEDR, we further introduce MM-FiD, an encoder-decoder multi-modal fusion-in-decoder model, for generating a textual answer for KI-VQA tasks. MM-FiD encodes the question, the image, and each retrieved passage separately and uses all passages jointly in its decoder. Compared to competitive baselines in the literature, this approach leads to 5.5% and 8.5% improvements in terms of question answering accuracy on OK-VQA and FVQA, respectively.

## Data

This section explains the datasets and files used to train the models.

### Data Format


1. DEDR:

  - train_data/validation_data/test_data: Inputs to each task should be JSONL (JSON lines) file, each line following this format:
```
{
  "question_id": "/*id of the sample*/", 
  "pos_passage": { /*a positive passage for the question*/
    "passage": "/*text*/"
  },
  "answers": [/*list of possible answers*/],
  "question": "/question about the image*/",
  "image_id": "/*id of the image in image dataset*/",
  "neg_passage": { /*a negative passage for the question*/
    "passage": "/*text*/"
  }
}
```
  - val_passages/all_blocks_file: This file should be be JSONL (JSON lines) file, each line following the this format:

```
{
  "text": "/*passage text*/",
  "id": /*id of this passage in all passages*/
}
```
  - image_feats_addr/image_feats_mapping_addr: In order to create the image_feats file, you can use [this](https://github.com/huggingface/transformers/blob/main/examples/research_projects/lxmert/extracting_data.py) script provided by HuggingFace. In order to create image_feats_mapping, which is used to speedup the search in image_feats dataset, we have provided the `shared/image_feats_to_ids.py` script for this purpose.

  - captions: We use ExpansionNet v2 to generate captions, which can be found [here](https://github.com/jchenghu/ExpansionNet_v2). However, if you want to use your own captions with our approach, the caption file should be a json file that follows this format:

```
{
  "/*image 1 id*/": ["/*word 1 caption image 1*/", "/*word 2 caption image 1*/", ..., "/*the last word caption image 1*/"],
  "/*image 2 id*/": ["/*word 1 caption image 2*/", "/*word 2 caption image 2*/", ..., "/*the last word caption image 2*/"],
  ...
  "/*image n id*/": ["/*word 1 caption image n*/", "/*word 2 caption image n*/", ..., "/*the last word caption image n*/"],
  
}
```
  
  - passage_id_to_line_id_file (only for OK-VQA dataset): This file is used to create a mapping between the passage ids and the line number in the file they are stored.

  - ann_file/ques_file (only for OK-VQA dataset): We use the files provided by OK-VQA dataset [here](https://okvqa.allenai.org/download.html).
  
  - qrel files (only for FVQA or any dataset that does not use dynamic evaluation): We use the original format of qrels file introduced by TREC.


2. Answer generation:


  - train_data/val_data/test_data: Inputs to each task should be JSONL (JSON lines) file, each line following this format:
```
{
  "question_id": "/*id of the sample*/", 
  "answers": [/*list of possible answers*/],
  "question": "/question about the image to the model*/",
  "image_id": "/*id of the image in image dataset*/",
  "ctx": [/*list of supporting passages as text*/]
}
```

  - image_feats_addr/image_feats_mapping_addr: In order to create the image_feats file, you can use [this](https://github.com/huggingface/transformers/blob/main/examples/research_projects/lxmert/extracting_data.py) script provided by HuggingFace. In order to create image_feats_mapping, which is used to speedup the search in image_feats dataset, we have provided the `utils/image_feats_to_ids.py` script for this purpose.

  - ann_file/ques_file (only for OK-VQA dataset): We use the files provided by OK-VQA dataset [here](https://okvqa.allenai.org/download.html).

  - gpt3_answers (only if you want to use an LLM data in addition to retrieved docs): This should be a json file that follows this format:

```
{
  "/*question 1 id*/" : "/*answer to question 1* with a prompt/",
  "/*question 2 id*/" : "/*answer to question 2 with a prompt*/",
  ...
  "/*question n id*/" : "/*answer to question n with a prompt*/",
  
}
```

### DEDE Data

1. OK-VQA:

  - train_data: [download](https://ciir.cs.umass.edu/downloads/okvqa/data/train2014_pairs_cap_combine_sum.txt)
  - test_data: [download](https://ciir.cs.umass.edu/downloads/okvqa/data/train2014_pairs_cap_combine_sum.txt)
  - val_data: [download](https://ciir.cs.umass.edu/downloads/okvqa/data/val2014_pairs_cap_combine_sum.txt)
  - val_passages: [download](https://ciir.cs.umass.edu/downloads/okvqa/data/val2014_blocks_cap_combine_sum.txt)
  - all_blocks_file: [download](https://ciir.cs.umass.edu/downloads/ORConvQA/all_blocks.txt.gz)
  - passage_id_to_line_id_file: [download](https://ciir.cs.umass.edu/downloads/okvqa/passage_id_to_line_id.json)
  - ann_file/ques_file: [download web page](https://okvqa.allenai.org/download.html)
  - image_feats: [download](https://ciir.cs.umass.edu/downloads/DEDR/image_feats/ok-vqa/mscoco2014.datasets)
  - image_feats_mapping: [download](https://ciir.cs.umass.edu/downloads/DEDR/image_feats/ok-vqa/image_feats_mapping_to_dataset.json)
  - captions: [download](https://ciir.cs.umass.edu/downloads/DEDR/captions/captions_okvqa.json)

2. FVQA:

  - train_data: [download](https://ciir.cs.umass.edu/downloads/DEDR/data/fvqa/train_new_ids.jsonl)
  - test_data: [download](https://ciir.cs.umass.edu/downloads/DEDR/data/fvqa/test_new_ids.jsonl)
  - val_data: [download](https://ciir.cs.umass.edu/downloads/DEDR/data/fvqa/dev_new_ids.jsonl)
  - val_passages: [download](https://ciir.cs.umass.edu/downloads/DEDR/data/fvqa/facts_new_ids.jsonl)
  - image_feats: [download](https://ciir.cs.umass.edu/downloads/DEDR/image_feats/fvqa/fvqa.datasets)
  - image_feats_mapping: [download](https://ciir.cs.umass.edu/downloads/DEDR/image_feats/fvqa/fvqa_id_to_index_mapping.json)
  - captions: [download](https://ciir.cs.umass.edu/downloads/DEDR/captions/captions_fvqa.json)
  - val_qrels: [download](https://ciir.cs.umass.edu/downloads/DEDR/data/fvqa/dev_qrel_new_ids.jsonl)
  - test_qrels: [download](https://ciir.cs.umass.edu/downloads/DEDR/data/fvqa/test_qrel_new_ids.jsonl)

### MM-FiD Data

1. OK-VQA

  - train_data: [download](https://ciir.cs.umass.edu/downloads/DEDR/mmfid/data/ok-vqa/train_retrieved_with_answers_captbert_lxmert_seprate_64_all_answers.jsonl)
  - test_data: [download](https://ciir.cs.umass.edu/downloads/DEDR/mmfid/data/ok-vqa/dev_retrieved_with_answers_captbert_lxmert_seprate_64_all_answers.jsonl)
  - ann_file/ques_file: [download web page](https://okvqa.allenai.org/download.html)
  - image_feats: [download](https://ciir.cs.umass.edu/downloads/DEDR/image_feats/ok-vqa/mscoco2014.datasets)
  - image_feats_mapping: [download](https://ciir.cs.umass.edu/downloads/DEDR/image_feats/ok-vqa/image_feats_mapping_to_dataset.json)
  - gpt3_answers: [download](https://ciir.cs.umass.edu/downloads/DEDR/mmfid/data/ok-vqa/gpt3_answers_new_format.json)

2. FVQA
  - train_data: [download fold1](https://ciir.cs.umass.edu/downloads/DEDR/mmfid/data/fvqa/fold_1/train.jsonl) [download fold2](https://ciir.cs.umass.edu/downloads/DEDR/mmfid/data/fvqa/fold_2/train.jsonl) [download fold3](https://ciir.cs.umass.edu/downloads/DEDR/mmfid/data/fvqa/fold_3/train.jsonl) [download fold4](https://ciir.cs.umass.edu/downloads/DEDR/mmfid/data/fvqa/fold_4/train.jsonl) [download fold5](https://ciir.cs.umass.edu/downloads/DEDR/mmfid/data/fvqa/fold_5/train.jsonl)
  - test_data: [download fold1](https://ciir.cs.umass.edu/downloads/DEDR/mmfid/data/fvqa/fold_1/test.jsonl) [download fold2](https://ciir.cs.umass.edu/downloads/DEDR/mmfid/data/fvqa/fold_2/test.jsonl) [download fold3](https://ciir.cs.umass.edu/downloads/DEDR/mmfid/data/fvqa/fold_3/test.jsonl) [download fold4](https://ciir.cs.umass.edu/downloads/DEDR/mmfid/data/fvqa/fold_4/test.jsonl) [download fold5](https://ciir.cs.umass.edu/downloads/DEDR/mmfid/data/fvqa/fold_5/test.jsonl)
  - image_feats: [download](https://ciir.cs.umass.edu/downloads/DEDR/image_feats/fvqa/fvqa.datasets)
  - image_feats_mapping: [download](https://ciir.cs.umass.edu/downloads/DEDR/image_feats/fvqa/fvqa_id_to_index_mapping.json)

### Checkpoints

Will be added soon!
  
## Dual Encoding Dense Retrieval (DEDR)

<table>
<tr>
<td><img src="images/dedr.drawio-new.png?raw=True" width="400"/></td>
</tr>
<tr>
<td>The training and inference procedure in the DEDR framework. DEDR first trains uni-modal and multi-modal encoders in isolation (left), then uses iterative knowledge distillation to adjust both representation spaces (middle). At inference, DEDR uses the aggregation of both encodings to construct a symmetric dual encoding dense retriever (right).<br>
</tr>
</table>

### Installation

To use DEDR, you first need to install its dependencies:

```
pip install -r dedr/requirements.txt
```

### Training

Training using DEDR is a multi-step process.

1. Training $E_T$: In this step, you train a textual model, such as BERT, for retrieval.

```
NGPU=/*Number of GPUs*/ python -m torch.distributed.launch --nproc_per_node=/*Number of GPUs*/ dedr/train_ranker.py \
  --train_data /*Address to the retrieval training data*/ \
  --val_data /*Address to the retrieval validation data*/ \
  --val_passages /*Address to the validation knowledge source*/ \
  --val_rep_batch_size /*Batch size for generating knowledge source representation*/ \
  --top_k_retrieve /*Number of retrieved documents*/ \
  --eval_freq /*The frequency of Evaluations should be given as the number of steps*/ \
  --name /*Output dir name*/ \
  --per_gpu_batch_size /*Per GPU batch size for training*/ \
  --max_length /*Maximum input length of model*/ \
  --accumulation_steps /*Number of gradient accumulation steps*/ \
  --total_steps /*Total training steps*/ \
  --weight_decay /*weight decay*/ \
  --warmup_steps /*Number of warmup steps*/ \
  --scheduler /*Learning rate scheduler type*/ \
  --optim /*Optimizer type*/ \
  --lr /*Learning rate*/ \
  --model_type E_T \
  --neg_type other_pos+all_neg \
  --image_feats_addr /*Address to the image features file*/ \
  --image_feats_mapping_addr /*Address to the index to image features file*/ \
  --caption_address /*Address to the caption file*/ \
  --do_train \
  /**IMPORTANT: The following parameters are only used for FVQA**/
  --qrels_addr /*Address to the qrel file, this is only used with FVQA dataset. For OK-VQA we use dynamic qrels*/ \
  /**IMPORTANT: The following parameters are only used for OKVQA**/
  --ann_file /*Address to Annotation file in OK-VQA dataset for dynamic eval*/ \
  --ques_file /*Address to Question file in OK-VQA dataset for dynamic eval*/ \
  --passage_id_to_line_id_file /*Address to maping between passage id and line id in knowledge source*/ \
  --all_blocks_file /*Address to the knowledge source, it can be the same as validation knowledge source or different*/ \
```

2. Training $E_{MM}$: In this step, you train a multi-modal model, such as LXMERT, for retrieval.

```
NGPU=/*Number of GPUs*/ python -m torch.distributed.launch --nproc_per_node=/*Number of GPUs*/ dedr/train_ranker.py \
  --train_data /*Address to the retrieval training data*/ \
  --val_data /*Address to the retrieval validation data*/ \
  --val_passages /*Address to the validation knowledge source*/ \
  --val_rep_batch_size /*Batch size for generating knowledge source representation*/ \
  --top_k_retrieve /*Number of retrieved documents*/ \
  --eval_freq /*The frequency of Evaluations should be given as the number of steps*/ \
  --name /*Output dir name*/ \
  --per_gpu_batch_size /*Per GPU batch size for training*/ \
  --max_length /*Maximum input length of model*/ \
  --accumulation_steps /*Number of gradient accumulation steps*/ \
  --total_steps /*Total training steps*/ \
  --weight_decay /*weight decay*/ \
  --warmup_steps /*Number of warmup steps*/ \
  --scheduler /*Learning rate scheduler type*/ \
  --optim /*Optimizer type*/ \
  --lr /*Learning rate*/ \
  --model_type E_MM \
  --neg_type other_pos+all_neg \
  --image_feats_addr /*Address to the image features file*/ \
  --image_feats_mapping_addr /*Address to the index to image features file*/ \
  --caption_address /*Address to the caption file*/ \
  --do_train \
  /**IMPORTANT: The following parameters are only used for FVQA**/
  --qrels_addr /*Address to the qrel file, this is only used with FVQA dataset. For OK-VQA we use dynamic qrels*/ \
  /**IMPORTANT: The following parameters are only used for OKVQA**/
  --ann_file /*Address to Annotation file in OK-VQA dataset for dynamic eval*/ \
  --ques_file /*Address to Question file in OK-VQA dataset for dynamic eval*/ \
  --passage_id_to_line_id_file /*Address to maping between passage id and line id in knowledge source*/ \
  --all_blocks_file /*Address to the knowledge source, it can be the same as validation knowledge source or different*/ \
```

3. Iterative knowledge distillation between modalities: In this step, we iteratively distill knowledge from one modality to another. We suggest starting with the better model among $E_T$ and $E_MM$ as the teacher and the weaker as the student. In the following codes, we assume that $E_T$ is the teacher.

iteration 1:

```
NGPU=/*Number of GPUs*/ python -m torch.distributed.launch --nproc_per_node=/*Number of GPUs*/ dedr/distill_ranker_to_ranker.py \
  --student_model_type E_MM \
  --student_model_path /*Address to the checkpoint of the model that is used as student*/ \
  --teacher_model_type E_T \
  --teacher_model_path /*Address to the checkpoint of the model that is used as teacher*/ \
  --train_data /*Address to the retrieval training data*/ \
  --val_data /*Address to the retrieval validation data*/ \
  --val_passages /*Address to the validation knowledge source*/ \
  --val_rep_batch_size /*Batch size for generating knowledge source representation*/ \
  --top_k_retrieve /*Number of retrieved documents*/ \
  --eval_freq /*The frequency of Evaluations should be given as the number of steps*/ \
  --name /*Output dir name*/ \
  --per_gpu_batch_size /*Per GPU batch size for training*/ \
  --max_length /*Maximum input length of model*/ \
  --accumulation_steps /*Number of gradient accumulation steps*/ \
  --total_steps /*Total training steps*/ \
  --weight_decay /*weight decay*/ \
  --warmup_steps /*Number of warmup steps*/ \
  --scheduler /*Learning rate scheduler type*/ \
  --optim /*Optimizer type*/ \
  --lr /*Learning rate*/ \
  --neg_type other_pos+all_neg \
  --image_feats_addr /*Address to the image features file*/ \
  --image_feats_mapping_addr /*Address to the index to image features file*/ \
  --caption_address /*Address to the caption file*/ \
  --do_train \
  /**IMPORTANT: The following parameters are only used for FVQA**/
  --qrels_addr /*Address to the qrel file, this is only used with FVQA dataset. For OK-VQA we use dynamic qrels*/ \
  /**IMPORTANT: The following parameters are only used for OKVQA**/
  --ann_file /*Address to Annotation file in OK-VQA dataset for dynamic eval*/ \
  --ques_file /*Address to Question file in OK-VQA dataset for dynamic eval*/ \
  --passage_id_to_line_id_file /*Address to maping between passage id and line id in knowledge source*/ \
  --all_blocks_file /*Address to the knowledge source, it can be the same as validation knowledge source or different*/ \
```

iteration 2:

```
NGPU=/*Number of GPUs*/ python -m torch.distributed.launch --nproc_per_node=/*Number of GPUs*/ dedr/distill_ranker_to_ranker.py \
  --student_model_type E_T \
  --student_model_path /*Address to the checkpoint of the model that was used as teacher in previous iteration*/ \
  --teacher_model_type E_MM \
  --teacher_model_path /*Address to the checkpoint of the best student in previous iteration*/ \
  --train_data /*Address to the retrieval training data*/ \
  --val_data /*Address to the retrieval validation data*/ \
  --val_passages /*Address to the validation knowledge source*/ \
  --val_rep_batch_size /*Batch size for generating knowledge source representation*/ \
  --top_k_retrieve /*Number of retrieved documents*/ \
  --eval_freq /*The frequency of Evaluations should be given as the number of steps*/ \
  --name /*Output dir name*/ \
  --per_gpu_batch_size /*Per GPU batch size for training*/ \
  --max_length /*Maximum input length of model*/ \
  --accumulation_steps /*Number of gradient accumulation steps*/ \
  --total_steps /*Total training steps*/ \
  --weight_decay /*weight decay*/ \
  --warmup_steps /*Number of warmup steps*/ \
  --scheduler /*Learning rate scheduler type*/ \
  --optim /*Optimizer type*/ \
  --lr /*Learning rate*/ \
  --neg_type other_pos+all_neg \
  --image_feats_addr /*Address to the image features file*/ \
  --image_feats_mapping_addr /*Address to the index to image features file*/ \
  --caption_address /*Address to the caption file*/ \
  --do_train \
  /**IMPORTANT: The following parameters are only used for FVQA**/
  --qrels_addr /*Address to the qrel file, this is only used with FVQA dataset. For OK-VQA we use dynamic qrels*/ \
  /**IMPORTANT: The following parameters are only used for OKVQA**/
  --ann_file /*Address to Annotation file in OK-VQA dataset for dynamic eval*/ \
  --ques_file /*Address to Question file in OK-VQA dataset for dynamic eval*/ \
  --passage_id_to_line_id_file /*Address to maping between passage id and line id in knowledge source*/ \
  --all_blocks_file /*Address to the knowledge source, it can be the same as validation knowledge source or different*/ \
```

iteration 3:

```
NGPU=/*Number of GPUs*/ python -m torch.distributed.launch --nproc_per_node=/*Number of GPUs*/ dedr/distill_ranker_to_ranker.py \
  --student_model_type E_MM \
  --student_model_path /*Address to the checkpoint of the model that was used as teacher in previous iteration*/ \
  --teacher_model_type E_T \
  --teacher_model_path /*Address to the checkpoint of the best student in previous iteration*/ \
  --train_data /*Address to the retrieval training data*/ \
  --val_data /*Address to the retrieval validation data*/ \
  --val_passages /*Address to the validation knowledge source*/ \
  --val_rep_batch_size /*Batch size for generating knowledge source representation*/ \
  --top_k_retrieve /*Number of retrieved documents*/ \
  --eval_freq /*The frequency of Evaluations should be given as the number of steps*/ \
  --name /*Output dir name*/ \
  --per_gpu_batch_size /*Per GPU batch size for training*/ \
  --max_length /*Maximum input length of model*/ \
  --accumulation_steps /*Number of gradient accumulation steps*/ \
  --total_steps /*Total training steps*/ \
  --weight_decay /*weight decay*/ \
  --warmup_steps /*Number of warmup steps*/ \
  --scheduler /*Learning rate scheduler type*/ \
  --optim /*Optimizer type*/ \
  --lr /*Learning rate*/ \
  --neg_type other_pos+all_neg \
  --image_feats_addr /*Address to the image features file*/ \
  --image_feats_mapping_addr /*Address to the index to image features file*/ \
  --caption_address /*Address to the caption file*/ \
  --do_train \
  /**IMPORTANT: The following parameters are only used for FVQA**/
  --qrels_addr /*Address to the qrel file, this is only used with FVQA dataset. For OK-VQA we use dynamic qrels*/ \
  /**IMPORTANT: The following parameters are only used for OKVQA**/
  --ann_file /*Address to Annotation file in OK-VQA dataset for dynamic eval*/ \
  --ques_file /*Address to Question file in OK-VQA dataset for dynamic eval*/ \
  --passage_id_to_line_id_file /*Address to maping between passage id and line id in knowledge source*/ \
  --all_blocks_file /*Address to the knowledge source, it can be the same as validation knowledge source or different*/ \
```

iteration 4:

```
NGPU=/*Number of GPUs*/ python -m torch.distributed.launch --nproc_per_node=/*Number of GPUs*/ dedr/distill_ranker_to_ranker.py \
  --student_model_type E_T \
  --student_model_path /*Address to the checkpoint of the model that was used as teacher in previous iteration*/ \
  --teacher_model_type E_MM \
  --teacher_model_path /*Address to the checkpoint of the best student in previous iteration*/ \
  --train_data /*Address to the retrieval training data*/ \
  --val_data /*Address to the retrieval validation data*/ \
  --val_passages /*Address to the validation knowledge source*/ \
  --val_rep_batch_size /*Batch size for generating knowledge source representation*/ \
  --top_k_retrieve /*Number of retrieved documents*/ \
  --eval_freq /*The frequency of Evaluations should be given as the number of steps*/ \
  --name /*Output dir name*/ \
  --per_gpu_batch_size /*Per GPU batch size for training*/ \
  --max_length /*Maximum input length of model*/ \
  --accumulation_steps /*Number of gradient accumulation steps*/ \
  --total_steps /*Total training steps*/ \
  --weight_decay /*weight decay*/ \
  --warmup_steps /*Number of warmup steps*/ \
  --scheduler /*Learning rate scheduler type*/ \
  --optim /*Optimizer type*/ \
  --lr /*Learning rate*/ \
  --neg_type other_pos+all_neg \
  --image_feats_addr /*Address to the image features file*/ \
  --image_feats_mapping_addr /*Address to the index to image features file*/ \
  --caption_address /*Address to the caption file*/ \
  --do_train \
  /**IMPORTANT: The following parameters are only used for FVQA**/
  --qrels_addr /*Address to the qrel file, this is only used with FVQA dataset. For OK-VQA we use dynamic qrels*/ \
  /**IMPORTANT: The following parameters are only used for OKVQA**/
  --ann_file /*Address to Annotation file in OK-VQA dataset for dynamic eval*/ \
  --ques_file /*Address to Question file in OK-VQA dataset for dynamic eval*/ \
  --passage_id_to_line_id_file /*Address to maping between passage id and line id in knowledge source*/ \
  --all_blocks_file /*Address to the knowledge source, it can be the same as validation knowledge source or different*/ \
  
```

You can continue training for more than four iterations. However, our experiments show that it does not have a considerable gain.

### Indexing (optional)

You can first index and then evaluate the model or create an index during evaluation. We suggest that you first create an index because it is time-consuming for large knowledge sources. Additionally, our indexing script provides sharding, which can speed up the process significantly.

```
python dedr/index.py \
  --model_path /*Address to the best E_MM model after iterative distillation*/ \
  --model2_path /*Address to the best E_T model after iterative distillation*/ \
  --model_type DEDR \
  --passages /*Address to the knowledge source*/ \
  --output /*Output index file name*/ \
  --shard_id /*Id of the current shard in a zero-based numbering*/ \
  --num_shards /*Total number of shards*/ \
  --batch_size /*Batch size for generating representation*/ \
  --device /*Name of the device to be used for example "cuda:0"*/ \
```

### Evaluation

You can use the following code to evaluate the DEDR trained model:

```
NGPU=/*Number of GPUs*/ python -m torch.distributed.launch --nproc_per_node=/*Number of GPUs*/ dedr/train_ranker.py \
  --model_path /*Address to the best E_MM model after iterative distillation*/ \
  --model2_path /*Address to the best E_T model after iterative distillation*/ \
  --train_data /*Address to the retrieval training data*/ \
  --val_data /*Address to the retrieval validation data*/ \
  --val_passages /*Address to the validation knowledge source*/ \
  --val_rep_batch_size /*Batch size for generating knowledge source representation*/ \
  --top_k_retrieve /*Number of retrieved documents*/ \
  --name /*Output dir name*/ \
  --max_length /*Maximum input length of model*/ \
  --model_type DEDR \
  --neg_type other_pos+all_neg \
  --image_feats_addr /*Address to the image features file*/ \
  --image_feats_mapping_addr /*Address to the index to image features file*/ \
  --caption_address /*Address to the caption file*/ \
  --do_validation \
  /**IMPORTANT: The following parameters are only used for FVQA**/
  --qrels_addr /*Address to the qrel file, this is only used with FVQA dataset. For OK-VQA we use dynamic qrels*/ \
  /**IMPORTANT: The following parameters are only used for OKVQA**/
  --ann_file /*Address to Annotation file in OK-VQA dataset for dynamic eval*/ \
  --ques_file /*Address to Question file in OK-VQA dataset for dynamic eval*/ \
  --passage_id_to_line_id_file /*Address to maping between passage id and line id in knowledge source*/ \
  --all_blocks_file /*Address to the knowledge source, it can be the same as validation knowledge source or different*/ \
  /**IMPORTANT: The following parameter is only used when you do indexing**/
  --index_addr /*Address to the folder contain all indexes*/ \
  
```

### Retrieval

You can use the following code to retrieve from your knowledge source:

```
python dedr/retrieve.py \
  --model_path /*Address to the best E_MM model after iterative distillation*/ \
  --model2_path /*Address to the best E_T model after iterative distillation*/ \
  --model_type DEDR \
  --passages /*Address to the knowledge source*/ \
  --output_file /*Address to the output file*/ \
  --top_k_retrieve /*Number of retrieved documents*/ \
  --input_queries /*Address to the input queries, they should have the same format as validation/test queries*/ \
  --image_feats_addr /*Address to the image features file*/ \
  --image_feats_mapping_addr /*Address to the index to image features file*/ \
  --caption_address /*Address to the caption file*/ \
  --device /*Name of the device to be used for example "cuda:0"*/ \
```

## Multi-modal Fusion in Decoder (MM-FiD)

<table>
<tr>
<td><img src="images/multi-fid.drawio.png?raw=True" width="400"/></td>
</tr>
<tr>
<td>TThe architecture of MM-FiD. It uses multi-modal encoder to encode each question-image-passage triplet separately and then concatenates their encodings as input to the decoder for knowledge aggregation and answer generation.<br>
</tr>
</table>

### Installation

To use MM-FiD, you first need to install its dependencies:

```
pip install -r mmfid/requirements.txt
```

### Training

You can use the following code to train MM-FiD:

```
NGPU=/*Number of available GPUs*/ python -m torch.distributed.launch --nproc_per_node=/*Number of available GPUs*/ train.py \
  --train_data /*Address to the training data for answer generation task*/ \
  --val_data /*Address to the validation data for answer generation task*/ \
  --eval_freq /*The frequency of Evaluations should be given as the number of steps*/ \
  --name /*Output dir name*/ \
  ---per_gpu_batch_size /*Per GPU batch size for training*/ \
  --total_steps /*Total training steps*/ \
  --accumulation_steps /*Number of gradient accumulation steps*/ \
  --warmup_steps /*Number of warmup steps*/ \
  --scheduler /*Learning rate scheduler type*/ \
  --optim /*Optimizer type*/ \
  --lr /*Learning rate*/ \
  --max_length_input /*Maximum length for the input of the model*/ \
  --max_length_output /*Maximum length for the output of the model*/ \
  --weight_decay/*weight decay*/ \
  --image_feats_addr /*Address to the image features file*/ \
  --image_feats_mapping_addr /*Address to the index to image features file*/ \
  --do_train \
  --model_path /*The path to the initial checkpoint of VLT5*/ \
  --n_context /*Number of documents to be used in generating answer*/ \
  --num_beams /*Number of beams in beam search*/ \
  /**IMPORTANT: The following parameters are only used for OKVQA when you want to use GPT3**/
  --gpt3_answers /*Address to the answers generated by an LLM*/ \
  /**IMPORTANT: The following parameters are only used for OKVQA**/
  --okvqa_acc /*Enables using OK-VQA original evaluation metric*/ \
  --ann_file /*Address to Annotation file in OK-VQA dataset for dynamic eval*/ \
  --ques_file /*Address to Question file in OK-VQA dataset for dynamic eval*/ \
```

You can download the VLT5 weights from [this](https://github.com/j-min/VL-T5) repo.

### Evaluation

```
NGPU=/*Number of available GPUs*/ python -m torch.distributed.launch --nproc_per_node=/*Number of available GPUs*/ train.py \
  --train_data /*Address to the training data for answer generation task*/ \
  --val_data /*Address to the validation data for answer generation task*/ \
  --eval_freq /*The frequency of Evaluations should be given as the number of steps*/ \
  --name /*Output dir name*/ \
  ---per_gpu_batch_size /*Per GPU batch size for training*/ \
  --total_steps /*Total training steps*/ \
  --accumulation_steps /*Number of gradient accumulation steps*/ \
  --warmup_steps /*Number of warmup steps*/ \
  --scheduler /*Learning rate scheduler type*/ \
  --optim /*Optimizer type*/ \
  --lr /*Learning rate*/ \
  --max_length_input /*Maximum length for the input of the model*/ \
  --max_length_output /*Maximum length for the output of the model*/ \
  --weight_decay/*weight decay*/ \
  --image_feats_addr /*Address to the image features file*/ \
  --image_feats_mapping_addr /*Address to the index to image features file*/ \
  --do_validation \
  --model_path /*The path to the checkpoint of MM-FiD you want to evaluate*/ \
  --n_context /*Number of documents to be used in generating answer*/ \
  --num_beams /*Number of beams in beam search*/ \
  /**IMPORTANT: The following parameters are only used for OKVQA when you want to use GPT3**/
  --gpt3_answers /*Address to the answers generated by an LLM*/ \
  /**IMPORTANT: The following parameters are only used for OKVQA**/
  --okvqa_acc /*Enables using OK-VQA original evaluation metric*/ \
  --ann_file /*Address to Annotation file in OK-VQA dataset for dynamic eval*/ \
  --ques_file /*Address to Question file in OK-VQA dataset for dynamic eval*/ \
```
