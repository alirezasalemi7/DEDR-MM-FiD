# Dual Encoding Dense Retrieval for Knowledge-Intensive VQA

<table>
<tr>
<td><img src="images/example.png?raw=True" width="400"/></td>
<td>An example KI-VQA question; answering it requires external knowledge.<br>
<sup><sub><a href="https://www.flickr.com/photos/zrimshots/2788695458">Image copyright zrim [https://www.flickr.com/photos/zrimshots/2788695458]</a></sub></sup></td>
</tr>
</table>

This is the replication package for our upcoming paper:

> Alireza Salemi, Juan Altmayer Pizzorno, and Hamed Zamani. A Symmetric Dual Encoding Dense Retrieval Framework for Knowledge-Intensive Visual Question Answering. In Proceedings of the 46th Int’l ACM SIGIR Conference on Research and Development in Information Retrieval (SIGIR ’23). Taipei, Taiwan, July 2023. [[BibTeX]](paper.bib)

[![License](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](LICENSE)



## File Tour

- `dedr`: Dense Passage Retriever
  - `data/`
  - `eval/`
  - `modeling/`
  - `utils/`
  - `distill_ranker_to_ranker.py`
  - `index.py`
  - `retrieve.py`
  - `train_ranker.py`

- `mmfid`: Answer Generator
  - `data/`
  - `eval/`
  - `modeling/`
  - `utils/`
  - `vlt5/`
  - `train.py`

## Additional Data
