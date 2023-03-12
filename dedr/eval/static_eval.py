from collections import defaultdict

class StaticEval(object):

    def __init__(self, qrel_addr, sep = "###@###@###") -> None:
        self.qrels = defaultdict(dict)
        with open(qrel_addr) as file:
            for line in file:
                if line.strip():
                    line = line.strip().split(sep)
                    question_id = line[0]
                    passage_id = line[2]
                    if question_id not in self.qrels:
                       self.qrels[str(question_id)] = {'placeholder': 0}
                    self.qrels[str(question_id)][passage_id] = int(line[3])
    
    def gen_qrels(self, *args, **kwd):
        return self.qrels
