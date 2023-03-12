from collections import defaultdict
import linecache
import re
import numpy as np
from io import open
import json
import datetime
import copy
import os
import skimage.io as io
from torch.utils.data import Dataset

class VQA:
    def __init__(self, annotation_file=None, question_file=None):
        """
        Constructor of VQA helper class for reading and visualizing questions and answers.
        :param annotation_file (str): location of VQA annotation file
        :return:
        """
        # load dataset
        self.dataset = {}
        self.questions = {}
        self.qa = {}
        self.qqa = {}
        self.imgToQA = {}
        if not annotation_file == None and not question_file == None:
            print('loading VQA annotations and questions into memory...')
            time_t = datetime.datetime.utcnow()
            dataset = json.load(open(annotation_file, 'r'))
            questions = json.load(open(question_file, 'r'))
            print('time elpased: ', datetime.datetime.utcnow() - time_t)
            self.dataset = dataset
            self.questions = questions
            self.createIndex()

    def createIndex(self):
        # create index
        print('creating index...')
        imgToQA = {ann['image_id']: [] for ann in self.dataset['annotations']}
        qa =  {ann['question_id']: [] for ann in self.dataset['annotations']}
        qqa = {ann['question_id']: [] for ann in self.dataset['annotations']}
        for ann in self.dataset['annotations']:
            imgToQA[ann['image_id']] += [ann]
            qa[ann['question_id']] = ann
        for ques in self.questions['questions']:
              qqa[ques['question_id']] = ques
        print('index created!')

        # create class members
        self.qa = qa
        self.qqa = qqa
        self.imgToQA = imgToQA

    def info(self):
        """
        Print information about the VQA annotation file.
        :return:
        """
        for key, value in self.dataset['info'].items():
            print('%s: %s'%(key, value))

    def getQuesIds(self, imgIds=[], quesTypes=[], ansTypes=[]):
        """
        Get question ids that satisfy given filter conditions. default skips that filter
        :param     imgIds    (int array)   : get question ids for given imgs
                quesTypes (str array)   : get question ids for given question types
                ansTypes  (str array)   : get question ids for given answer types
        :return:    ids   (int array)   : integer array of question ids
        """
        imgIds = imgIds if type(imgIds) == list else [imgIds]
        quesTypes = quesTypes if type(quesTypes) == list else [quesTypes]
        ansTypes = ansTypes if type(ansTypes) == list else [ansTypes]

        if len(imgIds) == len(quesTypes) == len(ansTypes) == 0:
            anns = self.dataset['annotations']
        else:
            if not len(imgIds) == 0:
                anns = sum([self.imgToQA[imgId] for imgId in imgIds if imgId in self.imgToQA], [])
            else:
                 anns = self.dataset['annotations']
            anns = anns if len(quesTypes) == 0 else [ann for ann in anns if ann['question_type'] in quesTypes]
            anns = anns if len(ansTypes) == 0 else [ann for ann in anns if ann['answer_type'] in ansTypes]
        ids = [ann['question_id'] for ann in anns]
        return ids

    def getImgIds(self, quesIds=[], quesTypes=[], ansTypes=[]):
        """
        Get image ids that satisfy given filter conditions. default skips that filter
        :param quesIds   (int array)   : get image ids for given question ids
               quesTypes (str array)   : get image ids for given question types
               ansTypes  (str array)   : get image ids for given answer types
        :return: ids     (int array)   : integer array of image ids
        """
        quesIds   = quesIds if type(quesIds) == list else [quesIds]
        quesTypes = quesTypes if type(quesTypes) == list else [quesTypes]
        ansTypes  = ansTypes if type(ansTypes) == list else [ansTypes]

        if len(quesIds) == len(quesTypes) == len(ansTypes) == 0:
            anns = self.dataset['annotations']
        else:
            if not len(quesIds) == 0:
                anns = sum([self.qa[quesId] for quesId in quesIds if quesId in self.qa], [])
            else:
                anns = self.dataset['annotations']
            anns = anns if len(quesTypes) == 0 else [ann for ann in anns if ann['question_type'] in quesTypes]
            anns = anns if len(ansTypes) == 0 else [ann for ann in anns if ann['answer_type'] in ansTypes]
        ids = [ann['image_id'] for ann in anns]
        return ids

    def loadQA(self, ids=[]):
        """
        Load questions and answers with the specified question ids.
        :param ids (int array)       : integer ids specifying question ids
        :return: qa (object array)   : loaded qa objects
        """
        if type(ids) == list:
            return [self.qa[id] for id in ids]
        elif type(ids) == int:
            return [self.qa[ids]]

    def showQA(self, anns):
        """
        Display the specified annotations.
        :param anns (array of object): annotations to display
        :return: None
        """
        if len(anns) == 0:
            return 0
        for ann in anns:
            quesId = ann['question_id']
            print("Question: %s" %(self.qqa[quesId]['question']))
            for ans in ann['answers']:
                print("Answer %d: %s" %(ans['answer_id'], ans['answer']))
                
    def returnQA(self, anns):
        """
        Display the specified annotations.
        :param anns (array of object): annotations to display
        :return: None
        """
        if len(anns) == 0:
            return 0
        res = []
        for ann in anns:
            quesId = ann['question_id']
            image_id = ann['image_id']
            qa = {'image_id': image_id, 
                  'question_id': quesId, 
                  'question': self.qqa[quesId]['question'], 
                  'answers': {}}
            for ans in ann['answers']:
                qa['answers'][ans['answer_id']] = ans['answer']
            res.append(qa)
        
        return res
        
    def loadRes(self, resFile, quesFile):
        """
        Load result file and return a result object.
        :param   resFile (str)     : file name of result file
        :return: res (obj)         : result api object
        """
        res = VQA()
        res.questions = json.load(open(quesFile))
        res.dataset['info'] = copy.deepcopy(self.questions['info'])
        res.dataset['task_type'] = copy.deepcopy(self.questions['task_type'])
        res.dataset['data_type'] = copy.deepcopy(self.questions['data_type'])
        res.dataset['data_subtype'] = copy.deepcopy(self.questions['data_subtype'])
        res.dataset['license'] = copy.deepcopy(self.questions['license'])

        print('Loading and preparing results...     ')
        time_t = datetime.datetime.utcnow()
        anns    = json.load(open(resFile))
        assert type(anns) == list, 'results is not an array of objects'
        annsQuesIds = [ann['question_id'] for ann in anns]
        assert set(annsQuesIds) == set(self.getQuesIds()), \
        'Results do not correspond to current VQA set. Either the results do not have predictions for all question ids in annotation file or there is atleast one question id that does not belong to the question ids in the annotation file.'
        for ann in anns:
            quesId                  = ann['question_id']
            if res.dataset['task_type'] == 'Multiple Choice':
                assert ann['answer'] in self.qqa[quesId]['multiple_choices'], 'predicted answer is not one of the multiple choices'
            qaAnn                = self.qa[quesId]
            ann['image_id']      = qaAnn['image_id'] 
            ann['question_type'] = qaAnn['question_type']
            ann['answer_type']   = qaAnn['answer_type']
        print('DONE (t=%0.2fs)'%((datetime.datetime.utcnow() - time_t).total_seconds()))

        res.dataset['annotations'] = anns
        res.createIndex()
        return res

class DynamicEval():
    def __init__(self, ann_file, ques_file, passage_id_to_line_id_file, all_blocks_file):
        
        with open(passage_id_to_line_id_file) as fin:
            self.passage_id_to_line_id = json.load(fin)
            
        self.vqa = VQA(ann_file, ques_file)
        self.all_blocks_file = all_blocks_file
            
    
    def get_answers(self, question_id):
        ann = self.vqa.loadQA(question_id)
        qa = self.vqa.returnQA(ann)[0]
        answers = set(answer.lower() for answer in qa['answers'].values() if answer)
        return answers
    
    
    def get_passage(self, passage_id):
        passage_line = linecache.getline(
            self.all_blocks_file, self.passage_id_to_line_id[passage_id])
        passage_dict = json.loads(passage_line)
        passage = passage_dict['text']
        assert passage_id == passage_dict['id']

        return passage
    
    
    def has_answers(self, answers, passage):
        passage = passage.lower()
        for answer in answers:
            # "\b" matches word boundaries.
            # answer_starts = [match.start() for match in re.finditer(
            #     r'\b{}\b'.format(answer.lower()), passage)]
            if re.search(r'\b{}\b'.format(answer), passage):
                return True
        return False
    
    
    def gen_qrels(self, question_ids, I, retrieved_id_to_passage_id):
       
        qrels = defaultdict(dict)
        for question_id, retrieved_ids in zip(question_ids, I):
            if question_id not in qrels:
                qrels[str(question_id)] = {'placeholder': 0}

            for retrieved_id in retrieved_ids:
                passage_id = retrieved_id_to_passage_id[retrieved_id]
                answers = self.get_answers(int(question_id))
                passage = self.get_passage(passage_id)

                if self.has_answers(answers, passage):
                    qrels[str(question_id)][passage_id] = 1

        return qrels
