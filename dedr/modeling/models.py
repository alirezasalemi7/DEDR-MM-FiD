from transformers.models.lxmert import LxmertConfig, LxmertPreTrainedModel, LxmertModel
from transformers.models.bert import BertConfig, BertPreTrainedModel, BertModel
from transformers import PreTrainedModel
from transformers.configuration_utils import PretrainedConfig
import torch
from torch import nn

class E_MM(LxmertPreTrainedModel):
    def __init__(self, config: LxmertConfig, expand_docs = True, *inputs, **kwargs):
        super().__init__(config, *inputs, **kwargs)
        self.lxmert = LxmertModel.from_pretrained("unc-nlp/lxmert-base-uncased")
        self.lxmert.config.expand_docs = expand_docs
        assert expand_docs == True, "E_MM only works with document expansion"
    
    def forward(self, input_ids = None, token_type_ids = None, attention_mask = None, visual_feats = None, visual_pos = None):
        
        if visual_feats is None:
            # visual_feats = 
            output = self.lxmert(
                input_ids = input_ids,
                token_type_ids = token_type_ids,
                attention_mask = attention_mask,
                visual_pos = torch.arange(0,2).float().repeat_interleave(2).repeat(input_ids.shape[0], 36, 1).to(input_ids.device),
                visual_feats = torch.zeros((input_ids.shape[0], 36, 2048)).float().to(input_ids.device),
                output_attentions=False
            )
            return output.pooled_output
        else:
            output = self.lxmert(
                input_ids = input_ids,
                token_type_ids = token_type_ids,
                attention_mask = attention_mask,
                visual_pos = visual_pos,
                visual_feats = visual_feats,
                output_attentions=False
            )
            return output.pooled_output

class E_T(BertPreTrainedModel):

    def __init__(self, config: BertConfig, *inputs, **kwargs):
        super().__init__(config, *inputs, **kwargs)
        self.bert = BertModel.from_pretrained("bert-base-uncased", config = config)

    def forward(self, input_ids, token_type_ids, attention_mask):

        output = self.bert(
            input_ids = input_ids,
            token_type_ids = token_type_ids,
            attention_mask = attention_mask,
        )

        return output.pooler_output

class DEDRJointTraining(PreTrainedModel):

    def __init__(self, config, expand_docs = True, *inputs, **kwargs):
        super().__init__(config, *inputs, **kwargs)
        self.et = E_T(BertConfig.from_dict(config.bert), *inputs, **kwargs)
        self.emm = E_MM(LxmertConfig.from_dict(config.lxmert), expand_docs, *inputs, **kwargs)
    
    def forward(self, input_ids_bert, token_type_ids_bert, attention_mask_bert, input_ids_lxmert, token_type_ids_lxmert, attention_mask_lxmert, visual_feats, visual_pos):
        
        reps_bert = self.et(
            input_ids = input_ids_bert,
            token_type_ids = token_type_ids_bert,
            attention_mask = attention_mask_bert
        )

        reps_lxmert = self.emm(
            input_ids = input_ids_lxmert,
            token_type_ids = token_type_ids_lxmert,
            attention_mask = attention_mask_lxmert,
            visual_feats = visual_feats,
            visual_pos = visual_pos
        )

        return torch.cat([reps_bert, reps_lxmert], dim = 1)

class DEDRConfig:

    def __init__(self, h) -> None:
        self.hidden_size = h

class DEDR(nn.Module):

    def __init__(self, et, emm):
        super().__init__()
        self.et = et
        self.emm = emm
        self.config = DEDRConfig(emm.config.hidden_size + et.config.hidden_size)
        
    
    def forward(self, input_ids_bert, token_type_ids_bert, attention_mask_bert, input_ids_lxmert, token_type_ids_lxmert, attention_mask_lxmert, visual_feats, visual_pos):
        
        reps_bert = self.et(
            input_ids = input_ids_bert,
            token_type_ids = token_type_ids_bert,
            attention_mask = attention_mask_bert
        )

        reps_lxmert = self.emm(
            input_ids = input_ids_lxmert,
            token_type_ids = token_type_ids_lxmert,
            attention_mask = attention_mask_lxmert,
            visual_feats = visual_feats,
            visual_pos = visual_pos
        )

        return torch.cat([reps_bert, reps_lxmert], dim = 1)

class RankerToRankerDistilationPipeline(nn.Module):

    def __init__(self, student, student_type, teacher, teacher_type, neg_type) -> None:
        super().__init__()

        self.model = student
        self.teacher = teacher
        self.student_type = student_type
        self.teacher_type = teacher_type
        self.neg_type = neg_type
    
    def forward(self,
        query_input_ids_bert = None,
        query_token_type_ids_bert = None,
        query_attention_mask_bert = None,
        query_input_ids_lxmert = None,
        query_token_type_ids_lxmert = None,
        query_attention_mask_lxmert = None,
        pos_input_ids_bert = None,
        pos_token_type_ids_bert = None,
        pos_attention_mask_bert = None,
        pos_input_ids_lxmert = None,
        pos_token_type_ids_lxmert = None,
        pos_attention_mask_lxmert = None,
        neg_input_ids_bert = None,
        neg_token_type_ids_bert = None,
        neg_attention_mask_bert = None,
        neg_input_ids_lxmert = None,
        neg_token_type_ids_lxmert = None,
        neg_attention_mask_lxmert = None,
        query_ids = None,
        visual_feats = None,
        visual_pos = None
    ):

        with torch.no_grad():
            self.teacher.eval()
            if self.teacher_type == "E_MM":
                query_reps = self.teacher(
                    input_ids = query_input_ids_lxmert,
                    token_type_ids = query_token_type_ids_lxmert,
                    attention_mask = query_attention_mask_lxmert,
                    visual_feats = visual_feats,
                    visual_pos = visual_pos
                )
                pos_passage_reps = self.teacher(
                    input_ids = pos_input_ids_lxmert,
                    token_type_ids = pos_token_type_ids_lxmert,
                    attention_mask = pos_attention_mask_lxmert,
                    visual_feats = None,
                    visual_pos = None
                )
                neg_passage_reps = self.teacher(
                    input_ids = neg_input_ids_lxmert,
                    token_type_ids = neg_token_type_ids_lxmert,
                    attention_mask = neg_attention_mask_lxmert,
                    visual_feats = None,
                    visual_pos = None
                )
            elif self.teacher_type == "E_T":
                query_reps = self.teacher(
                    input_ids = query_input_ids_bert,
                    token_type_ids = query_token_type_ids_bert,
                    attention_mask = query_attention_mask_bert,
                )
                pos_passage_reps = self.teacher(
                    input_ids = pos_input_ids_bert,
                    token_type_ids = pos_token_type_ids_bert,
                    attention_mask = pos_attention_mask_bert,
                )
                neg_passage_reps = self.teacher(
                    input_ids = neg_input_ids_bert,
                    token_type_ids = neg_token_type_ids_bert,
                    attention_mask = neg_attention_mask_bert,
                )
            elif self.teacher_type == "DEDR":
                query_reps = self.teacher(
                    input_ids_bert = query_input_ids_bert,
                    token_type_ids_bert = query_token_type_ids_bert,
                    attention_mask_bert = query_attention_mask_bert,
                    input_ids_lxmert = query_input_ids_lxmert,
                    token_type_ids_lxmert = query_token_type_ids_lxmert,
                    attention_mask_lxmert = query_attention_mask_lxmert,
                    visual_feats = visual_feats,
                    visual_pos = visual_pos
                )
                pos_passage_reps = self.teacher(
                    input_ids_bert = pos_input_ids_bert,
                    token_type_ids_bert = pos_token_type_ids_bert,
                    attention_mask_bert = pos_attention_mask_bert,
                    input_ids_lxmert = pos_input_ids_lxmert,
                    token_type_ids_lxmert = pos_token_type_ids_lxmert,
                    attention_mask_lxmert = pos_attention_mask_lxmert,
                    visual_feats = None,
                    visual_pos = None
                )
                neg_passage_reps = self.teacher(
                    input_ids_bert = neg_input_ids_bert,
                    token_type_ids_bert = neg_token_type_ids_bert,
                    attention_mask_bert = neg_attention_mask_bert,
                    input_ids_lxmert = neg_input_ids_lxmert,
                    token_type_ids_lxmert = neg_token_type_ids_lxmert,
                    attention_mask_lxmert = neg_attention_mask_lxmert,
                    visual_feats = None,
                    visual_pos = None
                )
            if self.neg_type == 'neg':
                # Shape: (batch_size, ).
                pos_logits = torch.sum(query_reps * pos_passage_reps, dim=-1)
                neg_logits = torch.sum(query_reps * neg_passage_reps, dim=-1)
                labels = torch.stack([pos_logits, neg_logits], dim=1)                    
            
            elif self.neg_type == 'all_neg':
                pos_logits = torch.sum(query_reps * pos_passage_reps, dim=-1)
                neg_logits = torch.matmul(query_reps, neg_passage_reps.transpose(0, 1))  # Shape: (batch_size, batch_size).
                labels = torch.cat([pos_logits.unsqueeze(dim=-1), neg_logits], dim=1)  # Shape: (batch_size, batch_size + 1).                    
                
            elif self.neg_type == 'other_pos+neg':
                pos_logits = torch.matmul(query_reps, pos_passage_reps.transpose(0, 1))  # Shape: (batch_size, batch_size).
                neg_logits = torch.sum(query_reps * neg_passage_reps, dim=-1)
                labels = torch.cat([pos_logits, neg_logits.unsqueeze(dim=-1)], dim=1)  # Shape: (batch_size, batch_size + 1).                  
                
            elif self.neg_type == 'other_pos+all_neg':
                pos_logits = torch.matmul(query_reps, pos_passage_reps.transpose(0, 1))  # Shape: (batch_size, batch_size).
                neg_logits = torch.matmul(query_reps, neg_passage_reps.transpose(0, 1))  # Shape: (batch_size, batch_size).
                labels = torch.cat([pos_logits, neg_logits], dim=1)  # Shape: (batch_size, 2 * batch_size).
            else:
                raise ValueError(f'`neg_type` should be one of `neg`, `all_neg`, `other_pos+neg`, or `other_pos+all_neg`, not {self.neg_type}.')
            

        if self.student_type == "E_MM":
            query_reps = self.model(
                input_ids = query_input_ids_lxmert,
                token_type_ids = query_token_type_ids_lxmert,
                attention_mask = query_attention_mask_lxmert,
                visual_feats = visual_feats,
                visual_pos = visual_pos
            )
            pos_passage_reps = self.model(
                input_ids = pos_input_ids_lxmert,
                token_type_ids = pos_token_type_ids_lxmert,
                attention_mask = pos_attention_mask_lxmert,
                visual_feats = None,
                visual_pos = None
            )
            neg_passage_reps = self.model(
                input_ids = neg_input_ids_lxmert,
                token_type_ids = neg_token_type_ids_lxmert,
                attention_mask = neg_attention_mask_lxmert,
                visual_feats = None,
                visual_pos = None
            )
        elif self.student_type == "E_T":
            query_reps = self.model(
                input_ids = query_input_ids_bert,
                token_type_ids = query_token_type_ids_bert,
                attention_mask = query_attention_mask_bert,
            )
            pos_passage_reps = self.model(
                input_ids = pos_input_ids_bert,
                token_type_ids = pos_token_type_ids_bert,
                attention_mask = pos_attention_mask_bert,
            )
            neg_passage_reps = self.model(
                input_ids = neg_input_ids_bert,
                token_type_ids = neg_token_type_ids_bert,
                attention_mask = neg_attention_mask_bert,
            )
        elif self.student_type == "DEDR":
            
            query_reps = self.model(
                input_ids_bert = query_input_ids_bert,
                token_type_ids_bert = query_token_type_ids_bert,
                attention_mask_bert = query_attention_mask_bert,
                input_ids_lxmert = query_input_ids_lxmert,
                token_type_ids_lxmert = query_token_type_ids_lxmert,
                attention_mask_lxmert = query_attention_mask_lxmert,
                visual_feats = visual_feats,
                visual_pos = visual_pos
            )

            pos_passage_reps = self.model(
                input_ids_bert = pos_input_ids_bert,
                token_type_ids_bert = pos_token_type_ids_bert,
                attention_mask_bert = pos_attention_mask_bert,
                input_ids_lxmert = pos_input_ids_lxmert,
                token_type_ids_lxmert = pos_token_type_ids_lxmert,
                attention_mask_lxmert = pos_attention_mask_lxmert,
                visual_feats = None,
                visual_pos = None
            )

            neg_passage_reps = self.model(
                input_ids_bert = neg_input_ids_bert,
                token_type_ids_bert = neg_token_type_ids_bert,
                attention_mask_bert = neg_attention_mask_bert,
                input_ids_lxmert = neg_input_ids_lxmert,
                token_type_ids_lxmert = neg_token_type_ids_lxmert,
                attention_mask_lxmert = neg_attention_mask_lxmert,
                visual_feats = None,
                visual_pos = None
            )

        if self.neg_type == 'neg':
            # Shape: (batch_size, ).
            pos_logits = torch.sum(query_reps * pos_passage_reps, dim=-1)
            neg_logits = torch.sum(query_reps * neg_passage_reps, dim=-1)
            logits = torch.stack([pos_logits, neg_logits], dim=1)                    
        elif self.neg_type == 'all_neg':
            pos_logits = torch.sum(query_reps * pos_passage_reps, dim=-1)
            neg_logits = torch.matmul(query_reps, neg_passage_reps.transpose(0, 1))  # Shape: (batch_size, batch_size).
            logits = torch.cat([pos_logits.unsqueeze(dim=-1), neg_logits], dim=1)  # Shape: (batch_size, batch_size + 1).                    
        elif self.neg_type == 'other_pos+neg':
            pos_logits = torch.matmul(query_reps, pos_passage_reps.transpose(0, 1))  # Shape: (batch_size, batch_size).
            neg_logits = torch.sum(query_reps * neg_passage_reps, dim=-1)
            logits = torch.cat([pos_logits, neg_logits.unsqueeze(dim=-1)], dim=1)  # Shape: (batch_size, batch_size + 1).                  
        elif self.neg_type == 'other_pos+all_neg':
            pos_logits = torch.matmul(query_reps, pos_passage_reps.transpose(0, 1))  # Shape: (batch_size, batch_size).
            neg_logits = torch.matmul(query_reps, neg_passage_reps.transpose(0, 1))  # Shape: (batch_size, batch_size).
            logits = torch.cat([pos_logits, neg_logits], dim=1)  # Shape: (batch_size, 2 * batch_size).
        else:
            raise ValueError(f'`neg_type` should be one of `neg`, `all_neg`, `other_pos+neg`, or `other_pos+all_neg`, not {self.neg_type}.')
        


        loss_fn = nn.KLDivLoss()
        loss = loss_fn(torch.log_softmax(logits, dim=-1), torch.softmax(labels, dim=-1))
        preds = torch.argmax(logits, dim = -1)

        return (loss, preds, labels)

class RankerTrainingPipeline(nn.Module):

    def __init__(self, config, neg_type, model_type, model = None) -> None:
        super().__init__()

        self.neg_type = neg_type
        self.model_type = model_type
        if model != None:
            self.model = model
        elif model_type == "E_MM":
            self.model = E_MM(config = config[0], expand_docs = True)
        elif model_type == "E_T":
            self.model = E_T(config = config[0])
        elif model_type == "DEDR_joint":
            config_new = PretrainedConfig()
            config_new.bert = config[0].to_dict()
            config_new.lxmert = config[1].to_dict()
            config_new.hidden_size = config[0].hidden_size + config[1].hidden_size
            self.model = DEDRJointTraining(config = config_new, expand_docs = True)
        
    def forward(self,
        query_input_ids_bert = None,
        query_token_type_ids_bert = None,
        query_attention_mask_bert = None,
        query_input_ids_lxmert = None,
        query_token_type_ids_lxmert = None,
        query_attention_mask_lxmert = None,
        pos_input_ids_bert = None,
        pos_token_type_ids_bert = None,
        pos_attention_mask_bert = None,
        pos_input_ids_lxmert = None,
        pos_token_type_ids_lxmert = None,
        pos_attention_mask_lxmert = None,
        neg_input_ids_bert = None,
        neg_token_type_ids_bert = None,
        neg_attention_mask_bert = None,
        neg_input_ids_lxmert = None,
        neg_token_type_ids_lxmert = None,
        neg_attention_mask_lxmert = None,
        query_ids = None,
        visual_feats = None,
        visual_pos = None
    ):
        if self.model_type == "E_MM":
            query_reps = self.model(
                input_ids = query_input_ids_lxmert,
                token_type_ids = query_token_type_ids_lxmert,
                attention_mask = query_attention_mask_lxmert,
                visual_feats = visual_feats,
                visual_pos = visual_pos
            )
            pos_passage_reps = self.model(
                input_ids = pos_input_ids_lxmert,
                token_type_ids = pos_token_type_ids_lxmert,
                attention_mask = pos_attention_mask_lxmert,
                visual_feats = None,
                visual_pos = None
            )
            neg_passage_reps = self.model(
                input_ids = neg_input_ids_lxmert,
                token_type_ids = neg_token_type_ids_lxmert,
                attention_mask = neg_attention_mask_lxmert,
                visual_feats = None,
                visual_pos = None
            )
        elif self.model_type == "E_T":
            query_reps = self.model(
                input_ids = query_input_ids_bert,
                token_type_ids = query_token_type_ids_bert,
                attention_mask = query_attention_mask_bert,
            )
            pos_passage_reps = self.model(
                input_ids = pos_input_ids_bert,
                token_type_ids = pos_token_type_ids_bert,
                attention_mask = pos_attention_mask_bert,
            )
            neg_passage_reps = self.model(
                input_ids = neg_input_ids_bert,
                token_type_ids = neg_token_type_ids_bert,
                attention_mask = neg_attention_mask_bert,
            )
        elif self.model_type == "DEDR_joint":
            query_reps = self.model(
                input_ids_bert = query_input_ids_bert,
                token_type_ids_bert = query_token_type_ids_bert,
                attention_mask_bert = query_attention_mask_bert,
                input_ids_lxmert = query_input_ids_lxmert,
                token_type_ids_lxmert = query_token_type_ids_lxmert,
                attention_mask_lxmert = query_attention_mask_lxmert,
                visual_feats = visual_feats,
                visual_pos = visual_pos
            )

            pos_passage_reps = self.model(
                input_ids_bert = pos_input_ids_bert,
                token_type_ids_bert = pos_token_type_ids_bert,
                attention_mask_bert = pos_attention_mask_bert,
                input_ids_lxmert = pos_input_ids_lxmert,
                token_type_ids_lxmert = pos_token_type_ids_lxmert,
                attention_mask_lxmert = pos_attention_mask_lxmert,
                visual_feats = None,
                visual_pos = None
            )

            neg_passage_reps = self.model(
                input_ids_bert = neg_input_ids_bert,
                token_type_ids_bert = neg_token_type_ids_bert,
                attention_mask_bert = neg_attention_mask_bert,
                input_ids_lxmert = neg_input_ids_lxmert,
                token_type_ids_lxmert = neg_token_type_ids_lxmert,
                attention_mask_lxmert = neg_attention_mask_lxmert,
                visual_feats = None,
                visual_pos = None
            )

        if self.neg_type == 'neg':
            # Shape: (batch_size, ).
            pos_logits = torch.sum(query_reps * pos_passage_reps, dim=-1)
            neg_logits = torch.sum(query_reps * neg_passage_reps, dim=-1)
            logits = torch.stack([pos_logits, neg_logits], dim=1)                    
            labels = torch.zeros(logits.size(0), device=logits.device, dtype=torch.long)
        
        elif self.neg_type == 'all_neg':
            pos_logits = torch.sum(query_reps * pos_passage_reps, dim=-1)
            neg_logits = torch.matmul(query_reps, neg_passage_reps.transpose(0, 1))  # Shape: (batch_size, batch_size).
            logits = torch.cat([pos_logits.unsqueeze(dim=-1), neg_logits], dim=1)  # Shape: (batch_size, batch_size + 1).                    
            labels = torch.zeros(logits.size(0), device=logits.device, dtype=torch.long)
            
        elif self.neg_type == 'other_pos+neg':
            pos_logits = torch.matmul(query_reps, pos_passage_reps.transpose(0, 1))  # Shape: (batch_size, batch_size).
            neg_logits = torch.sum(query_reps * neg_passage_reps, dim=-1)
            logits = torch.cat([pos_logits, neg_logits.unsqueeze(dim=-1)], dim=1)  # Shape: (batch_size, batch_size + 1).                  
            labels = torch.arange(logits.size(0), device=logits.device, dtype=torch.long)
            
        elif self.neg_type == 'other_pos+all_neg':
            pos_logits = torch.matmul(query_reps, pos_passage_reps.transpose(0, 1))  # Shape: (batch_size, batch_size).
            neg_logits = torch.matmul(query_reps, neg_passage_reps.transpose(0, 1))  # Shape: (batch_size, batch_size).
            logits = torch.cat([pos_logits, neg_logits], dim=1)  # Shape: (batch_size, 2 * batch_size).
            labels = torch.arange(logits.size(0), device=logits.device, dtype=torch.long)
        else:
            raise ValueError(f'`neg_type` should be one of `neg`, `all_neg`, `other_pos+neg`, or `other_pos+all_neg`, not {self.neg_type}.')
        
        loss_fn = nn.CrossEntropyLoss()
        loss = loss_fn(logits, labels)
        preds = torch.argmax(logits, dim = -1)

        return (loss, preds, labels)