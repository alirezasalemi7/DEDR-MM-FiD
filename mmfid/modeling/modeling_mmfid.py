from vlt5.modeling_t5 import VLT5
from transformers.models.t5.configuration_t5 import T5Config
import torch

class MMFiDConfig(T5Config):

    def __init__(self, vocab_size=32128, d_model=512, d_kv=64, d_ff=2048, num_layers=6, num_decoder_layers=None, num_heads=8, relative_attention_num_buckets=32, dropout_rate=0.1, layer_norm_epsilon=0.000001, initializer_factor=1, feed_forward_proj="relu", is_encoder_decoder=True, use_cache=True, pad_token_id=0, eos_token_id=1, **kwargs):
        super().__init__(vocab_size, d_model, d_kv, d_ff, num_layers, num_decoder_layers, num_heads, relative_attention_num_buckets, dropout_rate, layer_norm_epsilon, initializer_factor, feed_forward_proj, is_encoder_decoder, use_cache, pad_token_id, eos_token_id, **kwargs)
        self.feat_dim = 2048
        self.pos_dim = 4
        self.n_images = 2
        self.use_vis_order_embedding = True
        self.vocab_size = 32200
        self.use_vis_layer_norm = True
        self.individual_vis_layer_norm = True
        self.share_vis_lang_layer_norm = False

class MMFiD(VLT5):

    def __init__(self, config):

        super().__init__(config)
        self.wrap_encoder()

    def forward(self, input_ids=None, attention_mask=None, vis_attention_mask = None, **kwargs):
        if input_ids != None:
            # inputs might have already be resized in the generate method
            if input_ids.dim() == 3:
                self.encoder.n_passages = input_ids.size(1)
            input_ids = input_ids.view(input_ids.size(0), -1)
        if attention_mask != None:
            attention_mask = attention_mask.view(attention_mask.size(0), -1)
        
        return super().forward(
            input_ids=input_ids,
            attention_mask=attention_mask,
            **kwargs
        )
    
    def generate(self, input_ids, attention_mask, **kwds):
        self.encoder.n_passages = input_ids.size(1)
        return super().generate(
            input_ids=input_ids.view(input_ids.size(0), -1),
            attention_mask=attention_mask.view(attention_mask.size(0), -1),
            **kwds
        )
    
    def wrap_encoder(self):
        self.encoder = EncoderWrapper(self.encoder)
    
    def unwrap_encoder(self):
        """
        Unwrap Fusion-in-Decoder encoder, useful to load VLT5 weights.
        """
        self.encoder = self.encoder.encoder
        block = []
        for mod in self.encoder.block:
            block.append(mod)
        block = torch.nn.ModuleList(block)
        self.encoder.block = block
    
    def load_VLT5(self, state_dict):
        self.unwrap_encoder()
        
        original_keys = list(state_dict.keys())
        for key in original_keys:
            if key.startswith("module.decoder."):
                new_key = 'decoder.' + key[len("module.decoder."):]
                state_dict[new_key] = state_dict.pop(key)

            elif key.startswith("module.vis_encoder."):
                new_key = 'encoder.' + key[len("module.vis_encoder."):]
                state_dict[new_key] = state_dict.pop(key)
            
            elif key.startswith("module.encoder."):
                new_key = 'encoder.' + key[len("module.encoder."):]
                state_dict[new_key] = state_dict.pop(key)

            elif key.startswith("module."):
                new_key = key[len("module."):]
                state_dict[new_key] = state_dict.pop(key)

        self.load_state_dict(state_dict, strict = False)
        
        self.wrap_encoder()
    

class EncoderWrapper(torch.nn.Module):
    def __init__(self, encoder, use_checkpoint=False):
        super().__init__()

        self.encoder = encoder

    def forward(self, input_ids=None, attention_mask=None, vis_inputs = None, vis_attention_mask = None, **kwargs,):
        # total_length = n_passages * passage_length
        bsz, total_length = input_ids.shape
        passage_length = total_length // self.n_passages
        input_ids = input_ids.view(bsz*self.n_passages, passage_length)
        attention_mask = attention_mask.view(bsz*self.n_passages, passage_length)
        vis_inputs = (vis_inputs[0].repeat_interleave(self.n_passages, dim = 0), vis_inputs[1].repeat_interleave(self.n_passages, dim = 0))
        if vis_attention_mask is not None:
            vis_attention_mask = vis_attention_mask.repeat_interleave(self.n_passages, dim = 0)
        outputs = self.encoder(input_ids = input_ids, attention_mask = attention_mask, vis_inputs = vis_inputs, vis_attention_mask = vis_attention_mask, **kwargs)
        outputs.last_hidden_state = outputs.last_hidden_state.view(bsz, self.n_passages * (passage_length + 36), -1)
        return outputs