import argparse
import os
import time

import glog
import torch
from transformers import AutoTokenizer

from lib import codebook, utils
from lib.utils.unsafe_import import model_from_hf_path
from model.llama import LlamaForCausalLM
from transformers import LlamaForCausalLM as OrigLlama
from model.qwen3_fp16 import Qwen3ForCausalLMFP16
from model.qwen3 import Qwen3ForCausalLM
torch.set_grad_enabled(False)

parser = argparse.ArgumentParser()
parser.add_argument('--quantized_path', type=str)
parser.add_argument('--hf_output_path', type=str)
parser.add_argument('--skip_list', default=None, type=str)


def main(args):
    assert os.path.exists(args.quantized_path)
    saved_config = torch.load(os.path.join(args.quantized_path, 'config.pt'), weights_only=False)
    model_config = saved_config['model_config']
    
    glog.info(model_config)
    fused = model_config.quip_params.get('fused', True)

    tokenizer = AutoTokenizer.from_pretrained(model_config._name_or_path)

    model = Qwen3ForCausalLM.from_pretrained(model_config._name_or_path,
                                             torch_dtype='auto',
                                             low_cpu_mem_usage=True,
                                             config=model_config)

    orig_model = Qwen3ForCausalLMFP16.from_pretrained(model_config._name_or_path,
                                           torch_dtype='auto',
                                           low_cpu_mem_usage=True,
                                           config=model_config)

    if model_config.quip_params['skip_list'] is None:
        model_config.quip_params['skip_list'] = []
    
    cpu = torch.device('cpu')
    if os.path.exists(f'{args.quantized_path}/lmhead.pt'):
        lmhead_data = torch.load(f'{args.quantized_path}/lmhead.pt',
                                 map_location=cpu)
        model.lm_head.weight.copy_(lmhead_data['lm_head'].to(
            model.lm_head.weight.dtype))
        model.model.norm.weight.copy_(lmhead_data['norm'].to(
            model.model.norm.weight.dtype))

    if args.skip_list is not None:
        args.skip_list = args.skip_list.split(',')
    else:
        args.skip_list = []

    skip_list_union = [*args.skip_list, *model_config.quip_params['skip_list']]
    model.config.quip_params['skip_list'] = skip_list_union

    for ii in range(len(model.model.layers)):
        layer = model.model.layers[ii]

        if os.path.exists(f'{args.quantized_path}/{ii}_layernorm.pt'):
            ln_data = torch.load(f'{args.quantized_path}/{ii}_layernorm.pt',
                                 map_location=cpu)
            layer.input_layernorm.weight.copy_(ln_data['input_layernorm'].to(
                layer.input_layernorm.weight.dtype))
            layer.post_attention_layernorm.weight.copy_(
                ln_data['post_attention_layernorm'].to(
                    layer.post_attention_layernorm.weight.dtype))

        if f'{ii}_q' not in skip_list_union:
            saved_layer = torch.load(f'{args.quantized_path}/{ii}_q.pt',
                                     map_location=cpu)
            utils.unpack_quip(layer.self_attn.q_proj, saved_layer)
        else:
            layer.self_attn.q_proj = orig_model.model.layers[ii].self_attn.q_proj
        
        if f'{ii}_k' not in skip_list_union:
            saved_layer = torch.load(f'{args.quantized_path}/{ii}_k.pt',
                                     map_location=cpu)
            utils.unpack_quip(layer.self_attn.k_proj, saved_layer)
        else:
            layer.self_attn.k_proj = orig_model.model.layers[ii].self_attn.k_proj
            
        if f'{ii}_v' not in skip_list_union:
            saved_layer = torch.load(f'{args.quantized_path}/{ii}_v.pt',
                                     map_location=cpu)
            utils.unpack_quip(layer.self_attn.v_proj, saved_layer)
        else:
            layer.self_attn.v_proj = orig_model.model.layers[ii].self_attn.v_proj

        if f'{ii}_o' not in skip_list_union:
            saved_layer = torch.load(f'{args.quantized_path}/{ii}_o.pt',
                                     map_location=cpu)
            utils.unpack_quip(layer.self_attn.o_proj, saved_layer)
        else:
            layer.self_attn.o_proj = orig_model.model.layers[ii].self_attn.o_proj

        if f'{ii}_up' not in skip_list_union:
            saved_layer = torch.load(f'{args.quantized_path}/{ii}_up.pt',
                                     map_location=cpu)
            utils.unpack_quip(layer.mlp.up_proj, saved_layer)
        else:
            layer.mlp.up_proj = orig_model.model.layers[ii].mlp.up_proj
            
        if f'{ii}_gate' not in skip_list_union:
            saved_layer = torch.load(f'{args.quantized_path}/{ii}_gate.pt',
                                     map_location=cpu)
            utils.unpack_quip(layer.mlp.gate_proj, saved_layer)
        else:
            layer.mlp.gate_proj = orig_model.model.layers[ii].mlp.gate_proj
                      
        if f'{ii}_down' not in skip_list_union:
            saved_layer = torch.load(f'{args.quantized_path}/{ii}_down.pt',
                                     map_location=cpu)
            utils.unpack_quip(layer.mlp.down_proj, saved_layer)
        else:
            layer.mlp.down_proj = orig_model.model.layers[ii].mlp.down_proj

        glog.info(f'loaded layer {ii}')
            
    glog.info(f'saving model...')
    model.save_pretrained(args.hf_output_path, safe_serialization=True)
    tokenizer.save_pretrained(args.hf_output_path)

    # save_pretrained serializes the config CLASS's model_type (the Qwen2Config
    # fallback writes 'qwen2') and can drop _name_or_path; both break the
    # model_from_hf_path reload. Patch the JSON on disk, which survives any
    # config-class games.
    import json as _json
    _cfg_path = os.path.join(args.hf_output_path, 'config.json')
    with open(_cfg_path) as _f:
        _cfg = _json.load(_f)
    _cfg['model_type'] = 'qwen3'
    _cfg['_name_or_path'] = model_config._name_or_path or 'Qwen/Qwen3-1.7B'
    with open(_cfg_path, 'w') as _f:
        _json.dump(_cfg, _f, indent=2)

    del model

    model, _ = model_from_hf_path(args.hf_output_path)

    glog.info('successfully loaded hfized model')


if __name__ == '__main__':
    torch.set_grad_enabled(False)
    torch.manual_seed(0)
    args = parser.parse_args()
    main(args)
