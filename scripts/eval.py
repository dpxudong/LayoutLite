from torch.nn.modules.module import Module
from torch.nn.modules.module import Module
from transformers import AutoConfig, AutoModelForCausalLM, \
                         Qwen3Config, Qwen3Model, Qwen2ForCausalLM, \
                         CLIPVisionModel, CLIPImageProcessor, AutoTokenizer, AutoProcessor
from transformers.modeling_outputs import BaseModelOutputWithPast, CausalLMOutputWithPast
from typing import Any, List, Optional, Tuple, Union
from transformers.cache_utils import Cache, DynamicCache
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import CrossEntropyLoss
from transformers.utils import is_torchdynamo_compiling
from transformers.processing_utils import Unpack
from transformers.utils.generic import check_model_inputs
from transformers.utils import auto_docstring, TransformersKwargs
from types import SimpleNamespace
from transformers.generation.configuration_utils import CompileConfig
from transformers.generation.utils import GenerateDecoderOnlyOutput

import os, re, argparse, random, json, time
from datetime import datetime
import Levenshtein
from tqdm import tqdm
from transformers.utils import logging
import cv2
import numpy as np
from fvcore.nn import FlopCountAnalysis
from layoutlite.modeling_layoutlite import MultiLayerVisionTokenScoreMLP, Qwen3VLTokenSelect, single_infer_qwen3vl
from layoutlite.utils.eval_utils import save_score_heatmap, save_score_mask, save_score_mask_with_text
from layoutlite.utils.utils import PROMPT_LOGICS, PROMPT_FIRERED
from layoutlite.utils.find_cluster_threshold import find_alpha, alpha_dict_firered, alpha_dict_logics, alpha_dict_firered_onlygrpo

logging.set_verbosity_error()
    
def eval(model=None, tokenizer=None, score_head=None, dataset=None, processor=None, compression_ratio=None, layoutlite_scores_dir=None):
    
    logics = True if model_type == 'logics' else False
    
    if logics:
        from layoutlite.utils.Logics_Parsing_v2_img2md import qwenvl_cast_html_tag
    
    time_str = datetime.now().strftime("%m%d%H%M%S")
    group = 'eval' + time_str + f'{random.randint(1,1000)}'
    os.environ['group'] = group
    output_dir = os.path.join(os.environ['output_dir'], 'eval_' + time_str)
    os.environ['output_dir'] = output_dir
    output_image_dir = os.path.join(output_dir, 'image')
    os.makedirs(output_dir)
    os.makedirs(os.path.join(output_dir, 'cache/scores'))
        
    log_file = os.path.join(output_dir, 'log_' + time_str + '.txt')
    with open(log_file, 'a') as f:
        f.write('mode: eval\n')

    with open(dataset, 'r', encoding='utf-8') as f:
        
        lines = list(f)
        
        if not layoutlite_scores_dir:
            os.environ['infer_mode'] = 'execute_layoutlite'
            for step, line in tqdm(
                enumerate((lines), start=1),
                total=len(lines),
                desc='Executing LayoutLite'
            ):
                data = json.loads(line)
                image_path = data["images"][0]
                os.environ['image_name'] = os.path.basename(image_path)
                
                try:
                    pred = single_infer_qwen3vl(
                        model=model,
                        tokenizer=tokenizer,
                        processor=processor,
                        image_path=image_path,
                        max_new_tokens=8192,
                        prompt=PROMPT_LOGICS if logics else PROMPT_FIRERED
                    )
                except:
                    pass
            
            alpha = find_alpha(os.path.join(output_dir, 'cache/scores'), compression_ratio)
        else:
            alpha = find_alpha(layoutlite_scores_dir, compression_ratio)
            
        os.environ['alpha'] = f'{alpha}'
        os.environ['infer_mode'] = 'full'
        
        for step, line in tqdm(
            enumerate((lines), start=1),
            total=len(lines)
        ):

            loss, rewards, log_prob_sums = 0,[],[]
            per_Leven, per_discard = [], []
            completions = []
            data = json.loads(line)

            gt = data["solution"]
            image_path = data["images"][0]
            
            if os.path.exists(output_image_dir[:-5]+'md/'+os.path.basename(image_path)[:-3]+'md'):
                continue
            
            

            os.environ['image_name'] = os.path.basename(image_path)
            try:
                pred = single_infer_qwen3vl(
                    model=model,
                    tokenizer=tokenizer,
                    processor=processor,
                    image_path=image_path,
                    max_new_tokens=8192,
                    prompt=PROMPT_LOGICS if logics else PROMPT_FIRERED
                )
            except Exception as e:
                if os.environ['infer_mode'] == 'vision_only':
                    continue
                else:
                    raise e
            if logics:
                pred = qwenvl_cast_html_tag(pred)
            completions.append(pred)
            group = os.environ['group']
            image_embeds = torch.load(os.path.join(output_dir, 'cache/firered_image_embed.pt'))
            
            flops = FlopCountAnalysis(score_head, image_embeds)
            with open(os.path.join(output_dir, 'efficiency.txt'), 'a') as f:
                f.write(f'layoutlite_flops: {flops.total() / 1e12:.6f} ')
                
            scores = score_head(image_embeds)
            # scores = torch.sigmoid(scores)
            scores = (scores - scores.mean()) / scores.std()
            
            Levenshtein_ratio = Levenshtein.ratio(gt, pred)

            mask = torch.load(os.path.join(output_dir, 'cache/firered_mask.pt'))
            mask = mask.float()
            
            with open(os.path.join(output_dir, 'efficiency.txt'), 'a') as f:
                f.write(f'seq_len: {mask.count_nonzero()}\n')
            
            try:
                save_score_heatmap(scores.detach().float(), image_path, output_image_dir)
                save_score_mask(mask.detach().cpu(), scores.detach().float(), image_path, output_image_dir+'mask')
            except:
                pass
            os.makedirs(output_image_dir[:-5]+'md', exist_ok=True)
            with open(output_image_dir[:-5]+'md/'+os.path.basename(image_path)[:-3]+'md','w') as md:
                md.write(pred)
            
            
            per_Leven.append(Levenshtein_ratio)
            if Levenshtein_ratio < 0.9:
                try:
                    save_score_mask_with_text(
                        mask.detach().cpu(),
                        scores,
                        image_path,
                        output_image_dir+'badcase',
                        Levenshtein_ratio,
                        gt,
                        pred,
                        threshold=0.1,
                    )
                except:
                    pass
            
            
            discard_ratio: torch.Tensor = 1 - mask.mean()
            per_discard.append(discard_ratio.item())
        
            info = f"Levenshtein_ratio={sum(per_Leven):.4f}, discard_ratio={sum(per_discard):.4f}"
            
            with open(log_file, 'a', encoding='utf-8') as l:
                l.write(info + f', completions: {completions}\n')
                
    
if __name__ == '__main__':
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_type", type=str, required=True)
    parser.add_argument("--model_dir", type=str, required=True)
    parser.add_argument("--layoutlite_ckp_path", type=str, required=True)
    parser.add_argument("--dataset", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--compression_ratio", type=float, required=True)
    parser.add_argument("--layoutlite_scores_dir", type=str)

    args = parser.parse_args()

    model_type = args.model_type
    model_dir = args.model_dir
    layoutlite_ckp_path = args.layoutlite_ckp_path
    dataset = args.dataset
    os.environ['output_dir'] = args.output_dir
    compression_ratio = args.compression_ratio
    layoutlite_scores_dir = args.layoutlite_scores_dir

    
    model = Qwen3VLTokenSelect.from_pretrained(model_dir, trust_remote_code=True)
    model = model.to(torch.bfloat16)
    
    tokenizer = AutoTokenizer.from_pretrained(
        model_dir,
        trust_remote_code=True
    )
    
    processor = AutoProcessor.from_pretrained(
        model_dir,
        trust_remote_code=True
    )
    
    if model_type == 'logics':
        score_head = MultiLayerVisionTokenScoreMLP(hidden_dim=2560, mid_dim=1280)
    else:
        score_head = MultiLayerVisionTokenScoreMLP()
        
    score_head.load_state_dict(torch.load(layoutlite_ckp_path))
    score_head = score_head.to(torch.bfloat16)
    model.model.score_mlp = score_head
    model = model.cuda()
    model.eval()
    model.model.score_mlp.eval()

    optimizer = torch.optim.Adam(score_head.parameters())
    
    eval(model=model, tokenizer=tokenizer, score_head=score_head, processor=processor, dataset=dataset, compression_ratio=compression_ratio, layoutlite_scores_dir=layoutlite_scores_dir)