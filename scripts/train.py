import os, re, json, time, cv2, random, argparse
import numpy as np
from torch.nn.modules.module import Module
from transformers import AutoTokenizer, AutoProcessor
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
import matplotlib.pyplot as plt
from datetime import datetime
import Levenshtein
from tqdm import tqdm
from transformers.utils import logging
from scipy import ndimage
from layoutlite.modeling_layoutlite import MultiLayerVisionTokenScoreMLP, Qwen3VLTokenSelect, single_infer_qwen3vl
from layoutlite.utils.utils import save_score_heatmap
from layoutlite.utils.utils import PROMPT_LOGICS, PROMPT_FIRERED

logging.set_verbosity_error()

os.environ['dtype'] = 'bfloat16'


def compute_reward(Levenshtein_ratio, discard_ratio, alpha=0.5):
    '''reward = -5|discard_ratio-0.5|**1.5 + Levenshtein'''
    
    discard_reward = - 5*abs(discard_ratio - 0.5)**1.5
    Levenshtein_reward = Levenshtein_ratio
    
    return Levenshtein_reward + discard_reward
    
def train(model=None, tokenizer=None, score_head=None, dataset=None, processor=None, model_type=None, batch_size=5, save_step=25):
    
    logics = True if model_type == 'logics' else False
    
    if logics:
        from layoutlite.utils.Logics_Parsing_v2_img2md import qwenvl_cast_html_tag
    
    time_str = datetime.now().strftime("%m%d%H%M%S")
    group = time_str + f'{random.randint(1,1000)}'
    os.environ['group'] = group
    output_dir = os.path.join(os.environ['output_dir'], 'train_' + time_str)
    os.environ['output_dir'] = output_dir
    output_image_dir = os.path.join(output_dir, 'image')
    os.makedirs(output_dir)
    os.makedirs(os.path.join(output_dir, 'cache'))
    
    log_file = os.path.join(output_dir, 'log_' + time_str + '.txt')
    
    with open(log_file, 'a') as f:
        f.write('mode: grpo ' + compute_reward.__doc__ + '\n')

    torch.save(score_head.state_dict(), os.path.join(output_dir, f'ckp_0.pt'))
    
    with open(dataset, 'r', encoding='utf-8') as f:
        
        lines = list(f)
        for step, line in tqdm(
            enumerate((lines[:]), start=1),
            total=len(lines)
        ):

            if step<=0:
                continue
            loss, rewards, log_prob_sums = 0,[],[]
            box_loss = 0
            per_Leven, per_discard = [], []
            completions = []
            data = json.loads(line)

            gt = data["solution"]
            prompt = data["messages"][0]["content"]
            image_path = data["images"][0]
            os.environ['image_name'] = os.path.basename(image_path)
            for idx in range(batch_size):
                pred = single_infer_qwen3vl(
                    model=model,
                    tokenizer=tokenizer,
                    processor=processor,
                    image_path=image_path,
                    max_new_tokens=8192,
                    prompt=PROMPT_LOGICS if logics else PROMPT_FIRERED
                )
                if logics:
                    pred = qwenvl_cast_html_tag(pred)
                completions.append(pred)
                group = os.environ['group']
                image_embeds = []

                image_embeds = torch.load(os.path.join(output_dir, f'cache/firered_image_embed.pt'))
                
                scores = score_head(image_embeds)
                scores = (scores - scores.mean()) / scores.std()
                scores = torch.sigmoid(scores)
                
                
                if idx == 0:
                    try:
                        save_score_heatmap(scores.detach().float(), image_path, output_image_dir, step)
                    except:
                        pass
                
                Levenshtein_ratio = Levenshtein.ratio(gt, pred)
                per_Leven.append(Levenshtein_ratio)
                
                mask = torch.load(os.path.join(output_dir, f'cache/firered_mask.pt'))
                
                if mask.isnan().any() or mask.isinf().any() or scores.isnan().any() or scores.isinf().any():
                    continue
                
                if 'layout_json_path' in os.environ:
                    box = torch.load(os.path.join(output_dir, f'cache/box.pt'))
                    box_loss += scores[~box].mean() - scores[box].mean()
                
                mask = mask.to(scores.device).float()
                discard_ratio = 1 - mask.mean()
                per_discard.append(discard_ratio.item())
                prob = scores.clamp(1e-6, 1 - 1e-6)
                dist = torch.distributions.Bernoulli(prob)

                log_prob = dist.log_prob(mask)  
                log_prob_sum = log_prob.mean()
                log_prob_sums.append(log_prob_sum)
                
                image_token_len = mask.sum()
                gt_len = len(tokenizer.encode(gt))
                text_density = gt_len / image_token_len
        
                
                reward = compute_reward(Levenshtein_ratio, discard_ratio, alpha=1)
                rewards.append(reward)
            
            baseline = torch.stack(rewards).mean()
            std = torch.stack(rewards).std() + 1e-6
            for j in range(len(rewards)):
                rewards[j] -= baseline
                rewards[j] /= std
            for reward, log_prob_sum in zip(rewards, log_prob_sums):
                loss += - reward * log_prob_sum / batch_size
            optimizer.zero_grad()
            if 'layout_json_path' in os.environ:
                loss += box_loss
            loss.backward()
            
            info = f"reward={baseline:.4f}, Levenshtein_ratio={sum(per_Leven)/batch_size:.4f}, discard_ratio={sum(per_discard)/batch_size:.4f}, loss={loss:.4f}, grad={score_head.fc2.weight.grad.norm()}"
            
            with open(log_file, 'a') as l:
                l.write(info + f', per_leven: {per_Leven}, per_discard: {per_discard}, completions: {completions}\n')
            print('\n' + info)
            optimizer.step()
            if step % save_step == 0:
                torch.save(score_head.state_dict(), os.path.join(output_dir, f'ckp_{step}.pt'))

    
if __name__ == '__main__':
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_type", type=str, required=True)
    parser.add_argument("--model_dir", type=str, required=True)
    parser.add_argument("--dataset", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--layout_json_path", type=str)
    parser.add_argument("--batch_size", type=int, default=5)
    parser.add_argument("--save_step", type=int, default=25)

    args = parser.parse_args()

    model_type = args.model_type
    model_dir = args.model_dir
    dataset = args.dataset
    os.environ['output_dir'] = args.output_dir
    if args.layout_json_path:
        os.environ['layout_json_path'] = args.layout_json_path
    batch_size = args.batch_size
    save_step = args.save_step
    
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
        
    nn.init.normal_(score_head.fc2.weight, mean=0.0, std=1e-3)
    nn.init.constant_(score_head.fc2.bias, 0)
    
    score_head = score_head.to(torch.bfloat16)
    model.model.score_mlp = score_head
    model = model.cuda()
    model.eval()
    model.model.score_mlp.train()


    optimizer = torch.optim.Adam(score_head.parameters())
    
    train(model=model, tokenizer=tokenizer, score_head=score_head, processor=processor, dataset=dataset, model_type=model_type, batch_size=batch_size, save_step=save_step)