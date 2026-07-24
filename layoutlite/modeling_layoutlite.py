import torch
import torch.nn as nn
import torch.nn.functional as F
import itertools
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any, Optional
from torch.nn import LayerNorm
from transformers import AutoProcessor, AutoTokenizer
from transformers.models.qwen3_vl.modeling_qwen3_vl import (
    Qwen3VLModelOutputWithPast, Qwen3VLModel, Qwen3VLVisionModel, Qwen3VLTextModel, Qwen3VLForConditionalGeneration, 
    BaseModelOutputWithDeepstackFeatures
)
from transformers import initialization as init
from transformers.activations import ACT2FN
from transformers.cache_utils import Cache, DynamicCache
from transformers.generation import GenerationMixin
from transformers.integrations import use_kernel_forward_from_hub, use_kernel_func_from_hub, use_kernelized_func
from transformers.masking_utils import create_causal_mask
from transformers.modeling_flash_attention_utils import FlashAttentionKwargs
from transformers.modeling_layers import GradientCheckpointingLayer
from transformers.modeling_outputs import BaseModelOutputWithPast, BaseModelOutputWithPooling, ModelOutput
from transformers.modeling_rope_utils import ROPE_INIT_FUNCTIONS, dynamic_rope_update
from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS, PreTrainedModel
from transformers.processing_utils import Unpack
from transformers.utils import TransformersKwargs, auto_docstring, can_return_tuple, torch_compilable_check
from transformers.utils.generic import is_flash_attention_requested, maybe_autocast, merge_with_config_defaults
from transformers.utils.output_capturing import capture_outputs
from transformers.models.qwen3_vl.modeling_qwen3_vl import Qwen3VLConfig, Qwen3VLTextConfig, Qwen3VLVisionConfig, Qwen3VLVisionModel
import os, cv2
import numpy as np
from scipy import ndimage
import time
from fvcore.nn import FlopCountAnalysis
import logging
logging.getLogger("fvcore.nn").setLevel(logging.ERROR)
from layoutlite.utils.utils import KeywordsStoppingCriteria, compute_mask_from_json, save_score_heatmap

dtype = torch.bfloat16
    
def cluster_two_classes(tensor: torch.Tensor) -> torch.Tensor:
    # 确保输入是 bfloat16
    assert tensor.dtype == torch.bfloat16, "输入 Tensor 必须是 torch.bfloat16"
    # 1. 展平并转为 float32 避免 bfloat16 在高频数学运算中的精度溢出
    flat_data = tensor.flatten().to(torch.float32)
    # 2. 初始化 2 个聚类中心 (选择最小值和最大值作为初始中心，能极快收敛)
    c0 = flat_data.min()
    c1 = flat_data.max()
    # 如果所有值都相等，直接返回全 False 或全 True
    if c0 == c1:
        return torch.zeros_like(tensor, dtype=torch.bool)
    # 3. 迭代 K-Means (对于一维2聚类，通常 3~5 次迭代就绝对收敛了)
    for _ in range(5):
        # 计算每个点到两个中心的距离
        dist0 = (flat_data - c0).abs()
        dist1 = (flat_data - c1).abs()
        # 分配标签：属于中心 0 还是中心 1
        labels = dist0 > dist1  # True 代表离 c1 更近
        # 更新中心
        mask0 = ~labels
        mask1 = labels
        # 防止某一个簇为空（虽然在极值初始化下极少发生）
        if mask0.any():
            c0 = flat_data[mask0].mean()
        if mask1.any():
            c1 = flat_data[mask1].mean()
    # 4. 确定哪个中心对应“较高”的那一类
    # 计算最终的决策边界（两个中心的黄金分割中点）
    alpha = float(os.environ['alpha'])
    threshold = c0 - alpha * (c1-c0)
    # 5. 用阈值对原 Tensor 进行比对，生成相同形状的 bool Tensor
    # 较高的是 True
    return tensor > threshold

PROMPT = '''You are an AI assistant specialized in converting PDF images to Markdown format. Please follow these instructions for the conversion:

            1. Text Processing:
            - Accurately recognize all text content in the PDF image without guessing or inferring.
            - Convert the recognized text into Markdown format.
            - Maintain the original document structure, including headings, paragraphs, lists, etc.

            2. Mathematical Formula Processing:
            - Convert all mathematical formulas to LaTeX format.
            - Enclose inline formulas with,(,). For example: This is an inline formula,( E = mc^2,)
            - Enclose block formulas with,\[,\]. For example:,[,frac{-b,pm,sqrt{b^2 - 4ac}}{2a},]

            3. Table Processing:
            - Convert tables to HTML format.
            - Wrap the entire table with <table> and </table>.

            4. Figure Handling:
            - Ignore figures content in the PDF image. Do not attempt to describe or convert images.

            5. Output Format:
            - Ensure the output Markdown document has a clear structure with appropriate line breaks between elements.
            - For complex layouts, try to maintain the original document's structure and format as closely as possible.

            Please strictly follow these guidelines to ensure accuracy and consistency in the conversion. Your task is to accurately convert the content of the PDF image into Markdown format without adding any extra explanations or comments.
            '''

class MultiLayerVisionTokenScoreMLP(nn.Module):
    def __init__(self, hidden_dim=2048, mid_dim=1024):
        super().__init__()
        
        self.norm = nn.LayerNorm(hidden_dim)
        self.fc1 = nn.Linear(hidden_dim, mid_dim)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(mid_dim, 1)
        self.conv = nn.Conv1d(
            in_channels=hidden_dim,
            out_channels=hidden_dim,
            kernel_size=4,
        )

    def forward(self, x):
        """
        x: (B, N, 1024)
        return: (B, N, 1)
        """
        # [L-1, N, D]
        x = x.permute(1, 2, 0)
        x = self.conv(x)
        x = x.mean(dim=-1)
        x = self.norm(x)
        x = self.fc1(x)
        x = self.act(x)
        x = self.fc2(x)
        return x
    
class Qwen3VLVisionModelUnifiedMerger(Qwen3VLVisionModel):
    
    @merge_with_config_defaults
    @capture_outputs
    def forward(
        self, hidden_states: torch.Tensor, grid_thw: torch.Tensor, **kwargs: Unpack[TransformersKwargs]
    ) -> tuple | BaseModelOutputWithDeepstackFeatures:
        """
        Args:
            hidden_states (`torch.Tensor` of shape `(seq_len, hidden_size)`):
                The final hidden states of the model.
            grid_thw (`torch.Tensor` of shape `(num_images_or_videos, 3)`):
                The temporal, height and width of feature shape of each image in LLM.

        Returns:
            `torch.Tensor`: hidden_states.
        """
        hidden_states = self.patch_embed(hidden_states)

        pos_embeds = self.fast_pos_embed_interpolate(grid_thw)
        hidden_states = hidden_states + pos_embeds

        rotary_pos_emb = self.rot_pos_emb(grid_thw)

        seq_len, _ = hidden_states.size()
        hidden_states = hidden_states.reshape(seq_len, -1)
        rotary_pos_emb = rotary_pos_emb.reshape(seq_len, -1)
        emb = torch.cat((rotary_pos_emb, rotary_pos_emb), dim=-1)
        position_embeddings = (emb.cos(), emb.sin())

        cu_seqlens = torch.repeat_interleave(grid_thw[:, 1] * grid_thw[:, 2], grid_thw[:, 0]).cumsum(
            dim=0,
            # Select dtype based on the following factors:
            #  - FA2 requires that cu_seqlens_q must have dtype int32
            #  - torch.onnx.export requires that cu_seqlens_q must have same dtype as grid_thw
            # See https://github.com/huggingface/transformers/pull/34852 for more information
            dtype=grid_thw.dtype if torch.jit.is_tracing() else torch.int32,
        )
        cu_seqlens = F.pad(cu_seqlens, (1, 0), value=0)

        deepstack_feature_lists = []
        deepstack_feature_lists_unified_merger = []
        for layer_num, blk in enumerate(self.blocks):
            hidden_states = blk(
                hidden_states,
                cu_seqlens=cu_seqlens,
                position_embeddings=position_embeddings,
                **kwargs,
            )
            if layer_num in self.deepstack_visual_indexes:
                deepstack_feature = self.deepstack_merger_list[self.deepstack_visual_indexes.index(layer_num)](
                    hidden_states
                )
                deepstack_feature_unified_merger = self.merger(
                    hidden_states
                )
                deepstack_feature_lists.append(deepstack_feature)
                deepstack_feature_lists_unified_merger.append(deepstack_feature_unified_merger)

        merged_hidden_states = self.merger(hidden_states)

        return BaseModelOutputWithDeepstackFeatures(
            last_hidden_state=hidden_states,
            pooler_output=merged_hidden_states,
            deepstack_features=deepstack_feature_lists+deepstack_feature_lists_unified_merger,
        )

class Qwen3VLModelVisionSelect(Qwen3VLModel):
    
    def __init__(self, config):
        super().__init__(config)
        self.visual = Qwen3VLVisionModelUnifiedMerger._from_config(config.vision_config)
        self.language_model = Qwen3VLTextModel._from_config(config.text_config)
        self.rope_deltas = None  # cache rope_deltas here
        # Vision select
        self.score_mlp = MultiLayerVisionTokenScoreMLP(hidden_dim=1024, mid_dim=512)
        # Initialize weights and apply final processing
        self.post_init()
        
    @auto_docstring
    @can_return_tuple
    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: torch.Tensor | None = None,
        position_ids: torch.LongTensor | None = None,
        past_key_values: Cache | None = None,
        inputs_embeds: torch.FloatTensor | None = None,
        pixel_values: torch.Tensor | None = None,
        pixel_values_videos: torch.FloatTensor | None = None,
        image_grid_thw: torch.LongTensor | None = None,
        video_grid_thw: torch.LongTensor | None = None,
        mm_token_type_ids: torch.IntTensor | None = None,
        **kwargs: Unpack[TransformersKwargs],
    ):
        r"""
        image_grid_thw (`torch.LongTensor` of shape `(num_images, 3)`, *optional*):
            The temporal, height and width of feature shape of each image in LLM.
        video_grid_thw (`torch.LongTensor` of shape `(num_videos, 3)`, *optional*):
            The temporal, height and width of feature shape of each video in LLM.
        """
        if (input_ids is None) ^ (inputs_embeds is not None):
            raise ValueError("You must specify exactly one of input_ids or inputs_embeds")

        if inputs_embeds is None:
            inputs_embeds = self.get_input_embeddings()(input_ids)

        image_mask = None
        video_mask = None

        if pixel_values is not None:
            image_outputs: BaseModelOutputWithDeepstackFeatures = self.get_image_features(
                pixel_values, image_grid_thw, return_dict=True
            )
            image_embeds = image_outputs.pooler_output
            deepstack_image_embeds = image_outputs.deepstack_features
            deepstack_image_embeds, deepstack_image_embeds_unified_merger = deepstack_image_embeds[0:3], deepstack_image_embeds[3:6]
            image_embeds = torch.cat(image_embeds, dim=0).to(inputs_embeds.device, inputs_embeds.dtype)
            
            group = os.environ['group']
            infer_mode = os.environ.get('infer_mode', None)
            output_dir = os.environ['output_dir']
            firered_image_embed = torch.stack(list(deepstack_image_embeds_unified_merger) + [image_embeds], dim=0).to(dtype)
            
            if infer_mode != 'execute_layoutlite':
                torch.save(firered_image_embed, os.path.join(output_dir, f'cache/firered_image_embed.pt'))

            torch.cuda.synchronize()
            start = time.time()
            scores = self.score_mlp(firered_image_embed).squeeze(-1)
            torch.cuda.synchronize()
            end = time.time()
            score_time = end - start
            
            if infer_mode != 'execute_layoutlite':
                with open(os.path.join(output_dir, 'efficiency.txt'), 'a') as f:
                    f.write(f'score: {score_time} ')
                
            processed_probs = torch.sigmoid(scores)
            # mask = probs.squeeze(-1) > 0.1
            # processed_probs = probs.squeeze(-1)
            # image_name = os.environ['image_name']
            
            if infer_mode == 'execute_layoutlite':
                image_name = os.environ['image_name']
                torch.save(scores, os.path.join(output_dir, f'cache/scores/{image_name}.pt'))
                assert False
            

            elif 'eval' in group:
                mask = cluster_two_classes(scores)

            else:
                if 'layout_json_path' in os.environ:
                    image_name = os.environ['image_name']
                    t, h, w = image_grid_thw[0]
                    structure = np.ones((3, 3), dtype=bool)
                    box = torch.from_numpy(
                        # ndimage.binary_erosion(
                            compute_mask_from_json(
                                image_name=os.environ['image_name'],
                                h=h.item()//2,
                                w=w.item()//2
                            ),
                            # structure=structure
                        # )
                    ).view(-1)
                    torch.save(box, os.path.join(output_dir, f'cache/box.pt'))
                mask = torch.bernoulli(processed_probs.clamp(1e-6, 1 - 1e-6)).bool()
                
                
            torch.save(mask, os.path.join(output_dir, f'cache/firered_mask.pt'))
            if mask.any():
                deepstack_image_embeds = tuple([feat[mask] for feat in deepstack_image_embeds])
                image_embeds = image_embeds[mask]
                    
                discard_ratio = (~mask).float().mean().item()
                
                # ====== 前提 ======
                # input_ids:        (1, L)
                # inputs_embeds:    (1, L, D)
                # mask:             (image_seq_len,)  bool tensor
                # image_token_id:   151655

                image_token_id = 151655

                # -------- step1: 找到 image token 位置 --------
                ids = input_ids[0]  # (L,)
                image_pos = (ids == image_token_id)  # (L,)

                # sanity check（强烈建议保留）
                image_indices = torch.nonzero(image_pos, as_tuple=True)[0]  # (image_seq_len,)
                assert image_indices.shape[0] == mask.shape[0], \
                    f"mask长度({mask.shape[0]})和image token数量({image_indices.shape[0]})不一致"

                # -------- step2: 构造全局 keep_mask --------
                keep_mask = torch.ones_like(ids, dtype=torch.bool)

                # 先把所有 image token 标为 False
                keep_mask[image_indices] = False

                # 再把需要保留的设回 True
                keep_mask[image_indices[mask]] = True

                # -------- step3: 原地“更新” input_ids --------
                input_ids = input_ids[:, keep_mask]   # (1, new_L)

                # -------- step4: 原地“更新” inputs_embeds --------
                inputs_embeds = inputs_embeds[:, keep_mask, :]  # (1, new_L, D)

                # -------- step5:（可选）同步 attention_mask --------
                if attention_mask is not None:
                    attention_mask = attention_mask[:, keep_mask]
                    
                # 保证 mask 在同一 device
                keep_mask = keep_mask.to(position_ids.device)

                # --- position_ids（关键）---
                position_ids = position_ids[:, :, keep_mask]

                # -------- step6:（强烈建议）清理中间变量 --------
                del ids, image_pos, image_indices, keep_mask
            
            
            image_mask, _ = self.get_placeholder_mask(
                input_ids, inputs_embeds=inputs_embeds, image_features=image_embeds
            )
            inputs_embeds = inputs_embeds.masked_scatter(image_mask, image_embeds)

        if pixel_values_videos is not None:
            video_outputs: BaseModelOutputWithDeepstackFeatures = self.get_video_features(
                pixel_values_videos, video_grid_thw, return_dict=True
            )
            video_embeds = video_outputs.pooler_output
            deepstack_video_embeds = video_outputs.deepstack_features
            video_embeds = torch.cat(video_embeds, dim=0).to(inputs_embeds.device, inputs_embeds.dtype)
            _, video_mask = self.get_placeholder_mask(
                input_ids, inputs_embeds=inputs_embeds, video_features=video_embeds
            )
            inputs_embeds = inputs_embeds.masked_scatter(video_mask, video_embeds)

        visual_pos_masks = None
        deepstack_visual_embeds = None
        if image_mask is not None and video_mask is not None:
            # aggregate visual_pos_masks and deepstack_visual_embeds
            image_mask = image_mask[..., 0]
            video_mask = video_mask[..., 0]
            visual_pos_masks = image_mask | video_mask
            deepstack_visual_embeds = []
            image_mask_joint = image_mask[visual_pos_masks]
            video_mask_joint = video_mask[visual_pos_masks]
            for img_embed, vid_embed in zip(deepstack_image_embeds, deepstack_video_embeds):
                embed_joint = img_embed.new_zeros(visual_pos_masks.sum(), img_embed.shape[-1]).to(img_embed.device)
                embed_joint[image_mask_joint, :] = img_embed
                embed_joint[video_mask_joint, :] = vid_embed
                deepstack_visual_embeds.append(embed_joint)
        elif image_mask is not None:
            image_mask = image_mask[..., 0]
            visual_pos_masks = image_mask
            deepstack_visual_embeds = deepstack_image_embeds
        elif video_mask is not None:
            video_mask = video_mask[..., 0]
            visual_pos_masks = video_mask
            deepstack_visual_embeds = deepstack_video_embeds

        if position_ids is None:
            position_ids = self.compute_3d_position_ids(
                input_ids=input_ids,
                image_grid_thw=image_grid_thw,
                video_grid_thw=video_grid_thw,
                inputs_embeds=inputs_embeds,
                attention_mask=attention_mask,
                past_key_values=past_key_values,
                mm_token_type_ids=mm_token_type_ids,
            )
        if pixel_values is not None:
            torch.cuda.synchronize()
            start = time.time()
        outputs = self.language_model(
            input_ids=None,
            position_ids=position_ids,
            attention_mask=attention_mask,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            visual_pos_masks=visual_pos_masks,
            deepstack_visual_embeds=deepstack_visual_embeds,
            **kwargs,
        )

        if pixel_values is not None and 'output_dir' in os.environ:
            torch.cuda.synchronize()
            kvcache = outputs.past_key_values
            total_bytes = 0

            for layer in kvcache:
                k, v = layer[:2]
                total_bytes += k.numel() * k.element_size()
                total_bytes += v.numel() * v.element_size()

            end = time.time()
            prefill_time = end - start
            output_dir = os.environ['output_dir']
            with open(os.path.join(output_dir, 'efficiency.txt'), 'a') as f:
                f.write(f'ttft: {prefill_time} kv:{total_bytes/1024**2:.2f} ')
        return Qwen3VLModelOutputWithPast(
            **outputs,
            rope_deltas=self.rope_deltas,
        )

class Qwen3VLTokenSelect(Qwen3VLForConditionalGeneration):

    def __init__(self, config):
        super().__init__(config)
        self.model = Qwen3VLModelVisionSelect(config)
        self.post_init()

def single_infer_qwen3vl(model=None, tokenizer=None, processor=None, image_path=None, prompt=PROMPT, max_new_tokens=3000):
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": image_path},
                {"type": "text", "text": prompt},
            ],
        }
    ]
    
    inputs = processor.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=True,
        return_dict=True,
        return_tensors="pt"
    )
    
    inputs = {k: v.cuda() for k, v in inputs.items()}

    with torch.no_grad():
        stopping_criteria = KeywordsStoppingCriteria(
            ['<|im_end|>'],
            tokenizer,
            inputs["input_ids"]
        )
        outputs = model.generate(
            **inputs,
            do_sample=False,
            max_new_tokens=max_new_tokens,
            stopping_criteria=[stopping_criteria]
        )

    generated_ids_trimmed = [
        out_ids[len(in_ids):]
        for in_ids, out_ids in zip(inputs["input_ids"], outputs)
    ]

    text = processor.batch_decode(
        generated_ids_trimmed,
        skip_special_tokens=True,
        clean_up_tokenization_spaces=False
    )[0]
    
    return text
