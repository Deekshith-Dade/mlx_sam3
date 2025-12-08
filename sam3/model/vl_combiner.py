from copy import copy

import mlx.core as mx
import mlx.nn as nn
from sam3.model.necks import Sam3DualViTDetNeck

class SAM3VLBackbone(nn.Module):
    def __init__(
        self,
        visual: Sam3DualViTDetNeck,
        text,
        compile_visual: bool = False,
        act_ckpt_whole_vision_backbone: bool = False,
        act_ckpt_whole_language_backbone: bool = False,
        scalp=0
    ):
        super().__init__()
        # TODO: check if we can compile like they do in PyTorch version
        self.vision_backbone: Sam3DualViTDetNeck = visual
        self.language_backbone = text
        self.scalp = scalp
        
        # TODO: Learn more about this from pytorch
        self.act_ckpt_whole_vision_backbone = act_ckpt_whole_vision_backbone
        self.act_ckpt_whole_language_backbone = act_ckpt_whole_language_backbone
    
    def __call__(self):
        pass

    def call_image(self, samples: mx.array):
        return self._call_image_no_ack_ckpt(
            samples=samples,
        )

    def _call_image_no_ack_ckpt(self, samples):
        sam3_features, sam3_pos, sam2_features, sam2_pos = self.vision_backbone(samples)

        if self.scalp > 0:
            sam3_features, sam3_pos = (
                sam3_features[: -self.scalp],
                sam3_pos[: -self.scalp],
            )
            if sam2_features is not None and sam2_pos is not None:
                sam2_features, sam2_pos = (
                    sam2_features[:-self.scalp],
                    sam2_pos[: -self.scalp],
                )
        
        sam2_output = None

        if sam2_features is not None and sam2_pos is not None:
            sam2_src = sam2_features[-1]
            sam2_output = {
                "vision_features": sam2_src,
                "vision_pos_enc": sam2_pos,
                "backbone_fpn": sam2_features
            }
        
        sam3_src = sam3_features[-1]
        output = {
            "vision_features": sam3_src,
            "vision_pos_enc": sam3_pos,
            "backbone_fpn": sam3_features,
            "sam2_backbone_out": sam2_output
        }

        return output
    
    def call_text(
       self,
       captions,
       input_boxes=None,
       additional_text=None 
    ):
        return self._call_text_no_ack_ckpt(
            captions=captions,
            input_boxes=input_boxes,
            additional_text=additional_text
        )

    def _call_text_no_ack_ckpt(
        self,
        captions,
        input_boxes=None,
        additional_text=None,
    ):
        output = {}

        text_to_encode = copy(captions)
        if additional_text is not None:
            text_to_encode += additional_text
        
        # TODO: pytorch uses a sdpa_kernel here
        text_attention_mask, text_memroy, text_embeds = self.language_backbone(
            text_to_encode, input_boxes
        )

        if additional_text is not None:
            output["additional_text_features"] = text_memroy[:, -len(additional_text) :]
            output["additional_text_mask"] = text_attention_mask[
                -len(additional_text) :
            ]
        
        text_memory = text_memroy[: ,:len(captions)]
        text_attention_mask = text_attention_mask[:, len(captions)]
        text_embeds = text_embeds[:, : len(captions)]
        output["langugage_features"] = text_memory
        output["language_mask"] = text_attention_mask
        output["language_embeds"] = (
            text_embeds
        )

        return output