from collections import OrderedDict
from typing import Callable, List, Optional, Tuple, Union

import mlx.core as mx
import mlx.nn as nn

from .model_misc import LayerScale

class MLP(nn.Module):
    def __init__(
        self,
        d_model: int,
        mlp_width: int,
        act_layer: Callable[[], nn.Module] = nn.GELU
    ):
        self.c_fc = nn.Linear(d_model, mlp_width)
        self.gelu = act_layer()
        self.c_proj = nn.Linear(mlp_width, d_model)
        
    def __call__(self, x: mx.array) -> mx.array:
        x = self.c_fc(x)
        x = self.gelu(x)
        x = self.c_proj(x)
        return x

class ResidualAttentionBlock(nn.Module):
    def __init__(
        self,
        d_model: int,
        n_head: int,
        mlp_ratio: float = 4.0,
        ls_init_value: Optional[float] = None,
        act_layer: Callable[[], nn.Module] = nn.GELU,
        norm_layer: Callable[[int], nn.Module] = nn.LayerNorm,
    ):
        super().__init__()
        # Attention
        self.attn = nn.MultiHeadAttention(d_model, n_head, bias=True)

        # LayerNorm, LayerScale
        self.ln_1 = norm_layer(d_model)
        self.ln_2 = norm_layer(d_model)

        self.ls_1 = (
            LayerScale(d_model, ls_init_value)
            if ls_init_value is not None
            else nn.Identity()
        )
        self.ls_2 = (
            LayerScale(d_model, ls_init_value)
            if ls_init_value is not None
            else nn.Identity()
        )

        # MLP
        mlp_width = int(d_model * mlp_ratio)
        self.mlp = MLP(d_model, mlp_width, act_layer=act_layer)

    def attention(
        self,
        q_x: mx.array,
        k_x: Optional[mx.array] = None,
        v_x: Optional[mx.array] = None,
        attn_mask: Optional[mx.array] = None,
    ) -> mx.array:
        k_x = k_x if k_x is not None else q_x
        v_x = v_x if v_x is not None else q_x
        if attn_mask is not None:
            # Leave boolean masks as is
            if not attn_mask.dtype == mx.bool_:
                attn_mask = attn_mask.astype(q_x.dtype)

        return self.attn(q_x, k_x, v_x, mask=attn_mask)

    def __call__(
        self,
        q_x: mx.array,
        k_x: Optional[mx.array] = None,
        v_x: Optional[mx.array] = None,
        attn_mask: Optional[mx.array] = None,
    ) -> mx.array:
        k_x = (
            self.ln_1_kv(k_x) if hasattr(self, "ln_1_kv") and k_x is not None else None
        )
        v_x = (
            self.ln_1_kv(v_x) if hasattr(self, "ln_1_kv") and v_x is not None else None
        )
        x = q_x + self.ls_1(
            self.attention(q_x=self.ln_1(q_x), k_x=k_x, v_x=v_x, attn_mask=attn_mask)
        )
        x = x + self.ls_2(self.mlp(self.ln_2(x)))
        return x


class Transformer(nn.Module):
    def __init__(
        self,
        width: int,
        layers: int,
        heads: int,
        mlp_ratio: float = 4.0,
        ls_init_value: Optional[float] = None,
        act_layer: Callable[[], nn.Module] = nn.GELU,
        norm_layer: Callable[[int], nn.Module] = nn.LayerNorm,
        compile_mode: Optional[str] = None,
        use_act_checkpoint: bool = False,
    ):
        super().__init__()
        self.width = width
        self.layers = layers
        self.grad_checkpointing = use_act_checkpoint
        self.resblocks = [
                ResidualAttentionBlock(
                    width,
                    heads,
                    mlp_ratio,
                    ls_init_value=ls_init_value,
                    act_layer=act_layer,
                    norm_layer=norm_layer,
                )
                for _ in range(layers)
            ]
        

        # if compile_mode is not None:
        #     self.forward = torch.compile(
        #         self.forward, mode=compile_mode, fullgraph=True
        #     )
        #     if self.grad_checkpointing:
        #         torch._dynamo.config.optimize_ddp = False

    def __call__(
        self,
        x: mx.array,
        attn_mask: Optional[mx.array] = None,
    ) -> mx.array:
        for _, r in enumerate(self.resblocks):
            # if (
            #     self.grad_checkpointing
            #     and not torch.jit.is_scripting()
            #     and self.training
            # ):
            #     x = checkpoint(r, x, None, None, attn_mask, use_reentrant=False)
            # else:
            #     x = r(
            #         x,
            #         attn_mask=attn_mask,
            #     )
            x = r(
                x,
                attn_mask=attn_mask,
            )
        return x


def text_global_pool(
    x: mx.array, text: Optional[mx.array] = None, pool_type: str = "argmax"
) -> Tuple[mx.array, mx.array]:
    if pool_type == "first":
        pooled, tokens = x[:, 0], x[:, 1:]
    elif pool_type == "last":
        pooled, tokens = x[:, -1], x[:, :-1]
    elif pool_type == "argmax":
        # take features from the eot embedding (eot_token is the highest number in each sequence)
        assert text is not None
        pooled, tokens = x[mx.arange(x.shape[0]), text.argmax(dim=-1)], x
    else:
        pooled = tokens = x
    return pooled, tokens


class TextTransformer(nn.Module):
    def __init__(
        self,
        context_length: int = 77,
        vocab_size: int = 49408,
        width: int = 512,
        heads: int = 8,
        layers: int = 12,
        mlp_ratio: float = 4.0,
        ls_init_value: Optional[float] = None,
        output_dim: int = 512,
        no_causal_mask: bool = False,
        pool_type: str = "none",  # no pooling
        proj_bias: bool = False,
        act_layer: Callable = nn.GELU,
        norm_layer: Callable = nn.LayerNorm,
        output_tokens: bool = False,
        use_ln_post: bool = True,
        compile_mode: Optional[str] = None,
        use_act_checkpoint: bool = False,
    ):
        super().__init__()
        assert pool_type in ("first", "last", "argmax", "none")
        self.output_tokens = output_tokens
        self.num_pos = self.context_length = context_length
        self.vocab_size = vocab_size
        self.width = width
        self.output_dim = output_dim
        self.heads = heads
        self.pool_type = pool_type

        self.token_embedding = nn.Embedding(self.vocab_size, width)
        self.positional_embedding = mx.zeros((self.num_pos, width))
        self.transformer = Transformer(
            width=width,
            layers=layers,
            heads=heads,
            mlp_ratio=mlp_ratio,
            ls_init_value=ls_init_value,
            act_layer=act_layer,
            norm_layer=norm_layer,
            compile_mode=compile_mode,
            use_act_checkpoint=use_act_checkpoint,
        )
        self.ln_final = norm_layer(width) if use_ln_post else nn.Identity()
        if no_causal_mask:
            self.attn_mask = None
        else:
            self.attn_mask = self.build_causal_mask()
            
        if proj_bias:
            self.text_projection = nn.Linear(width, output_dim)
        else:
            self.text_projection = mx.zeros((width, output_dim))

    def build_causal_mask(self) -> mx.array:
        # lazily create causal attention mask, with full attention between the tokens
        # pytorch uses additive attention mask; fill with -inf
        mask = mx.full((self.num_pos, self.num_pos), -float('inf'))
        mask = mx.triu(mask, k=1)
        return mask

    def __call__(
        self, text: mx.array
    ) -> Union[mx.array, Tuple[mx.array, mx.array]]:
        seq_len = text.shape[1]
        x = self.token_embedding(text)  # [batch_size, n_ctx, d_model]

        attn_mask = self.attn_mask
        if attn_mask is not None:
            attn_mask = attn_mask[:seq_len, :seq_len]

        x = x + self.positional_embedding[:seq_len]
        x = self.transformer(x, attn_mask=attn_mask)

        x = self.ln_final(x)
        pooled, tokens = text_global_pool(x, text, pool_type=self.pool_type)
        if self.text_projection is not None:
            if isinstance(self.text_projection, nn.Linear):
                pooled = self.text_projection(pooled)
            else:
                pooled = pooled @ self.text_projection
        if self.output_tokens:
            return pooled, tokens
        return pooled


class VETextEncoder(nn.Module):
    def __init__(
        self,
        d_model: int,
        tokenizer: Callable,
        width: int = 1024,
        heads: int = 16,
        layers: int = 24,
        context_length: int = 32,
        vocab_size: int = 49408,
        use_ln_post: bool = True,
        compile_mode: Optional[str] = None,
        use_act_checkpoint: bool = True,
    ):
        super().__init__()
        self.context_length = context_length
        self.use_ln_post = use_ln_post
        self.tokenizer = tokenizer

        self.encoder = TextTransformer(
            context_length=self.context_length,
            vocab_size=vocab_size,
            width=width,
            heads=heads,
            layers=layers,
            # we want the tokens, not just the pooled output
            output_tokens=True,
            use_ln_post=use_ln_post,
            compile_mode=compile_mode,
            use_act_checkpoint=use_act_checkpoint,
        )
        self.resizer = nn.Linear(self.encoder.width, d_model)

    def __call__(
        self,
        text: Union[List[str], Tuple[mx.array, mx.array, dict]],
        input_boxes: Optional[List] = None,
    ) -> Tuple[mx.array, mx.array, mx.array]:
        if isinstance(text[0], str):
            # no use case for this
            assert input_boxes is None or len(input_boxes) == 0, "not supported"

            # Encode the text
            tokenized = self.tokenizer(text, context_length=self.context_length)  # [b, seq_len]
            text_attention_mask = (tokenized != 0).astype(mx.bool_)

            # manually embed the tokens
            inputs_embeds = self.encoder.token_embedding(
                tokenized
            )  # [b, seq_len, d=1024]
            _, text_memory = self.encoder(tokenized)  # [b, seq_len, d=1024]

            assert text_memory.shape[1] == inputs_embeds.shape[1]
            # Invert attention mask because its the opposite in pytorch transformer
            text_attention_mask = mx.not_equal(text_attention_mask, 1)
            # Transpose memory because pytorch's attention expects sequence first
            text_memory = text_memory.transpose(1, 0, 2)
            # Resize the encoder hidden states to be of the same d_model as the decoder
            text_memory_resized = self.resizer(text_memory)
        else:
            # The text is already encoded, use as is.
            text_attention_mask, text_memory_resized, tokenized = text
            inputs_embeds = tokenized["inputs_embeds"]
            assert (
                input_boxes is None or len(input_boxes) == 0
            ), "Can't replace boxes in text if it's already encoded"

        # Note that the input_embeds are returned in pytorch's convention (sequence first)
        return (
            text_attention_mask,
            text_memory_resized,
            inputs_embeds.transpose(1, 0, 2),
        )
