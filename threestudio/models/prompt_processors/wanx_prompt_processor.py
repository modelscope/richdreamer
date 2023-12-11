import json
import os
import torch
import torch.nn as nn
from dataclasses import dataclass
from transformers import AutoTokenizer, CLIPTextModel

import threestudio
from extern.wanx.atom import data, models, ops
from extern.wanx.config import cfg
from threestudio.models.prompt_processors.base import (PromptProcessor,
                                                       hash_prompt,)
from threestudio.utils.misc import cleanup
from threestudio.utils.typing import *


def encode_text(m, tokens):
    b, s = tokens.shape
    mask = tokens.ne(m.pad_token).long()

    # embeddings
    x = (
        m.token_embedding(tokens)
        + m.type_embedding(torch.zeros_like(tokens))
        + m.pos_embedding(m.pad_token + torch.cumsum(mask, dim=1) * mask)
    )
    x = m.norm(x)
    x = m.dropout(x)

    # blocks
    for block in m.blocks[:-1]:
        x = block(x, mask.view(b, 1, 1, s))

    words = x

    sentence = m.blocks[-1](x, mask.view(b, 1, 1, s))
    mask = tokens.ne(m.pad_token).unsqueeze(-1).to(sentence)
    sentence = (sentence * mask).sum(dim=1) / mask.sum(dim=1)
    sentence = m.head(sentence)

    # return {
    #     'context': words,
    #     'y': sentence
    # }

    res = torch.cat([words, sentence.unsqueeze(1)], dim=1)  # [1, 78, 1024]
    return res


def wanx_model_init(device, model_path):
    # [model] clip
    tokenizer = data.HuggingfaceTokenizer(
        path=f"{model_path}/clip_tokenizer", length=cfg.text_len, clean=True
    )
    clip = (
        getattr(models, cfg.clip_model)(
            vocab_size=len(tokenizer.tokenizer), pretrained=False
        )
        .eval()
        .requires_grad_(False)
        .textual.to(device)
    )
    clip.load_state_dict(torch.load(f"{model_path}/clip.pth", map_location="cpu"))

    return clip, tokenizer


@threestudio.register("wanx-prompt-processor")
class WanXPromptProcessor(PromptProcessor):
    @dataclass
    class Config(PromptProcessor.Config):
        prefix: str = "<wanx> "

    cfg: Config

    ### these functions are unused, kept for debugging ###
    def configure_text_encoder(self) -> None:
        os.environ["TOKENIZERS_PARALLELISM"] = "false"  # TODO check

        self.text_encoder, self.tokenizer = wanx_model_init(
            device=self.device, model_path=self.cfg.pretrained_model_name_or_path
        )

        for p in self.text_encoder.parameters():
            p.requires_grad_(False)

    def destroy_text_encoder(self) -> None:
        del self.tokenizer
        del self.text_encoder
        cleanup()

    def get_text_embeddings(
        self, prompt: Union[str, List[str]], negative_prompt: Union[str, List[str]]
    ) -> Tuple[Float[Tensor, "B 77 768"], Float[Tensor, "B 77 768"]]:
        if isinstance(prompt, str):
            prompt = [prompt]
        if isinstance(negative_prompt, str):
            negative_prompt = [negative_prompt]
        # Tokenize text and get embeddings
        tokens = self.tokenizer(
            prompt,
            padding="max_length",
            max_length=self.tokenizer.model_max_length,
            return_tensors="pt",
        )
        uncond_tokens = self.tokenizer(
            negative_prompt,
            padding="max_length",
            max_length=self.tokenizer.model_max_length,
            return_tensors="pt",
        )

        with torch.no_grad():
            text_embeddings = self.text_encoder(tokens.input_ids.to(self.device))[0]
            uncond_text_embeddings = self.text_encoder(
                uncond_tokens.input_ids.to(self.device)
            )[0]

        return text_embeddings, uncond_text_embeddings

    ###

    @staticmethod
    def spawn_func(pretrained_model_name_or_path, prompts, cache_dir, device=None):
        os.environ["TOKENIZERS_PARALLELISM"] = "false"  # TODO check

        text_encoder, tokenizer = wanx_model_init(
            device=device, model_path=pretrained_model_name_or_path
        )

        with torch.no_grad():
            text_embeddings = encode_text(text_encoder, tokenizer(prompts).to(device))

        for prompt, embedding in zip(prompts, text_embeddings):
            torch.save(
                embedding,
                os.path.join(
                    cache_dir,
                    f"{hash_prompt(pretrained_model_name_or_path, prompt)}.pt",
                ),
            )

        del text_encoder
