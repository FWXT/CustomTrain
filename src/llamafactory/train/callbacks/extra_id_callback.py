# Copyright 2025 the LlamaFactory team.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from typing import TYPE_CHECKING

import torch
from transformers import TrainerCallback, TrainerControl, TrainerState

from ...extras import logging


if TYPE_CHECKING:
    from transformers import PreTrainedTokenizer, TrainingArguments
    from ...hparams import FinetuningArguments


logger = logging.get_logger(__name__)


class ExtraIdFreezeCallback(TrainerCallback):
    r"""Callback to freeze extra_id token embeddings for the first K steps."""

    def __init__(
        self,
        tokenizer: "PreTrainedTokenizer",
        finetuning_args: "FinetuningArguments",
    ) -> None:
        self.tokenizer = tokenizer
        self.freeze_steps = finetuning_args.freeze_extra_id_steps
        self.extra_id_token_ids = []
        self.frozen_params = []
        self.is_frozen = False
        
        if self.freeze_steps > 0:
            # Find all extra_id tokens by encoding <extra_id_{i}> patterns
            self.extra_id_token_ids = []
            # Try to find extra_id tokens by encoding them directly
            for i in range(100):  # Check up to extra_id_999
                extra_id_token = f"<extra_id_{i}>"
                try:
                    # Encode the token and get its ID(s)
                    token_ids = tokenizer.encode(extra_id_token, add_special_tokens=False)
                    # If the token exists and is encoded as a single token, add it
                    if len(token_ids) == 1:
                        token_id = token_ids[0]
                        if token_id not in self.extra_id_token_ids:
                            self.extra_id_token_ids.append(token_id)
                    elif len(token_ids) == 0:
                        # If encoding returns empty, the token doesn't exist
                        break
                    else:
                        # If encoding returns multiple tokens, it means the token is not in vocab as a single token
                        # We still add all the token IDs to be safe
                        for token_id in token_ids:
                            if token_id not in self.extra_id_token_ids:
                                self.extra_id_token_ids.append(token_id)
                except Exception:
                    # If encoding fails, skip this token
                    continue
            
            logger.info_rank0(f"Found {len(self.extra_id_token_ids)} extra_id tokens: {self.extra_id_token_ids}")

    def on_train_begin(
        self,
        args: "TrainingArguments",
        state: "TrainerState",
        control: "TrainerControl",
        model=None,
        **kwargs,
    ) -> None:
        if self.freeze_steps > 0 and len(self.extra_id_token_ids) > 0:
            self._freeze_extra_id_embeddings(model)

    def on_step_end(
        self,
        args: "TrainingArguments",
        state: "TrainerState",
        control: "TrainerControl",
        model=None,
        **kwargs,
    ) -> None:
        if self.is_frozen and state.global_step >= self.freeze_steps:
            self._unfreeze_extra_id_embeddings(model)
            logger.info_rank0(f"Unfroze extra_id token embeddings at step {state.global_step}")

    def _freeze_extra_id_embeddings(self, model) -> None:
        """Freeze the embeddings of extra_id tokens."""
        if hasattr(model, 'module'):  # Handle DataParallel/DistributedDataParallel
            actual_model = model.module
        else:
            actual_model = model
            
        # Get input embeddings
        input_embeddings = actual_model.get_input_embeddings()
        if input_embeddings is None:
            logger.warning_rank0("Could not find input embeddings to freeze")
            return
            
        # Store original requires_grad state and freeze specific token embeddings
        for token_id in self.extra_id_token_ids:
            if token_id < input_embeddings.weight.size(0):
                param = input_embeddings.weight[token_id]
                self.frozen_params.append((param, param.requires_grad))
                param.requires_grad_(False)
                
        self.is_frozen = True
        logger.info_rank0(f"Froze embeddings for {len(self.extra_id_token_ids)} extra_id tokens for {self.freeze_steps} steps")

    def _unfreeze_extra_id_embeddings(self, model) -> None:
        """Unfreeze the embeddings of extra_id tokens."""
        # Restore original requires_grad state
        for param, original_requires_grad in self.frozen_params:
            param.requires_grad_(original_requires_grad)
            
        self.frozen_params.clear()
        self.is_frozen = False