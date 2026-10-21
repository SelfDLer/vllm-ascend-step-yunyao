# SPDX-License-Identifier: Apache-2.0
"""Validate a complete Step5 update across streamed checkpoint chunks."""


class Step5WeightUpdateGuard:
    def __init__(self, model, *, stage_to_cpu=False):
        from vllm.models.step5.model import _reset_fused_qkv_indexer_load_state

        self.model = model
        self.stage_to_cpu = stage_to_cpu
        self.original_load = model.load_weights
        self.expected = set(dict(model.named_parameters()))
        self.loaded = set()
        _reset_fused_qkv_indexer_load_state(model)
        model.load_weights = self.load_weights

    def load_weights(self, weights):
        from vllm.model_executor.model_loader.mtp_validation import (
            disable_mtp_completeness_check,
        )

        # A chunk is partial; completeness is enforced at transaction finish.
        with disable_mtp_completeness_check():
            if self.stage_to_cpu:
                weights = ((name, tensor.cpu()) for name, tensor in weights)
            loaded = self.original_load(weights)
        if loaded is None:
            raise RuntimeError("Step5 loader did not return loaded parameter names")
        self.loaded.update(loaded)
        return loaded

    def finish(self):
        from vllm.models.step5.model import _validate_fused_qkv_indexer_weights

        try:
            missing = self.expected - self.loaded
            if missing:
                raise ValueError(f"Incomplete Step5 weight update: {sorted(missing)}")
            _validate_fused_qkv_indexer_weights(self.model)
        finally:
            self.abort()

    def abort(self):
        self.model.load_weights = self.original_load
