#
# Copyright (c) 2025 Huawei Technologies Co., Ltd. All Rights Reserved.
# This file is a part of the vllm-ascend project.
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
#
"""Ascend Step5 vision-language wrapper.

Upstream vLLM's Step5ForConditionalGeneration (step5_vl.py -> step3p7.py)
implements the complete multimodal pipeline: the PerceptionEncoder vision
tower, the vit_large_projector and the Step3VL processor family. Its
language model is built via init_vllm_registered_model which resolves
text_config.architectures == ["Step5ForCausalLM"] through vLLM's
ModelRegistry — and this plugin already registers the
AscendStep5ForCausalLM override for that architecture. Therefore a thin
subclass is sufficient: the upstream multimodal wrapper runs with the
Ascend-optimized text model as backbone, while the vision tower transparently
picks up the platform op overrides (AscendMMEncoderAttention,
AscendQuickGELU, Ascend parallel linears) registered via
CustomOp.register_oot.

The multimodal processor registration is inherited through the MRO from
Step3VLForConditionalGeneration (_processor_factory is looked up on
the class hierarchy by MultimodalRegistry), so the Step3VL
processor/dummy-input builders serve this class unchanged.
"""

from vllm.model_executor.models.step5_vl import (
    Step5ForConditionalGeneration as UpstreamStep5ForConditionalGeneration,
)


class AscendStep5ForConditionalGeneration(UpstreamStep5ForConditionalGeneration):
    """Upstream Step5 VLM wrapper with the Ascend Step5 text backbone."""
