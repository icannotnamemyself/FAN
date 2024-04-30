# Copyright The Lightning team.
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
from typing import Tuple, Union

import torch
from torch import Tensor

from torchmetrics.utilities.checks import _check_same_shape
# Copyright The Lightning team.
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
from typing import Any, Optional, Sequence, Union

from torch import Tensor, tensor

from torchmetrics.functional.regression.mape import (
    _mean_absolute_percentage_error_compute,
    _mean_absolute_percentage_error_update,
)
from torchmetrics.metric import Metric
from torchmetrics.utilities.imports import _MATPLOTLIB_AVAILABLE
from torchmetrics.utilities.plot import _AX_TYPE, _PLOT_OUT_TYPE

if not _MATPLOTLIB_AVAILABLE:
    __doctest_skip__ = ["MeanAbsolutePercentageError.plot"]


class MeanAbsolutePercentageError(Metric):
    is_differentiable: bool = True
    higher_is_better: bool = False
    full_state_update: bool = False
    plot_lower_bound: float = 0.0

    sum_abs_per_error: Tensor
    total: Tensor

    def __init__(
        self,
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)

        self.add_state("sum_abs_per_error", default=tensor(0.0), dist_reduce_fx="sum")
        self.add_state("total", default=tensor(0.0), dist_reduce_fx="sum")

    def update(self, preds: Tensor, target: Tensor) -> None:
        """Update state with predictions and targets."""
        sum_abs_per_error, num_obs = _mean_absolute_percentage_error_update(preds, target)

        self.sum_abs_per_error += sum_abs_per_error
        self.total += num_obs

    def compute(self) -> Tensor:
        """Compute mean absolute percentage error over state."""
        return _mean_absolute_percentage_error_compute(self.sum_abs_per_error, self.total)

    def plot(
        self, val: Optional[Union[Tensor, Sequence[Tensor]]] = None, ax: Optional[_AX_TYPE] = None
    ) -> _PLOT_OUT_TYPE:
        return self._plot(val, ax)


def _mean_absolute_percentage_error_update(
    preds: Tensor,
    target: Tensor,
    epsilon: float = 1.17e-06,
) -> Tuple[Tensor, int]:
    _check_same_shape(preds, target)

    abs_diff = torch.abs(preds - target)
    abs_per_error = abs_diff / torch.clamp(torch.abs(target), min=epsilon)

    sum_abs_per_error = torch.sum(abs_per_error)

    num_obs = target.numel()

    return sum_abs_per_error, num_obs


def _mean_absolute_percentage_error_compute(sum_abs_per_error: Tensor, num_obs: Union[int, Tensor]) -> Tensor:
    return sum_abs_per_error / num_obs


def mean_absolute_percentage_error(preds: Tensor, target: Tensor) -> Tensor:
    sum_abs_per_error, num_obs = _mean_absolute_percentage_error_update(preds, target)
    return _mean_absolute_percentage_error_compute(sum_abs_per_error, num_obs)
