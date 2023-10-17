"""Contains classes and functions related to optimizing an input to a function such that its value is maximized."""

from __future__ import annotations
from typing import Callable, Tuple
import random
import torch
from torch import Tensor
from torch.optim.optimizer import Optimizer
import wandb
import matplotlib.pyplot as plt

from .domain import Input, State
from .stoppers import OptimizationStopper
from .tracking import Tracker

# from .background_helper import bg_gen,bg_wn

# noinspection PyUnusedLocal
def default_transform(mei: Tensor, i_iteration: int) -> Tensor:
    """Default transform used when no transform is provided to MEI."""
    return mei


# noinspection PyUnusedLocal
def default_regularization(mei: Tensor, i_iteration: int) -> Tensor:
    """Default regularization used when no regularization is provided to MEI."""
    return torch.tensor(0.0)


# noinspection PyUnusedLocal
def default_precondition(grad: Tensor, i_iteration: int) -> Tensor:
    """Default preconditioning used when no preconditioning is provided to MEI."""
    return grad


# noinspection PyUnusedLocal
def default_postprocessing(mei: Tensor, i_iteration: int) -> Tensor:
    """Default postprocessing function used when not postprocessing function is provided to MEI."""
    return mei


# noinspection PyUnusedLocal
def default_background(mei: Tensor, i_iteration: int) -> Tensor:
    """Default postprocessing function used when not postprocessing function is provided to MEI."""
    return None


# noinspection PyUnusedLocal
def default_transform(mei: Tensor, i_iteration: int) -> Tensor:
    """Default transform used when no transform is provided to MEI."""
    return mei


import numpy as np


class MEI:
    """Wrapper around the function and the MEI tensor."""

    input_cls = Input
    state_cls = State

    def __init__(
        self,
        func: Callable[[Tensor], Tensor],
        initial: Tensor,
        optimizer: Optimizer,
        transparency,  #: False, then normal MEI without transparencytransform: Callable[[Tensor, int], Tensor] = default_transform,
        inhibitory,  # for ring or surround MEI: if False, then excitatory
        transparency_weight: float = 0.0,
        transform: Callable[[Tensor, int], Tensor] = default_transform,
        regularization: Callable[[Tensor, int], Tensor] = default_regularization,
        precondition: Callable[[Tensor, int], Tensor] = default_precondition,
        postprocessing: Callable[[Tensor, int], Tensor] = default_postprocessing,
        background: Callable[[Tensor, int], Tensor] = default_background,
        use_wandb_every_n_epochs: int = None,
        ref_level: int = None,  # only for VEIs and CEIs
        scale: float = None,  # only for VEIs
        dx: float = None,  # only for VEIs
        variance_optimization: str = None,  # only for VEIs
        pixel_tanh_scale=False,  # Pass the optimized image through a scaled tanh to achieve man and min pixel values
        reference_mei=None,
        potential_well_function=None,
    ):
        """Initializes MEI.

        Args:
            func: A callable that will receive the to be optimized MEI tensor of floats as its only argument and that
                must return a tensor containing a single float.
            initial: A tensor from which the optimization process will start.
            optimizer: A PyTorch-style optimizer class.
            transform: A callable that will receive the current MEI and the index of the current iteration as inputs and
                that must return a transformed version of the current MEI. Optional.
            regularization: A callable that should have the current mei and the index of the current iteration as
                parameters and that should return a regularization term.
            precondition: A callable that should have the gradient of the MEI and the index of the current iteration as
                parameters and that should return a preconditioned gradient.
            postprocessing: A callable that should have the current MEI and the index of the current iteration as
                parameters and that should return a post-processed MEI. The operation performed by this callable on the
                MEI has no influence on its gradient.
        """
        initial = self.input_cls(initial, reference_mei=reference_mei, pixel_tanh_scale=pixel_tanh_scale)
        self.func = func
        self.initial = initial.clone()
        self.optimizer = optimizer
        self.transform = transform
        self.transparency = transparency
        self.transparency_weight = transparency_weight
        self.regularization = regularization
        self.precondition = precondition
        self.postprocessing = postprocessing
        self.i_iteration = 0
        self._current_input = initial
        self._transformed = None
        self.background = background
        self.inhibitory = inhibitory
        self.use_wandb_every_n_epochs = use_wandb_every_n_epochs
        self.ref_level = ref_level
        self.scale = scale
        self.dx = dx
        self.variance_optimization = variance_optimization
        self.pixel_tanh_scale_ = pixel_tanh_scale
        self.reference_mei = reference_mei
        self.potential_well_function = potential_well_function

        print(f"Using a transparency weight of {self.transparency_weight}")

    @property
    def _transformed_input(self) -> Tensor:
        if self._transformed is None:
            self._transformed = self.transform(self._current_input.tensor, self.i_iteration)
        return self._transformed

    def transparentize(self) -> Tensor:
        ch_img, ch_alpha = self._current_input.tensor[:, [0], ...], self._current_input.tensor[:, -1, ...]
        ch_bg = (
            self.background(self._current_input.tensor, self.i_iteration).cuda()[[0], ...] * self.transparency_weight
        )
        transparentized_mei = ch_bg[[0]] * (1.0 - ch_alpha) + ch_img * ch_alpha
        transparentized_mei = torch.cat((transparentized_mei, self._current_input.tensor[:, 1:-1, ...]), dim=1)
        return transparentized_mei

    def mean_alpha_value(self) -> Tensor:
        return torch.mean(self._current_input.tensor[:, -1, ...])

    def evaluate(self) -> Tensor:
        """Evaluates the function on the current MEI."""
        input = self.transparentize().float() if self.transparency else self._transformed_input

        behavior = torch.zeros((input.shape[0], 3)).to(input.device) if self.func.model.members[0].modulator else None
        pupil_center = torch.zeros((input.shape[0], 2)).to(input.device) if self.func.model.members[0].shifter else None

        mean = self.func.predict_mean(input, behavior=behavior, pupil_center=pupil_center)
        variance = self.func.predict_variance(input, behavior=behavior, pupil_center=pupil_center)
        return mean, mean, variance

    def step(self) -> State:
        """Performs an optimization step."""
        state = dict(i_iter=self.i_iteration, input_=self._current_input.cloned_data)
        self.optimizer.zero_grad()
        objective, mean, variance = self.evaluate()
        evaluation = objective * (self.inhibitory != True) + objective * (self.inhibitory == True) * (-1)
        assert not (
            torch.isinf(evaluation).any() or torch.isnan(evaluation).any()
        ), f"nan or inf value encountered, iteration={self.i_iteration}"

        state["evaluation"] = evaluation.item()
        state["transformed_input"] = self._transformed_input.data.cpu().clone()  ### may need to change

        scale_reg_term = 0
        if not self.pixel_tanh_scale_ is False:
            state["pixel_tanh_scale"] = self._current_input.pixel_tanh_scale.item()
            if self.pixel_tanh_scale_.requires_grad:
                power_evaluation = torch.floor((torch.log10(torch.abs(evaluation)))).item()
                power_reg_term = torch.floor((torch.log10(torch.abs(self._current_input.pixel_tanh_scale)))).item()
                scale_reg_term = self._current_input.pixel_tanh_scale * 10 ** (power_evaluation - power_reg_term - 1)
                assert not (
                    torch.isinf(scale_reg_term).any() or torch.isnan(scale_reg_term).any()
                ), f"nan or inf value encountered, iteration={self.i_iteration}"

        if self.transparency:
            mean_alpha_value = self.mean_alpha_value()
            reg_term = self.regularization(mean_alpha_value, self.i_iteration) + scale_reg_term
            (
                (-evaluation + reg_term) * (1 - mean_alpha_value * self.transparency_weight)
            ).backward()  ### add transparency to objective; mean_alpha_value here should be a function?
        else:
            reg_term = self.regularization(self._transformed_input, self.i_iteration) + scale_reg_term

            (-evaluation + reg_term).backward()

        if self._current_input.grad is None:
            raise RuntimeError("Gradient did not reach MEI")

        if hasattr(self.func, "zero_grad_mask"):
            self._current_input.grad[self.func.zero_grad_mask] = 0
        state["reg_term"] = reg_term.item()

        state["grad"] = self._current_input.cloned_grad
        self._current_input.grad = self.precondition(self._current_input.grad, self.i_iteration)
        # update gradient use transparency gradient
        state["preconditioned_grad"] = self._current_input.cloned_grad
        self.optimizer.step()  # current_input already changed here??

        # post process new mei after optimization
        self._current_input.data = self.postprocessing(self._current_input.data, self.i_iteration)
        state["post_processed_input"] = self._current_input.cloned_data
        _, mean, variance = self.evaluate()
        state["mean"] = mean.item()
        state["variance"] = variance.item()

        if self.use_wandb_every_n_epochs is not None and self.i_iteration % self.use_wandb_every_n_epochs == 0:
            fig, ax = plt.subplots(1, 1)
            ax.imshow(self._current_input.cloned_data.squeeze(), cmap="gray")
            ax.axis("off")
            plt.close()

            wandb.log(
                {
                    "mei": wandb.Image(fig),
                    "iteration": self.i_iteration,
                    "evaluation": evaluation.item(),
                }
            )

        self._transformed = None
        self.i_iteration += 1
        return self.state_cls.from_dict(state)

    def angle(self, u, v):
        import numpy as np

        u = u.cpu().data.numpy().reshape(-1)
        v = v.cpu().data.numpy().reshape(-1)

        nominator = (u * v).sum()
        denominator = np.linalg.norm(u) * np.linalg.norm(v)
        return np.degrees(np.arccos(nominator / denominator)).round(1)

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__qualname__}({self.func}, {self.initial}, {self.optimizer}, "
            f"transform={self.transform}, regularization={self.regularization}, precondition={self.precondition}, "
            f"postprocessing={self.postprocessing})"
        )


class CEI(MEI):
    def evaluate(self) -> Tensor:
        """Evaluates the function on the current CEI (Certain exciting input). This class creates an input which excites
        the neuron to a certain percentage (ref_level) of the MEI"""
        _, mean, variance = super().evaluate()

        diff = self.ref_level - mean / self.func.mei_mean
        objective = -(diff**2)
        return objective, mean, variance


class VEI(MEI):
    def exp_potential_well(self, mean):
        x = mean / self.func.mei_mean
        out = torch.exp(-self.scale * (x + self.dx - self.ref_level)) + torch.exp(
            self.scale * (x - self.dx - self.ref_level)
        )
        return out

    def linear_potential_well(self, mean):
        x = mean / self.func.mei_mean
        if self.ref_level - self.dx < x < self.ref_level + self.dx:
            out = 0
        else:
            out = torch.abs(x - self.ref_level) * self.scale
        return out

    def evaluate(self) -> Tensor:
        """Evaluates the function on the current VEI (Variably exciting input)."""
        _, mean, variance = super().evaluate()

        if self.potential_well_function == "exp":
            objective = -self.exp_potential_well(mean)
        elif self.potential_well_function == "linear":
            objective = -self.linear_potential_well(mean)
        else:
            raise ValueError()

        if self.variance_optimization == "max":
            objective += variance / self.func.mei_variance
        elif self.variance_optimization == "min":
            objective -= variance / self.func.mei_variance
        else:
            raise ValueError()
        assert not abs(objective.item()) > 1.e5, "very big objective, iteration: {}".format(self.i_iteration)
        return objective, mean, variance


def optimize(mei: MEI, stopper: OptimizationStopper, tracker: Tracker) -> Tuple[float, Tensor]:
    """Optimizes the input to a given function such that it maximizes said function using gradient ascent.

    Args:
        mei: An instance of the to be optimized MEI.
        stopper: A subclass of "OptimizationStopper" used to stop the optimization process.
        tracker: A tracker object used to track pre-defined objectives during the optimization process. The current
            state of the optimization process is passed to the "track" method of the object in each iteration.

    Returns:
        A float representing the final evaluation and a tensor of floats having the same shape as "initial_guess"
        representing the input that maximizes the function.
    """
    while True:
        current_state = mei.step()
        stop, output = stopper(current_state)
        current_state.stopper_output = output
        tracker.track(current_state)
        if stop:
            break
    if hasattr(stopper, "best_state"):
        print("Restoring best state...")
        current_state = stopper.best_state
    return current_state.evaluation, current_state.post_processed_input, current_state.mean, current_state.variance
