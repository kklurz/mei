"""Contains callable classes used to stop the MEI optimization process once it has reached an acceptable result."""

from __future__ import annotations
from abc import ABC, abstractmethod
from typing import Tuple, Optional, Any
from copy import deepcopy

from .domain import State


class OptimizationStopper(ABC):
    """Implements the interface used to stop the MEI optimization process once it has reached an acceptable result."""

    @abstractmethod
    def __call__(self, current_state: State) -> Tuple[bool, Optional[Any]]:
        """Should return "True" if the MEI optimization process has reached an acceptable result."""

    def __repr__(self):
        return f"{self.__class__.__qualname__}({self.num_iterations})"


class NumIterations(OptimizationStopper):
    """Callable that stops the optimization process after a specified number of steps."""

    def __init__(self, num_iterations):
        """Initializes NumIterations.

        Args:
            num_iterations: The number of optimization steps before the process is stopped.
        """
        self.num_iterations = num_iterations

    def __call__(self, current_state: State) -> Tuple[bool, Optional[Any]]:
        """Stops the optimization process after a set number of steps by returning True."""
        if current_state.i_iter == self.num_iterations:
            return True, None
        return False, None


class EarlyStopping(OptimizationStopper):
    """Callable that stops the optimization process after a specified number of steps."""

    def __init__(self, patience, min_iter=100, max_iter=2000):
        """Initializes NumIterations.

        Args:
            num_iterations: The number of optimization steps before the process is stopped.
        """
        assert min_iter < max_iter, "min_iter must be smaller than max_iter"
        self.patience = patience
        self.min_iter = min_iter
        self.max_iter = max_iter
        self.evaluation_value = -1.e12
        self.patience_iter = 0
        self.i_iter = 0

    def __call__(self, current_state: State) -> Tuple[bool, Optional[Any]]:
        """Stops the optimization process after a set number of steps by returning True."""
        if self.evaluation_value > current_state.evaluation:
            self.patience_iter += 1
        else:
            self.patience_iter = 0
            self.evaluation_value = current_state.evaluation
            self.best_state = deepcopy(current_state)
        self.i_iter += 1

        if (self.patience_iter >= self.patience and self.i_iter >= self.min_iter) or self.i_iter == self.max_iter:
            print(f"Early stopping finished after {current_state.i_iter} iterations.")
            return True, None
        return False, None
