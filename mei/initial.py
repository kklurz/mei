from abc import ABC, abstractmethod
import torch
from torch import Tensor, randn


class InitialGuessCreator(ABC):
    """Implements the interface used to create an initial guess."""

    @abstractmethod
    def __call__(self, *shape, **kwargs) -> Tensor:
        """Creates an initial guess from which to start the MEI optimization process given a shape."""

    def __repr__(self):
        return f"{self.__class__.__qualname__}()"


class ImageLoader(InitialGuessCreator):
    """Used to create an initial guess tensor filled with values distributed according to a normal distribution."""

    def __init__(self, mei_type, ref_level="no_ref_level", l1="no_l1", zero_grad_threshold=None):
        self.mei_type = mei_type
        self.ref_level = ref_level
        self.l1 = l1
        self.zero_grad_threshold = zero_grad_threshold

    def __call__(self, *shape, model=None):
        """Loads an image as inital guess"""
        if self.mei_type == "MEI":
            assert self.ref_level == "no_ref_level" and self.l1 == "no_l1", "ref_level and l1 are not used!"
            image = torch.from_numpy(model.mei)
        elif self.mei_type == "CEI":
            image = torch.from_numpy(model.cei[self.ref_level][self.l1])
        else:
            raise NotImplementedError()

        if self.zero_grad_threshold is not None:
            cut = torch.abs(image).max() * self.zero_grad_threshold
            model.zero_grad_mask = torch.abs(image) < cut

        assert image.shape[1:] == shape[1:], "Loaded image shape does not match parameter 'shape'"
        n_repeats = int(shape[0] / image.shape[0])
        return image.repeat(n_repeats, 1, 1, 1)


class RandomNormal(InitialGuessCreator):
    """Used to create an initial guess tensor filled with values distributed according to a normal distribution."""

    _create_random_tensor = randn

    def __call__(self, *shape, **kwargs):
        """Creates a random initial guess from which to start the MEI optimization process given a shape."""
        return self._create_random_tensor(*shape)


class OneValue(InitialGuessCreator):
    def __init__(self, fill_value=0.01):
        super().__init__()
        self.fill_value = fill_value
    """Used to create an initial guess tensor filled with a single grey value."""

    def __call__(self, *shape, **kwargs):
        """Creates a random initial guess from which to start the MEI optimization process given a shape."""
        return torch.ones(shape) * self.fill_value


class RandomNormalNullChannel(InitialGuessCreator):
    """Used to create an initial guess tensor filled with values distributed according to a normal distribution."""

    _create_random_tensor = randn

    def __init__(self, null_channel, null_value=0):
        self.null_channel = null_channel
        self.null_value = null_value

    def __call__(self, *shape, **kwargs):
        """Creates a random initial guess from which to start the MEI optimization process given a shape."""
        inital = self._create_random_tensor(*shape)
        inital[:, self.null_channel, ...] = self.null_value
        return inital
