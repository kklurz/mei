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

    def __init__(self, mei_type, ref_level=None):
        self.mei_type = mei_type
        self.ref_level = ref_level

    def __call__(self, *shape, model=None):
        """Loads an image as inital guess"""
        if self.mei_type == "MEI":
            image = torch.from_numpy(model.mei)
        elif self.mei_type == "CEI":
            assert self.ref_level is not None, "Reference level 'ref_level' must be given for CEI type"
            image = torch.from_numpy(model.cei[self.ref_level])
        else:
            raise NotImplementedError()

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
