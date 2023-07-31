"""This module contains mix-ins for the main tables and table templates."""

from __future__ import annotations
import os
import tempfile
from typing import Callable, Iterable, Mapping, Optional, Tuple, Dict, Any
from string import ascii_letters
from random import choice

import torch
import numpy as np
from torch.nn import Module
from torch.utils.data import DataLoader
from nnfabrik.utility.dj_helpers import make_hash

from . import integration
from . import optimization
from .modules import EnsembleModel, ConstrainedOutputModel


Key = Dict[str, Any]
Dataloaders = Dict[str, DataLoader]


class TrainedEnsembleModelTemplateMixin:
    definition = """
    # contains ensemble ids
    -> self.dataset_table
    ensemble_hash                   : char(32)      # the hash of the ensemble
    ---
    ensemble_comment        = ''    : varchar(256)  # a short comment describing the ensemble
    """

    class Member:
        definition = """
        # contains assignments of trained models to a specific ensemble id
        -> master
        -> master.trained_model_table
        """

        insert: Callable[[Iterable], None]
        __and__: Callable[[Key], TrainedEnsembleModelTemplateMixin.Member]
        fetch: Callable

    dataset_table = None
    trained_model_table = None
    ensemble_model_class = EnsembleModel

    insert1: Callable[[Mapping], None]
    __and__: Callable[[Key], TrainedEnsembleModelTemplateMixin]
    fetch1: Callable

    def create_ensemble(self, key: Key, comment: str = "", skip_duplicates=False) -> None:
        if len(self.dataset_table() & key) != 1:
            raise ValueError("Provided key not sufficient to restrict dataset table to one entry!")
        dataset_key = (self.dataset_table().proj() & key).fetch1()
        models = (self.trained_model_table().proj() & key).fetch(as_dict=True)
        primary_key = dict(dataset_key, ensemble_hash=integration.hash_list_of_dictionaries(models))
        self.insert1(dict(primary_key, ensemble_comment=comment), skip_duplicates=skip_duplicates)
        self.Member().insert([{**primary_key, **m} for m in models], skip_duplicates=skip_duplicates)

    def load_model(
        self,
        key: Optional[Key] = None,
        include_dataloader: Optional[bool] = True,
        include_state_dict: Optional[bool] = True,
    ) -> Tuple[Dataloaders, EnsembleModel]:
        if key is None:
            key = self.fetch1("KEY")
        return self._load_ensemble_model(
            key=key,
            include_dataloader=include_dataloader,
            include_state_dict=include_state_dict,
        )

    def _load_ensemble_model(
        self,
        key: Optional[Key] = None,
        include_dataloader: Optional[bool] = True,
        include_state_dict: Optional[bool] = True,
    ) -> Tuple[Dataloaders, EnsembleModel]:

        ensemble_key = (self & key).fetch1()
        model_keys = (self.Member() & ensemble_key).fetch(as_dict=True)

        if include_dataloader:
            dataloaders, models = tuple(
                list(x)
                for x in zip(
                    *[
                        self.trained_model_table().load_model(
                            key=k,
                            include_dataloader=include_dataloader,
                            include_state_dict=include_state_dict,
                        )
                        for k in model_keys
                    ]
                )
            )
        else:
            models = [
                self.trained_model_table().load_model(
                    key=k,
                    include_dataloader=include_dataloader,
                    include_state_dict=include_state_dict,
                )
                for k in model_keys
            ]

        return (
            (dataloaders[0], self.ensemble_model_class(*models))
            if include_dataloader
            else self.ensemble_model_class(*models)
        )


class CSRFV1ObjectiveTemplateMixin:
    definition = """
    # contains information that can be used to map a neuron's id to its corresponding integer position in the output of
    # the model. 
    -> self.dataset_table
    neuron_id       : smallint unsigned # unique neuron identifier
    ---
    neuron_position : smallint unsigned # integer position of the neuron in the model's output 
    session_id      : varchar(13)       # unique session identifier
    """

    dataset_table = None
    dataset_fn = "csrf_v1"
    constrained_output_model = ConstrainedOutputModel

    insert: Callable[[Iterable], None]
    __and__: Callable[[Mapping], CSRFV1SelectorTemplateMixin]
    fetch1: Callable

    @property
    def _key_source(self):
        return self.dataset_table() & dict(dataset_fn=self.dataset_fn)

    def make(self, key: Key, get_mappings: Callable = integration.get_mappings) -> None:
        dataset_config = (self.dataset_table() & key).fetch1("dataset_config")
        mappings = get_mappings(dataset_config, key)
        self.insert(mappings)

    def get_output_selected_model(self, model: Module, key: Key) -> constrained_output_model:
        neuron_pos, session_id = (self & key).fetch1("neuron_position", "session_id")
        return self.constrained_output_model(model, neuron_pos, forward_kwargs=dict(data_key=session_id))


class MEIMethodMixin:
    definition = """
    # contains methods for generating MEIs and their configurations.
    method_fn                           : varchar(64)   # name of the method function
    method_hash                         : varchar(32)   # hash of the method config
    ---
    method_config                       : longblob      # method configuration object
    method_ts       = CURRENT_TIMESTAMP : timestamp     # UTZ timestamp at time of insertion
    method_comment                      : varchar(256)  # a short comment describing the method
    """

    insert1: Callable[[Mapping], None]
    __and__: Callable[[Mapping], MEIMethodMixin]
    fetch1: Callable

    seed_table = None
    import_func = staticmethod(integration.import_module)
    optional_names = optional_names = (
        "initial",
        "transform",
        "regularization",
        "precondition",
        "postprocessing",
    )

    def add_method(self, method_fn: str, method_config: Mapping, comment: str = "") -> None:
        self.insert1(
            dict(
                method_fn=method_fn,
                method_hash=make_hash(method_config),
                method_config=method_config,
                method_comment=comment,
            )
        )

    def generate_mei(self, dataloaders: Dataloaders, model: Module, key: Key, seed: int) -> Dict[str, Any]:
        method_fn, method_config = (self & key).fetch1("method_fn", "method_config")
        method_fn = self.import_func(method_fn)
        self.insert_key_in_ops(method_config=method_config, key=key)
        mei_class_name = method_config.pop("mei_class_name", "MEI")
        if mei_class_name == "MEI":
            mei_class = optimization.MEI
        elif mei_class_name == "VEI":
            mei_class = optimization.VEI
        elif mei_class_name == "CEI":
            mei_class = optimization.CEI
        else:
            raise ValueError(f"mei_class_name '{mei_class_name}' not recognized")
        mei, score, output, mean, variance = method_fn(dataloaders, model, method_config, seed, mei_class=mei_class)
        return dict(key, mei=mei, score=score, output=output, mean=mean, variance=variance)

    def generate_ringmei(
        self, dataloaders: Dataloaders, model: Module, key: Key, seed: int, ring_mask: Tensor
    ) -> Dict[str, Any]:
        method_fn, method_config = (self & key).fetch1("method_fn", "method_config")
        method_fn = self.import_func(method_fn)
        mei, score, output = method_fn(dataloaders, model, method_config, seed, ring_mask)
        return dict(key, mei=mei, score=score, output=output)

    def insert_key_in_ops(self, method_config, key):
        for k, v in method_config.items():
            if k in self.optional_names:
                if "key" in v.get("kwargs", ""):
                    v["kwargs"]["key"] = key


class MEISeedMixin:
    definition = """
    # contains seeds used to make the MEI generation process reproducible
    mei_seed    : int   # MEI seed
    """


class MEITemplateMixin:
    definition = """
    # contains maximally exciting images (MEIs)
    -> self.method_table
    -> self.trained_model_table
    -> self.selector_table
    -> self.seed_table
    ---
    mei                 : attach@minio  # the MEI as a tensor
    score               : float         # some score depending on the used method function
    mean                : float         # mean activation
    variance            : float         # variance of activation
    output              : attach@minio  # object returned by the method function
    """

    trained_model_table = None
    selector_table = None
    method_table = None
    seed_table = None
    model_loader_class = integration.ModelLoader
    save = staticmethod(torch.save)
    get_temp_dir = tempfile.TemporaryDirectory

    insert1: Callable[[Mapping], None]

    def __init__(self, *args, cache_size_limit: int = 10, **kwargs):
        super().__init__(*args, **kwargs)
        self.model_loader = self.model_loader_class(self.trained_model_table, cache_size_limit=cache_size_limit)

    def make(self, key: Key, return_before_inserting=False) -> None:
        dataloaders, model = self.model_loader.load(key=key)
        seed = (self.seed_table() & key).fetch1("mei_seed")
        output_selected_model = self.selector_table().get_output_selected_model(model, key)
        self.add_params_to_model(output_selected_model, key)
        mei_entity = self.method_table().generate_mei(dataloaders, output_selected_model, key, seed)
        if return_before_inserting:
            return mei_entity
        self._insert_mei(mei_entity)

    def add_params_to_model(self, model, key):

        # Find other existing MEIs/CEIs for the current key
        new_key = {k: v for k, v in key.items() if k not in ["method_fn", "method_hash"]}
        table = self.method_table * self & new_key

        if len(table) != 0:
            method_fns, method_hashs, method_configs, means, variances, mei_paths = table.fetch(
                "method_fn", "method_hash", "method_config", "mean", "variance", "mei"
            )
            meis = []
            for mei_path in mei_paths:
                meis.append(torch.load(mei_path))
                os.remove(mei_path)
            meis = np.stack(meis)

            # Find which indices are for MEIs and CEIs
            idx_mei = np.where([config.get("mei_class_name", "MEI") == "MEI" for config in method_configs])[0]
            idx_cei = np.where([config.get("mei_class_name", "MEI") == "CEI" for config in method_configs])[0]

            # If there are MEI entries: add mean, variance and MEI of the max MEI to the model
            if len(idx_mei) != 0:
                max_mei_idx = np.argmax(means[idx_mei])
                model.mei_mean = means[idx_mei][max_mei_idx]
                model.mei_variance = variances[idx_mei][max_mei_idx]
                model.mei = meis[idx_mei][max_mei_idx]

            # If there are CEI entries: add all CEIs to the model with their respective rev_level
            if len(idx_cei) != 0:
                model.cei = {}
                for cei, cei_method_config in zip(meis[idx_cei], method_configs[idx_cei]):
                    model.cei[cei_method_config["ref_level"]] = cei
            print("Finished adding params to model!")

    def _insert_mei(self, mei_entity: Dict[str, Any]) -> None:
        """Saves the MEI to a temporary directory and inserts the prepared entity into the table."""
        with self.get_temp_dir() as temp_dir:
            for name in ("mei", "output"):
                self._save_to_disk(mei_entity, temp_dir, name)
            self.insert1(mei_entity, ignore_extra_fields=True)

    def _save_to_disk(self, mei_entity: Dict[str, Any], temp_dir: str, name: str) -> None:
        data = mei_entity.pop(name)
        filename = name + "_" + self._create_random_filename() + ".pth.tar"
        filepath = os.path.join(temp_dir, filename)
        self.save(data, filepath)
        mei_entity[name] = filepath

    @staticmethod
    def _create_random_filename(length: Optional[int] = 32) -> str:
        return "".join(choice(ascii_letters) for _ in range(length))
