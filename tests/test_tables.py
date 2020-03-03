from unittest.mock import MagicMock, call
from contextlib import contextmanager

import pytest
import torch

from featurevis import tables


@contextmanager
def does_not_raise():
    yield


class TestTrainedEnsembleModelTemplate:
    @pytest.fixture
    def trained_ensemble_model_template(self, dataset_table, trained_model_table, insert1, insert):
        trained_ensemble_model_template = tables.TrainedEnsembleModelTemplate
        trained_ensemble_model_template.dataset_table = dataset_table
        trained_ensemble_model_template.trained_model_table = trained_model_table
        trained_ensemble_model_template.insert1 = insert1
        trained_ensemble_model_template.Member.insert = insert
        return trained_ensemble_model_template

    @pytest.fixture
    def dataset_table(self):
        dataset_table = MagicMock()
        dataset_table.return_value.__and__.return_value.__len__.return_value = 1
        dataset_table.return_value.proj.return_value.__and__.return_value.fetch1.return_value = dict(ds=0)
        return dataset_table

    @pytest.fixture
    def trained_model_table(self, model):
        trained_model_table = MagicMock()
        trained_model_table.return_value.proj.return_value.__and__.return_value.fetch.return_value = [
            dict(m=0),
            dict(m=1),
        ]
        trained_model_table.return_value.__and__.return_value.fetch.return_value = [dict(m=0, a=0), dict(m=1, a=1)]
        trained_model_table.return_value.load_model = MagicMock(
            side_effect=[("dataloaders1", model), ("dataloaders2", model)]
        )
        return trained_model_table

    @pytest.fixture
    def model(self):
        return MagicMock(side_effect=[torch.tensor([4.0, 7.0]), torch.tensor([6.0, 8.0])])

    @pytest.fixture
    def insert1(self):
        return MagicMock()

    @pytest.fixture
    def insert(self):
        return MagicMock()

    @pytest.mark.parametrize(
        "n_datasets,expectation",
        [(0, pytest.raises(ValueError)), (1, does_not_raise()), (2, pytest.raises(ValueError))],
    )
    def test_if_key_correctness_is_checked(
        self, trained_ensemble_model_template, dataset_table, n_datasets, expectation
    ):
        dataset_table.return_value.__and__.return_value.__len__.return_value = n_datasets
        with expectation:
            trained_ensemble_model_template().create_ensemble("key")

    def test_if_dataset_key_is_correctly_fetched(self, trained_ensemble_model_template, dataset_table):
        trained_ensemble_model_template().create_ensemble("key")
        dataset_table.return_value.proj.return_value.__and__.assert_called_once_with("key")
        dataset_table.return_value.proj.return_value.__and__.return_value.fetch1.assert_called_once_with()

    def test_if_primary_model_keys_are_correctly_fetched(self, trained_ensemble_model_template, trained_model_table):
        trained_ensemble_model_template().create_ensemble("key")
        trained_model_table.return_value.proj.return_value.__and__.assert_called_once_with("key")
        trained_model_table.return_value.proj.return_value.__and__.return_value.fetch.assert_called_once_with(
            as_dict=True
        )

    def test_if_ensemble_key_is_correctly_inserted(self, trained_ensemble_model_template, insert1):
        trained_ensemble_model_template().create_ensemble("key")
        insert1.assert_called_once_with(dict(ds=0, ensemble_hash="536072017a2a3501ea8f09fffa51ee61"))

    def test_if_member_models_are_correctly_inserted(self, trained_ensemble_model_template, insert):
        trained_ensemble_model_template().create_ensemble("key")
        insert.assert_called_once_with(
            [
                dict(ds=0, ensemble_hash="536072017a2a3501ea8f09fffa51ee61", m=0),
                dict(ds=0, ensemble_hash="536072017a2a3501ea8f09fffa51ee61", m=1),
            ]
        )

    def test_if_model_keys_are_correctly_fetched(self, trained_ensemble_model_template, trained_model_table):
        trained_ensemble_model_template().load_model("key")
        trained_model_table.return_value.__and__.assert_called_once_with("key")
        trained_model_table.return_value.__and__.return_value.fetch.assert_called_once_with(as_dict=True)

    def test_if_models_are_correctly_loaded(self, trained_ensemble_model_template, trained_model_table):
        trained_ensemble_model_template().load_model("key")
        trained_model_table.return_value.load_model.assert_has_calls(
            [call(key=dict(m=0, a=0)), call(key=dict(m=1, a=1))]
        )

    def test_if_models_are_switched_to_eval_mode(self, trained_ensemble_model_template, model):
        trained_ensemble_model_template().load_model("key")
        model.eval.assert_has_calls([call(), call()])

    def test_if_ensemble_model_averaging_is_correct(self, trained_ensemble_model_template):
        _, model = trained_ensemble_model_template().load_model("key")
        assert torch.allclose(model("x"), torch.tensor([5.0, 7.5]))

    def test_if_only_first_dataloader_is_returned(self, trained_ensemble_model_template):
        dataloaders, _ = trained_ensemble_model_template().load_model("key")
        assert dataloaders == "dataloaders1"


class TestCSRFV1SelectorTemplate:
    @pytest.fixture
    def selector_template(self, dataset_table, insert, magic_and):
        selector_template = tables.CSRFV1SelectorTemplate
        selector_template.dataset_table = dataset_table
        selector_template.dataset_fn = "dataset_fn"
        selector_template.insert = insert
        selector_template.__and__ = magic_and
        return selector_template

    @pytest.fixture
    def dataset_table(self):
        dataset_table = MagicMock()
        dataset_table.return_value.__and__.return_value.fetch1.return_value = "dataset_config"
        return dataset_table

    @pytest.fixture
    def insert(self):
        return MagicMock()

    @pytest.fixture
    def magic_and(self):
        magic_and = MagicMock()
        magic_and.return_value.fetch1.return_value = "neuron_pos", "session_id"
        return magic_and

    @pytest.fixture
    def get_mappings(self):
        return MagicMock(return_value="mappings")

    @pytest.fixture
    def get_output_selected_model(self):
        return MagicMock(return_value="output_selected_model")

    def test_if_key_source_is_correct(self, selector_template, dataset_table):
        dataset_table.return_value.__and__.return_value = "key_source"
        assert selector_template()._key_source == "key_source"
        dataset_table.return_value.__and__.assert_called_once_with(dict(dataset_fn="dataset_fn"))

    def test_if_dataset_config_is_correctly_fetched(self, selector_template, dataset_table, get_mappings):
        selector_template().make("key", get_mappings=get_mappings)
        dataset_table.return_value.__and__.assert_called_once_with("key")
        dataset_table.return_value.__and__.return_value.fetch1.assert_called_once_with("dataset_config")

    def test_if_get_mappings_is_correctly_called(self, selector_template, get_mappings):
        selector_template().make("key", get_mappings=get_mappings)
        get_mappings.assert_called_once_with("dataset_config", "key")

    def test_if_mappings_are_correctly_inserted(self, selector_template, insert, get_mappings):
        selector_template().make("key", get_mappings=get_mappings)
        insert.assert_called_once_with("mappings")

    def test_if_neuron_position_and_session_id_are_correctly_fetched(self, selector_template, magic_and):
        selector_template().get_output_selected_model("model", "key")
        magic_and.assert_called_once_with("key")
        magic_and.return_value.fetch1.assert_called_once_with("neuron_position", "session_id")

    def test_if_get_output_selected_model_is_called_correctly(self, selector_template, get_output_selected_model):
        selector_template().get_output_selected_model(
            "model", "key", get_output_selected_model=get_output_selected_model
        )
        get_output_selected_model.assert_called_once_with("neuron_pos", "session_id", "model")

    def test_if_output_selected_model_is_correctly_returned(self, selector_template, get_output_selected_model):
        output_selected_model = selector_template().get_output_selected_model(
            "model", "key", get_output_selected_model=get_output_selected_model
        )
        assert output_selected_model == "output_selected_model"


class TestMEIMethod:
    @pytest.fixture
    def mei_method(self, insert1, magic_and, import_func):
        mei_method = tables.MEIMethod
        mei_method.insert1 = insert1
        mei_method.__and__ = magic_and
        mei_method.import_func = import_func
        return mei_method

    @pytest.fixture
    def insert1(self):
        return MagicMock()

    @pytest.fixture
    def magic_and(self):
        magic_and = MagicMock()
        magic_and.return_value.fetch1.return_value = "method_fn", "method_config"
        return magic_and

    @pytest.fixture
    def import_func(self, method_fn):
        return MagicMock(return_value=method_fn)

    @pytest.fixture
    def method_fn(self):
        return MagicMock(return_value=("mei", "evaluations"))

    def test_that_method_is_correctly_inserted(self, mei_method, insert1):
        mei_method().add_method("method_fn", "method_config")
        insert1.assert_called_once_with(
            dict(method_fn="method_fn", method_hash="57f270bf813f42465bd9c21a364bdb2b", method_config="method_config")
        )

    def test_that_method_is_correctly_fetched(self, mei_method, magic_and):
        mei_method().generate_mei("dataloader", "model", dict(key="key"))
        magic_and.assert_called_once_with(dict(key="key"))
        magic_and.return_value.fetch1.assert_called_once_with("method_fn", "method_config")

    def test_if_method_function_is_correctly_imported(self, mei_method, import_func):
        mei_method().generate_mei("dataloader", "model", dict(key="key"))
        import_func.assert_called_once_with("method_fn")

    def test_if_method_function_is_correctly_called(self, mei_method, method_fn):
        mei_method().generate_mei("dataloader", "model", dict(key="key"))
        method_fn.assert_called_once_with("dataloader", "model", "method_config")

    def test_if_returned_mei_entity_is_correct(self, mei_method):
        mei_entity = mei_method().generate_mei("dataloader", "model", dict(key="key"))
        assert mei_entity == dict(key="key", evaluations="evaluations", mei="mei")


class TestMEITemplate:
    @pytest.fixture
    def mei_template(self, trained_model_table, selector_table, method_table, insert1, save_func, model_loader_class):
        mei_template = tables.MEITemplate
        mei_template.trained_model_table = trained_model_table
        mei_template.selector_table = selector_table
        mei_template.method_table = method_table
        mei_template.insert1 = insert1
        mei_template.save_func = save_func
        mei_template.model_loader_class = model_loader_class
        temp_dir_func = MagicMock()
        temp_dir_func.return_value.__enter__.return_value = "/temp_dir"
        mei_template.temp_dir_func = temp_dir_func
        return mei_template

    @pytest.fixture
    def trained_model_table(self):
        return MagicMock()

    @pytest.fixture
    def selector_table(self):
        selector_table = MagicMock()
        selector_table.return_value.get_output_selected_model.return_value = "output_selected_model"
        return selector_table

    @pytest.fixture
    def method_table(self):
        method_table = MagicMock()
        mei_entity = MagicMock()
        mei_entity.squeeze.return_value = "mei"
        method_table.return_value.generate_mei.return_value = dict(mei=mei_entity)
        return method_table

    @pytest.fixture
    def insert1(self):
        return MagicMock()

    @pytest.fixture
    def save_func(self):
        return MagicMock()

    @pytest.fixture
    def model_loader_class(self, model_loader):
        return MagicMock(return_value=model_loader)

    @pytest.fixture
    def model_loader(self):
        model_loader = MagicMock()
        model_loader.load.return_value = "dataloaders", "model"
        return model_loader

    def test_if_model_loader_is_correctly_initialized(self, mei_template, trained_model_table, model_loader_class):
        mei_template(cache_size_limit=5)
        model_loader_class.assert_called_once_with(trained_model_table, cache_size_limit=5)

    def test_if_model_is_correctly_loaded(self, mei_template, model_loader):
        mei_template().make("key")
        model_loader.load.assert_called_once_with(key="key")

    def test_if_correct_model_output_is_selected(self, mei_template, selector_table):
        mei_template().make("key")
        selector_table.return_value.get_output_selected_model.assert_called_once_with("model", "key")

    def test_if_mei_is_correctly_generated(self, mei_template, method_table):
        mei_template().make("key")
        method_table.return_value.generate_mei.assert_called_once_with("dataloaders", "output_selected_model", "key")

    def test_if_mei_is_correctly_saved(self, mei_template, save_func):
        mei_template().make("key")
        save_func.assert_called_once_with("mei", "/temp_dir/d41d8cd98f00b204e9800998ecf8427e.pth.tar")

    def test_if_mei_entity_is_correctly_saved(self, mei_template, insert1):
        mei_template().make("key")
        insert1.assert_called_once_with(dict(mei="/temp_dir/d41d8cd98f00b204e9800998ecf8427e.pth.tar"))
