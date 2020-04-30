from unittest.mock import MagicMock, call
from functools import partial
from typing import Type

import pytest

from featurevis import methods
from featurevis.domain import Input
from featurevis.tracking import Tracker


class TestAscendGradient:
    @pytest.fixture
    def ascend_gradient(
        self,
        dataloaders,
        model,
        config,
        get_dims,
        create_initial_guess,
        input_cls,
        mei_class,
        import_func,
        optimize_func,
        tracker_cls,
    ):
        def _ascend_gradient(
            use_transform=False, use_regularization=False, use_precondition=False, use_postprocessing=False
        ):
            return partial(
                methods.ascend_gradient,
                dataloaders,
                model,
                config(
                    use_transform=use_transform,
                    use_regularization=use_regularization,
                    use_precondition=use_precondition,
                    use_postprocessing=use_postprocessing,
                ),
                42,
                get_dims=get_dims,
                create_initial_guess=create_initial_guess,
                input_cls=input_cls,
                mei_class=mei_class,
                import_func=import_func,
                optimize_func=optimize_func,
                tracker_cls=tracker_cls,
            )

        return _ascend_gradient

    @pytest.fixture
    def dataloaders(self):
        return dict(train="train_dataloaders")

    @pytest.fixture
    def model(self):
        return MagicMock(name="model")

    @pytest.fixture
    def config(self):
        def _config(use_transform=False, use_regularization=False, use_precondition=False, use_postprocessing=False):
            config = dict(
                device="cpu",
                optimizer="optimizer",
                optimizer_kwargs=dict(optimizer_kwarg1=0, optimizer_kwarg2=1),
                stopper="stopper",
                stopper_kwargs=dict(stopper_kwarg1=0, stopper_kwarg2=1),
                objectives=dict(obj1=dict(obj1_kwarg1=0, obj1_kwarg2=1), obj2=dict(obj2_kwarg1=0, obj2_kwarg2=1)),
            )
            if use_transform:
                config = dict(
                    config, transform="transform", transform_kwargs=dict(transform_kwarg1=0, transform_kwarg2=1)
                )
            else:
                config = dict(config, transform=None, transform_kwargs=None)
            if use_regularization:
                config = dict(
                    config,
                    regularization="regularization",
                    regularization_kwargs=dict(regularization_kwarg1=0, regularization_kwarg2=1),
                )
            else:
                config = dict(config, regularization=None, regularization_kwargs=None)
            if use_precondition:
                config = dict(
                    config,
                    precondition="precondition",
                    precondition_kwargs=dict(precondition_kwarg1=0, precondition_kwarg2=1),
                )
            else:
                config = dict(config, precondition=None, precondition_kwargs=None)
            if use_postprocessing:
                config = dict(
                    config,
                    postprocessing="postprocessing",
                    postprocessing_kwargs=dict(postprocessing_kwarg1=0, postprocessing_kwarg2=1),
                )
            else:
                config = dict(config, postprocessing=None, postprocessing_kwargs=None)
            return config

        return _config

    @pytest.fixture
    def get_dims(self):
        return MagicMock(name="get_dims", return_value=dict(dl1=dict(inputs=(10, 5, 15, 15))))

    @pytest.fixture
    def create_initial_guess(self):
        return MagicMock(name="create_initial_guess", return_value="initial_guess")

    @pytest.fixture
    def input_cls(self):
        return MagicMock(name="input_cls", return_value="input_instance", spec=Input)

    @pytest.fixture
    def mei_class(self):
        return MagicMock(name="mei_class", return_value="mei")

    @pytest.fixture
    def import_func(self):
        def _import_func(name, _kwargs):
            return name

        return MagicMock(name="import_func", side_effect=_import_func)

    @pytest.fixture
    def optimize_func(self):
        return MagicMock(name="optimize_func", return_value=("mei", "final_evaluation"))

    @pytest.fixture
    def tracker_cls(self, tracker_instance):
        return MagicMock(name="tracker_cls", spec=Type[Tracker], return_value=tracker_instance)

    @pytest.fixture
    def tracker_instance(self):
        tracker = MagicMock(name="tracker_instance", spec=Tracker)
        tracker.log = "tracker_log"
        return tracker

    @pytest.fixture
    def import_func_calls(self):
        def _import_func_calls(
            use_transform=False, use_regularization=False, use_precondition=False, use_postprocessing=False
        ):
            import_func_calls = [
                call("optimizer", dict(params=["initial_guess"], optimizer_kwarg1=0, optimizer_kwarg2=1)),
                call("stopper", dict(stopper_kwarg1=0, stopper_kwarg2=1)),
                call("obj1", dict(obj1_kwarg1=0, obj1_kwarg2=1)),
                call("obj2", dict(obj2_kwarg1=0, obj2_kwarg2=1)),
            ]
            if use_transform:
                import_func_calls.append(call("transform", dict(transform_kwarg1=0, transform_kwarg2=1)))
            if use_regularization:
                import_func_calls.append(call("regularization", dict(regularization_kwarg1=0, regularization_kwarg2=1)))
            if use_precondition:
                import_func_calls.append(call("precondition", dict(precondition_kwarg1=0, precondition_kwarg2=1)))
            if use_postprocessing:
                import_func_calls.append(call("postprocessing", dict(postprocessing_kwarg1=0, postprocessing_kwarg2=1)))
            return import_func_calls

        return _import_func_calls

    @pytest.fixture
    def mei_class_call(self, model):
        def _mei_class_call(
            use_transform=False, use_regularization=False, use_precondition=False, use_postprocessing=False
        ):
            args = (model, "input_instance", "optimizer")
            kwargs = {}
            if use_transform:
                kwargs["transform"] = "transform"
            if use_regularization:
                kwargs["regularization"] = "regularization"
            if use_precondition:
                kwargs["precondition"] = "precondition"
            if use_postprocessing:
                kwargs["postprocessing"] = "postprocessing"
            return call(*args, **kwargs)

        return _mei_class_call

    def test_if_seed_is_set(self, ascend_gradient):
        set_seed = MagicMock(name="set_seed")
        ascend_gradient(use_transform=True)(set_seed=set_seed)
        set_seed.assert_called_once_with(42)

    def test_model_is_switched_to_eval_mode(self, ascend_gradient, model):
        ascend_gradient(use_transform=True)()
        model.eval.assert_called_once_with()

    def test_if_model_is_switched_to_device(self, ascend_gradient, model):
        ascend_gradient(use_transform=True)()
        model.to.assert_called_once_with("cpu")

    def test_if_get_dims_is_correctly_called(self, ascend_gradient, get_dims):
        ascend_gradient(use_transform=True)()
        get_dims.assert_called_once_with("train_dataloaders")

    def test_if_create_initial_guess_is_correctly_called(self, ascend_gradient, create_initial_guess):
        ascend_gradient(use_transform=True)()
        create_initial_guess.assert_called_once_with(1, 5, 15, 15, device="cpu")

    def test_if_input_class_is_correctly_called(self, ascend_gradient, input_cls):
        ascend_gradient()()
        input_cls.assert_called_once_with("initial_guess")

    @pytest.mark.parametrize("use_transform", [True, False])
    @pytest.mark.parametrize("use_regularization", [True, False])
    @pytest.mark.parametrize("use_precondition", [True, False])
    @pytest.mark.parametrize("use_postprocessing", [True, False])
    def test_if_import_func_is_correctly_called(
        self,
        ascend_gradient,
        import_func,
        import_func_calls,
        use_transform,
        use_regularization,
        use_precondition,
        use_postprocessing,
    ):
        ascend_gradient(
            use_transform=use_transform,
            use_regularization=use_regularization,
            use_precondition=use_precondition,
            use_postprocessing=use_postprocessing,
        )()
        calls = import_func_calls(
            use_transform=use_transform,
            use_regularization=use_regularization,
            use_precondition=use_precondition,
            use_postprocessing=use_postprocessing,
        )
        assert import_func.mock_calls == calls

    def test_if_tracker_is_correctly_called(self, ascend_gradient, tracker_cls):
        ascend_gradient()()
        tracker_cls.assert_called_once_with(obj1="obj1", obj2="obj2")

    @pytest.mark.parametrize("use_transform", [True, False])
    @pytest.mark.parametrize("use_regularization", [True, False])
    @pytest.mark.parametrize("use_precondition", [True, False])
    @pytest.mark.parametrize("use_postprocessing", [True, False])
    def test_if_mei_is_correctly_initialized(
        self,
        ascend_gradient,
        model,
        mei_class,
        mei_class_call,
        use_transform,
        use_regularization,
        use_precondition,
        use_postprocessing,
    ):
        ascend_gradient(
            use_transform=use_transform,
            use_regularization=use_regularization,
            use_precondition=use_precondition,
            use_postprocessing=use_postprocessing,
        )()
        assert mei_class.mock_calls == [
            mei_class_call(
                use_transform=use_transform,
                use_regularization=use_regularization,
                use_precondition=use_precondition,
                use_postprocessing=use_postprocessing,
            )
        ]

    def test_if_optimize_func_is_correctly_called(self, ascend_gradient, optimize_func, tracker_instance):
        ascend_gradient(use_transform=True)()
        optimize_func.assert_called_once_with("mei", "stopper", tracker_instance)

    def test_if_result_is_returned(self, ascend_gradient):
        assert ascend_gradient(use_transform=True)() == ("final_evaluation", "mei", "tracker_log")
