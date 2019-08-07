import logging
from abc import abstractmethod
import torch
from delira.models.backends.torch.abstract_network import AbstractPyTorchNetwork


class BaseSegmentationTorchNetwork(AbstractPyTorchNetwork):
    def __init__(self, *args, **kwargs):
        """
        Provide a segmentation network skeleton to be subclassed by the
        actual network implementations

        Parameters
        ----------
        model_cls : callable
            class of model
        args :
            positional arguments passed to model_cls
        kwargs :
            keyword arguments passed to model_cls

        """
        super().__init__()

        self._build_model(*args, **kwargs)

    @abstractmethod
    def forward(self, *args, **kwargs) -> dict:
        """
        Feed tensor through network

        Parameters
        ----------
        *args :
            positional arguments
        **kwargs :
            keyword arguments

        Returns
        -------
        dict
            a dict containing all predictions

        """
        raise NotImplementedError

    @abstractmethod
    def _build_model(self, *args, **kwargs) -> None:
        """
        Implements the actual model

        """
        raise NotImplementedError

    @staticmethod
    def closure(model, data_dict: dict, optimizers: dict, criterions={},
                metrics={}, fold=0, **kwargs):
        """
        closure method to do a single backpropagation step

        Parameters
        ----------
        model : :class:`ClassificationNetworkBaseTorch`
            trainable model
        data_dict : dict
            dictionary containing the data
        optimizers : dict
            dictionary of optimizers to optimize model's parameters
        criterions : dict
            dict holding the criterions to calculate errors
            (gradients from different criterions will be accumulated)
        metrics : dict
            dict holding the metrics to calculate
        fold : int
            Current Fold in Crossvalidation (default: 0)
        **kwargs:
            additional keyword arguments

        Returns
        -------
        dict
            Metric values (with same keys as input dict metrics)
        dict
            Loss values (with same keys as input dict criterions)
        list
            Arbitrary number of predictions as torch.Tensor

        Raises
        ------
        AssertionError
            if optimizers or criterions are empty or the optimizers are not
            specified

        """

        assert (optimizers and criterions) or not optimizers, \
            "Criterion dict cannot be emtpy, if optimizers are passed"

        loss_vals = {}
        metric_vals = {}
        total_loss = 0

        # choose suitable context manager:
        if optimizers:
            context_man = torch.enable_grad

        else:
            context_man = torch.no_grad

        with context_man():

            inputs = data_dict.pop("data")
            preds = model(inputs)

            if data_dict:

                for key, crit_fn in criterions.items():
                    _loss_val = crit_fn(preds, *data_dict.values())
                    loss_vals[key] = _loss_val.detach()
                    total_loss += _loss_val

                with torch.no_grad():
                    for key, metric_fn in metrics.items():
                        metric_vals[key] = metric_fn(
                            preds, *data_dict.values())

        if optimizers:
            optimizers['default'].zero_grad()
            # perform loss scaling via apex if half precision is enabled
            with optimizers["default"].scale_loss(total_loss) as scaled_loss:
                scaled_loss.backward()
            optimizers['default'].step()

        else:

            # add prefix "val" in validation mode
            eval_loss_vals, eval_metrics_vals = {}, {}
            for key in loss_vals.keys():
                eval_loss_vals["val_" + str(key)] = loss_vals[key]

            for key in metric_vals:
                eval_metrics_vals["val_" + str(key)] = metric_vals[key]

            loss_vals = eval_loss_vals
            metric_vals = eval_metrics_vals

        for key, val in {**metric_vals, **loss_vals}.items():
            logging.info({"value": {"value": val.item(), "name": key,
                                    "env_appendix": "_%02d" % fold
                                    }})

        return metric_vals, loss_vals, [preds]

    @staticmethod
    def prepare_batch(batch: dict, input_device, output_device):
        """
        Helper Function to prepare Network Inputs and Labels (convert them
        to correct type and shape and push them to correct devices)

        Parameters
        ----------
        batch : dict
            dictionary containing all the data
        input_device : torch.device
            device for network inputs
        output_device : torch.device
            device for network outputs

        Returns
        -------
        dict
            dictionary containing data in correct type and shape and on
            correct device

        """
        data = torch.from_numpy(
            batch.pop("data")).to(input_device).to(torch.float)
        label = torch.from_numpy(
            batch.pop("label")).to(output_device).to(torch.long)
        return {'data': data, 'label': label, **batch}
