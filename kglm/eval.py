from tqdm.auto import tqdm
import warnings
import numpy as np
from typing import List, Dict, Union, Callable
from abc import abstractmethod, ABC
import torch


from utils.misc import change_device


class Metric(ABC):
    """
        Expected to hold values for only one epoch and should be reset after every epoch.
    """

    def __init__(self, name: str, verbose: bool = False):
        self._log: List[float] = []
        self.name = name
        self.verbose = verbose

    def aggregate(self):

        if len(self._log) == 0:
            warnings.warn("You called report, without storing anything in the metrics. Double check your loop logic.")
            return 0

        mean = np.array(self._log).mean()
        if self.verbose:
            print(f"{self.name}: {mean:.5f}")

        return mean

    def reset(self):
        self._log = []

    def log(self, scalar):
        if type(scalar) is torch.Tensor:
            val = scalar.cpu().detach().item()
        elif isinstance(scalar, np.generic):
            val = scalar.item()
        elif type(scalar) in [int, float]:
            val = float(scalar)
        else:
            raise TypeError(f"Scalar type is not known: {type(scalar)}")
        self._log.append(val)

    @abstractmethod
    def compute(self, logits, labels):
        ...


class PreComputedMetric(Metric, ABC):
    """ This here computes nothing, just acts as a vessel for whatever metric you define."""
    ...

    def compute(self, logits, labels):
        pass


class Perplexity(PreComputedMetric):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.name = 'log_perplexity'


class PenalizedPerplexity(PreComputedMetric):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.name = 'penalized_log_perplexity'


class Evaluator:

    def __init__(
            self,
            metric_classes: List[Callable],
            predict_fn: Callable = None,
            data_loader_callable: Callable = None,
            device: Union[str, torch.device] = None,
            ):
        """
        TODO: this is a work in progress. keep fleshing it out

        If you provide a predict function, and dataset, we can run a whole set of metrics on it by calling <>.run()
        If not, you can still just update the metrics by providing <>.update(instance, outputs)
            where instance and outputs are dicts as in the loop.
        """

        self._device = device
        self._predict_fn = predict_fn
        self._data_loader_callable = data_loader_callable

        # Metrics val contains metric's classes. We need to initialize them to get objects
        self.metrics = {}
        for metric_cls in metric_classes:
            metric_obj = metric_cls()
            self.metrics[metric_obj.name] = metric_obj

        self.results = {}
        self._computed_results: bool = False

    def reset(self):
        """ Reset all metrics inside """

        for metric in self.metrics.values():
            metric.reset()

        self.results = {}
        self._computed_results: bool = False

    def update(self, instance: dict, outputs: dict):
        """ For now, ignore instance, just take some keys from outputs"""

        # TODO: automate this somehow. for now, its all hardcoded
        known_keys = ['log_perplexity', 'penalized_log_perplexity']
        for key in known_keys:
            val = outputs[key]
            self.metrics[key].log(val)

    def report(self):
        """ Combine the reports of metrics inside """
        if self._computed_results:
            return self.results

        else:
            for metric_nm, metric_obj in self.metrics.items():

                val = metric_obj.aggregate()
                self.results[metric_nm] = val

            self._computed_results = True
            return self.results

    def run(self):

        if self._predict_fn is None or self._data_loader_callable is None:
            raise ValueError("Either the predict_fn or the data_loader_callable is not provided."
                             "Evaluator can not execute the run function without the two.")

        # Make the dataset
        dataset = self._data_loader_callable()

        with torch.no_grad():

            for i, instance in enumerate(tqdm(dataset)):

                instance = change_device(instance, self._device)
                outputs = self._predict_fn(**instance)
                self.update(instance, outputs)

        return self.report()

    @staticmethod
    def aggregate_reports(aggregate, current):
        """ Expect current to have scalars where aggregate may have lists. """

        for metric_name, metric_scalar in current.items():
            if metric_name not in aggregate:
                aggregate[metric_name] = [metric_scalar]
            else:
                aggregate[metric_name].append(metric_scalar)

        return aggregate
