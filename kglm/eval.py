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

    def __init__(self, name: str = '', verbose: bool = False):
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


# class Perplexity(PreComputedMetric):
#
#     def __init__(self, *args, **kwargs):
#         super().__init__(*args, **kwargs)
#         self.name = 'log_perplexity'
#
#
# class PenalizedPerplexity(PreComputedMetric):
#
#     def __init__(self, *args, **kwargs):
#         super().__init__(*args, **kwargs)
#         self.name = 'penalized_log_perplexity'


class Evaluator:
    """
        Actual metric computation is done within the model.
        TODO: move those things here as well later?

        This just runs the eval dataset over the model and asks for the metrics at the end.
        And stores them across epochs
    """

    def __init__(
            self,
            model: torch.nn.Module,
            predict_fn: Callable = None,
            data_loader_callable: Callable = None,
            device: Union[str, torch.device] = None,
            ):
        """
        TODO: this is a work in progress. keep fleshing it out

        If you provide a predict function, and dataset, we can run a whole set of metrics on it by calling <>.run()
        If not, you can still just update the metrics by providing <>.update(model.get_metrics())
            where instance and outputs are dicts as in the loop.
        """

        self._device = device
        self._predict_fn = predict_fn
        self._data_loader_callable = data_loader_callable
        self._model = model

        # Metrics val contains metric's classes. We need to initialize them to get objects
        self.metrics = {}
        # for metric_cls in metric_classes:
        #     metric_obj = metric_cls()
        #     self.metrics[metric_obj.name] = metric_obj

    def report(self):
        """ Return the last values in self.metrics """
        return self.get_last(self.metrics)

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

        self.aggregate_reports(self.metrics, self._model.get_metrics(reset=True))
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

    @staticmethod
    def get_last(aggregate: Dict[str, list]) -> Dict[str, Union[float, int]]:
        """
            From a dict of lists, get the last item corresponding to every dict.
            E.g.
                {
                    'acc': [0.2, 0.5],
                    'p':[0.8, 0.9]
                } -> {'acc': 0.5, 'p': 0.9}
        """
        return {k: v[-1] for k, v in aggregate.items()}
