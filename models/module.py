"""My custom module class."""

from collections import defaultdict
from collections import OrderedDict
import numpy as np

import torch
import torch.nn as nn

class Module(nn.Module):
    def __init__(self):
        super(Module, self).__init__()
        self.analyzing = False

    def analyze(self, mode=True):
        r"""Sets the module in analysis mode.

        Args:
            mode (bool): whether to set analysis mode (``Default False``).

        Returns:
            Module: self
        """
        self.analyzing = mode
        if mode:
            self.writer = defaultdict(list)
        for module in self.children():
            try:
                module.analyze(mode)
            except AttributeError:
                pass
        return self

    def _save_to_writer_dict(self, destination, prefix):
        r"""Saves module state to `destination` dictionary, containing a state
        of the module, but not its descendants. This is called on every
        submodule in :meth:`~torch.nn.Module.state_dict`.

        In rare cases, subclasses can achieve class-specific behavior by
        overriding this method with custom logic.

        Arguments:
            destination (dict): a dict where state will be stored
            prefix (str): the prefix for parameters and buffers used in this
                module
        """
        for key, value in self.writer.items():
            new_value = np.array([v.data.cpu().numpy() for v in value])
            destination[prefix + key] = new_value

    def writer_dict(self, destination=None, prefix=''):
        r"""Returns a dictionary containing a whole state of the module.

        Both parameters and persistent buffers (e.g. running averages) are
        included. Keys are corresponding parameter and buffer names.

        Returns:
            dict:
                a dictionary containing a whole state of the module

        Example::

            >>> module.state_dict().keys()
            ['bias', 'weight']

        """
        if destination is None:
            destination = OrderedDict()
            destination._metadata = OrderedDict()
        destination._metadata[prefix[:-1]] = dict(version=self._version)
        self._save_to_writer_dict(destination, prefix)
        for name, module in self._modules.items():
            if module is not None:
                try:
                    module.writer_dict(destination, prefix + name + '.')
                except AttributeError:
                    pass
        return destination

    def load(self, f, map_location=None, **kwargs):
        """Default wrapper of pytorch load.

        Main purpose is to allow this be overridden.
        """
        print('Loading model from ', f)
        state_dict = torch.load(f, map_location=map_location)
        self.load_state_dict(state_dict)

