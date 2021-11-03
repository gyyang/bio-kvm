"""To be finished"""

import unittest

import torch
import torch.nn as nn
from models.module import Module


class TestModule(unittest.TestCase):

    def test_modulewriter(self):
        class TmpNet(Module):
            def __init__(self, input_size, output_size):
                super(TmpNet, self).__init__()
                self.fc = nn.Linear(input_size, output_size)

            def forward(self, x):
                y = self.fc(x)
                if self.analyzing:
                    self.writer['y'].append(y)
                return y

        class TmpParentNet(Module):
            def __init__(self, input_size, output_size):
                super(TmpParentNet, self).__init__()
                self.fc1 = nn.Linear(input_size, output_size)
                self.fc2 = TmpNet(output_size, output_size)

            def forward(self, x):
                y1 = self.fc1(x)
                y2 = self.fc2(y1)
                if self.analyzing:
                    self.writer['y1'].append(y1)
                    self.writer['y2'].append(y2)
                return y2

        with torch.no_grad():
            x = torch.rand((10, 20))
            net = TmpParentNet(20, 15)
            net.analyze()
            y = net(x)

        writer_dict = net.writer_dict()
        print(writer_dict)

if __name__ == '__main__':
    unittest.main()