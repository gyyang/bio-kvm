"""Test plasticnet"""

import unittest

import torch

from plasticnet import PlasticLinear, ResMLP


class TestModel(unittest.TestCase):

    def test_plasticlayer(self):
        in_features = 10
        out_features = 20
        batch_size = 5

        layer = PlasticLinear(in_features, out_features)

        input = torch.randn(batch_size, in_features)

        layer.reset_hebb()
        output = layer(input)
        layer.update_hebb(pre=input, post=output)

        print('Parameters')
        for name, param in layer.named_parameters():
            print(name, param.shape)

        print('Buffers')
        for name, param in layer.named_buffers():
            print(name, param.shape)

    def test_resmlp(self):
        net = ResMLP(input_size=1, hidden_sizes=[5, 5])
        input = torch.randn(10, 1)
        output = net(input)
        print(output.shape)


if __name__ == '__main__':
    unittest.main()