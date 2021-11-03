"""To be finished"""

import unittest

import torch as T

from model import AdaptiveAttnNet, NormalizedSoftmax


class TestModel(unittest.TestCase):

    def test_adaptiveattnnet(self):
        n_input, n_concept, n_output, batch_size = 10, 100, 4, 64

        net = AdaptiveAttnNet(n_input, n_concept, n_output, batch_size)

        X = T.randn(batch_size, n_input, 1)
        Y_target = T.randn(batch_size, n_output, 1)

        Y, h, loss = net(X, Y_target)

        self.assertListEqual(list(Y.shape), [batch_size, n_output, 1])

        print('Parameters')
        for name, param in net.named_parameters():
            print(name, param.shape)

        print('Buffers')
        for name, param in net.named_buffers():
            print(name, param.shape)

        Y, h, loss = net(X, Y_target, plasticity=False)
        Y, h, loss = net(X, plasticity=True)

    def test_normalizedsoftmax(self):
        x = T.randn(100, 10, 50)
        y = NormalizedSoftmax(dim=0)(x)

        x = T.zeros(1, 10, 1)
        y = NormalizedSoftmax(dim=1, constant=20.0)(x)
        # print(y)
        self.assertTrue(y.numpy().sum()<1e-3)


if __name__ == '__main__':
    unittest.main()