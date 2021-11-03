"""Test memory"""

import unittest

import torch
import torch.nn as nn

from models.mlmemory import read_from_memory
from models.mlmemory import MemoryReader
from models.mlmemory import MemoryWriter
from models.mlmemory import Memory


class TestMemory(unittest.TestCase):

    def test_read_from_memory(self, B=10, M=5, R=20, H=3, K=2):
        """Function for cosine similarity content based reading from memory matrix.

        In the args list, we have the following conventions:
            B: batch size
            M: number of slots in a row of the memory matrix
            R: number of rows in the memory matrix
            H: number of read heads (of the controller or the policy)
            K: top_k if top_k>0
        """
        read_keys = torch.rand((B, H, M))
        read_strengths = torch.randn((B, H))
        mem_state = torch.rand((B, R, M))
        out = read_from_memory(read_keys, read_strengths, mem_state, top_k=K)
        memory_reads, read_weights, read_indices, read_strengths = out

        if K == 0:
            K = R
        assert memory_reads.shape == (B, H, M)
        assert read_weights.shape == (B, H, K)
        assert read_indices.shape == (B, H, K)
        assert read_strengths.shape == (B, H)

    def test_read_from_memory_all(self):
        for B in [1, 5, 10]:
            self.test_read_from_memory(B=B)
        for M in [1, 5, 10]:
            self.test_read_from_memory(M=M)
        for R in [3, 5, 10]:
            self.test_read_from_memory(R=R)
        for H in [1, 5, 10]:
            self.test_read_from_memory(H=H)
        for K in [1, 5, 10]:
            self.test_read_from_memory(K=K)

    def test_memory_reader(self, B=10, M=5, R=20, H=3, K=2):
        input_size = 9
        memory_word_size = M
        num_read_heads = H
        top_k = K
        mr = MemoryReader(
            input_size=input_size, memory_word_size=memory_word_size,
            num_read_heads=num_read_heads, top_k=top_k)

        read_inputs = torch.rand((B, input_size))
        mem_state = torch.rand((B, R, M))

        concatenated_reads, info = mr((read_inputs, mem_state))

        assert concatenated_reads.shape == (B, H*M)
        assert info.keys.shape == (B, H, M)
        assert info.weights.shape == (B, H, K)
        assert info.indices.shape == (B, H, K)
        assert info.strengths.shape == (B, H)

    def test_memory_writer(self, B=10, M=5, R=20):
        mw = MemoryWriter(mem_shape=(R, M))

        z = torch.rand((B, M))
        mem_state = torch.rand((B, R, M))
        inputs = (z, mem_state)
        state = torch.tensor(10)

        new_mem, new_state = mw(inputs, state)
        assert new_mem.shape == (B, R, M)

    def test_memory(self, B=10, R=20, K=2):
        input_size = 9
        memory_word_size = input_size  # TODO: Temp assuming the same
        num_read_heads = 1
        top_k = K
        mem = Memory(
            input_size=input_size,
            batch_size=B, memory_word_size=memory_word_size,
            num_rows_memory=R, num_read_heads=num_read_heads, top_k=top_k)

        seq_len = 10
        inputs = torch.rand((seq_len, B, input_size))

        outputs, _ = mem(inputs, None)

        assert outputs.shape == (seq_len, B, input_size)


if __name__ == '__main__':
    unittest.main()