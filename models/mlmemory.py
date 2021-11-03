"""Machine learning style memory network.

Adopted from the memory network in Tensorflow in
Optimizing agent behavior over long time scales by transporting value
https://www.nature.com/articles/s41467-019-13073-w
https://github.com/deepmind/deepmind-research/blob/master/tvt/memory.py
"""
import collections
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

ReadInformation = collections.namedtuple(
    'ReadInformation', ('weights', 'indices', 'keys', 'strengths'))


class MemoryWriter(nn.Module):
    """Memory writer module."""

    def __init__(self, mem_shape):
        super().__init__()
        self._mem_shape = mem_shape  # (R, M)

    def forward(self, inputs, state):
        """Inserts z into the argmin row of usage markers and updates all rows.
        Returns an operation that, when executed, correctly updates the internal
        state and usage markers.
        Args:
          inputs: A tuple consisting of:
              * z, the value to write at this timestep
              * mem_state, the state of the memory at this timestep before writing
          state: The state is just the write_counter.
        Returns:
          A tuple of the new memory state and a tuple containing the next state.
        """
        z, mem_state = inputs  # z: (B, M), mem_state: (B, R, M)

        # Stop gradient on writes to memory
        z = z.detach()

        prev_write_counter = state  # int
        new_row_value = z  # (B, M)

        # Find the index to insert the next row into.
        num_mem_rows = self._mem_shape[0]  # R
        write_index = prev_write_counter.to(torch.int64) % num_mem_rows ##writes sequentially to next row
        one_hot_row = F.one_hot(write_index, num_classes=num_mem_rows) ##used to select which row to zero out; 1xR because write_index has dim 1
        write_counter = prev_write_counter + 1 ##next write will write to next row

        # Insert state variable to new row.
        # First you need to size it up to the full size.
        insert_new_row = lambda mem, o_hot, z: mem - (o_hot * mem) + (o_hot * z) ##replaces the write_index'th row of mem with z
        # new_mem: (B, R, M)
        new_mem = insert_new_row(mem_state,
                                 torch.unsqueeze(one_hot_row, dim=-1),
                                 torch.unsqueeze(new_row_value, dim=-2))

        new_state = write_counter  # int
        return new_mem, new_state


class MemoryReader(nn.Module):
    """Memory Reader Module."""

    def __init__(self,
                 input_size,
                 memory_word_size,
                 num_read_heads,
                 top_k=0,
                 memory_size=None, keys_and_read_strengths_mode='identity_keys_and_read_strengths'):
        """Initializes the `MemoryReader`.

        Args:
            input_size: Input size.
            memory_word_size: The dimension of the 1-D read keys this memory
                reader should produce. Each row of the memory is of length
                `memory_word_size`.
            num_read_heads: The number of reads to perform.
            top_k: Softmax and summation when reading is only over top k most
                similar entries in memory. top_k=0 (default) means dense
                reads, i.e. no top_k.
            memory_size: Number of rows in memory.
        """
        print("TVT MemoryReader keys_and_read_strengths_mode = ", keys_and_read_strengths_mode)
        super().__init__()
        ##input size = memory_word_size
        self._memory_word_size = memory_word_size ## M, or W in the paper
        self._num_read_heads = num_read_heads ## number of read heads = number of read keys, k in paper
        self._top_k = top_k

        # This is not an RNNCore but it is useful to expose the output size.
        self._output_size = num_read_heads * memory_word_size

        num_read_weights = top_k if top_k > 0 else memory_size
        self._read_info_size = ReadInformation(
            weights=[num_read_heads, num_read_weights],
            indices=[num_read_heads, num_read_weights],
            keys=[num_read_heads, memory_word_size],
            strengths=[num_read_heads],
        )

        # Transforms to value-based read for each read head.
        output_dim = (memory_word_size + 1) * num_read_heads ## output_dim size = (M + 1)xH

        self._keys_and_read_strengths_mode = keys_and_read_strengths_mode
        self._keys_and_read_strengths_generator = nn.Identity()
        self._keys_generator = nn.Identity()
        if self._keys_and_read_strengths_mode == 'identity_keys_and_read_strengths' or self._keys_and_read_strengths_mode =='identity_keys_and_read_strengths_sharpened':
            pass
        elif self._keys_and_read_strengths_mode == 'separate_linear_keys_and_read_strengths':
            self._keys_generator = nn.Linear(input_size, memory_word_size)
            self._read_strengths_generator = nn.Linear(input_size, num_read_heads)
        elif self._keys_and_read_strengths_mode == 'single_linear_keys_and_read_strengths' or 'linear_with_softplus_orig_TVT': ##this is used in original TVT
            self._keys_and_read_strengths_generator = nn.Linear(input_size, output_dim)
        elif self._keys_and_read_strengths_mode == 'identity_keys_and_scalar_read_strength':
            self._scalar_read_strength = nn.Parameter(torch.tensor(1.0, requires_grad=True))
        elif self._keys_and_read_strengths_mode == 'linear_keys_and_scalar_read_strength':
            self._keys_generator = nn.Linear(input_size, memory_word_size)
            self._scalar_read_strength = nn.Parameter(torch.tensor(1.0, requires_grad=True))

    def forward(self, inputs):
        """Looks up rows in memory.

        In the args list, we have the following conventions:
            B: batch size
            M: number of slots in a row of the memory matrix (memory_word_size)
            R: number of rows in the memory matrix (num_rows_memory)
            H: number of read heads in the memory controller

        Args:
            inputs: A tuple of
                *  read_inputs, a tensor of shape [B, ...] that will be
                    flattened and passed through a linear layer to get read
                    keys/read_strengths for each head.
                *  mem_state, the primary memory tensor. Of shape [B, R, M].
        Returns:
            The read from the memory (concatenated across read heads) and read
            information.
        """
        # Assert input shapes are compatible and separate inputs.
        _assert_compatible_memory_reader_input(inputs)
        read_inputs, mem_state = inputs ##read_inputs shape: [B, M]

        # Determine the read weightings for each key.
        flat_outputs = self._keys_and_read_strengths_generator( ##never used for: identity_keys_and_read_strengths, identity_keys_and_read_strengths_sharpened
            read_inputs.flatten(
                start_dim=1))  ## flat_outputs shape: [B, ((M+1)xH)], arranged such that last H cols are read strengths
        h = self._num_read_heads  ##k in paper
        read_strengths = F.softplus(flat_outputs[:, -h:])  # a scalar per head
        flat_keys = flat_outputs[:, :-h] ## [B, MH] (keys concatenated horizontally)
        # Separate the read_strengths from the rest of the weightings.
        #flat_keys = flat_outputs[:, :-h] ## [B, MH] (keys concatenated horizontally)
        ##flat_keys = flat_outputs if using Identity for keys_and_read_strengths_generator
        ##flat_keys = flat_outputs#[:, :-h] if using Linear layer for keys_and_read_strengths_generator
        if self._keys_and_read_strengths_mode == 'identity_keys_and_read_strengths': ##doesn't (and shouldn't) work
            read_strengths = torch.ones_like(read_strengths)
            flat_keys = read_inputs.flatten(start_dim=1)
        elif self._keys_and_read_strengths_mode == 'identity_keys_and_read_strengths_sharpened': ##works
            read_strengths = 10*torch.ones_like(read_strengths)
            flat_keys = read_inputs.flatten(start_dim=1)
        elif self._keys_and_read_strengths_mode == 'separate_linear_keys_and_read_strengths': ##works
            flat_keys = self._keys_generator(read_inputs.flatten(start_dim=1))
            read_strengths = self._read_strengths_generator(read_inputs.flatten(start_dim=1))
        elif self._keys_and_read_strengths_mode == 'single_linear_keys_and_read_strengths': ##works
            flat_outputs = self._keys_and_read_strengths_generator(read_inputs.flatten(start_dim=1))
            flat_keys = flat_outputs[:, :-h]
            read_strengths = flat_outputs[:, -h:]
        elif self._keys_and_read_strengths_mode == 'identity_keys_and_scalar_read_strength': ##works
            flat_keys = read_inputs.flatten(start_dim=1)
            read_strengths = torch.ones_like(read_strengths) * self._scalar_read_strength
        elif self._keys_and_read_strengths_mode == 'linear_keys_and_scalar_read_strength': ##works
            flat_keys = self._keys_generator(read_inputs.flatten(start_dim=1))
            read_strengths = torch.ones_like(read_strengths) * self._scalar_read_strength
        elif self._keys_and_read_strengths_mode == 'single_linear_keys_and_read_strengths': ##works
            flat_outputs = self._keys_and_read_strengths_generator(read_inputs.flatten(start_dim=1))
            flat_keys = flat_outputs[:, :-h]
            read_strengths = flat_outputs[:, -h:]
        elif self._keys_and_read_strengths_mode == 'linear_with_softplus_orig_TVT':
            flat_outputs = self._keys_and_read_strengths_generator(read_inputs.flatten(start_dim=1))  ## flat_outputs shape: [B, ((M+1)xH)], arranged such that last H cols are read strengths
            read_strengths = F.softplus(flat_outputs[:, -h:])  # a scalar per head
            flat_keys = flat_outputs[:, :-h]

        # Reshape the weights.
        read_shape = (self._num_read_heads, self._memory_word_size)
        read_keys = flat_keys.unflatten(1, read_shape) ##[B, H, M]

        # Read from memory.
        memory_reads, read_weights, read_indices, read_strengths = (
            read_from_memory(read_keys, read_strengths, mem_state, self._top_k))
        concatenated_reads = memory_reads.flatten(start_dim=1)  # [B, H * M]

        return concatenated_reads, ReadInformation(
            weights=read_weights,
            indices=read_indices,
            keys=read_keys,
            strengths=read_strengths)


def batch_gather(input, index):
    """Similar to tf.compat.v1.batch_gather.

    Args:
        input: (Batch, Dim1, Dim2)
        index: (Batch, N_ind)

    Returns:
        output: (Batch, N_ind, Dim2)
            output[i, j, k] = input[i, index[i, j], k]
    """
    index_exp = index.unsqueeze(2).expand(-1, -1, input.size(2))
    out = torch.gather(input, 1, index_exp)  # (Batch, N_ind, Dim2)
    return out


def read_from_memory(read_keys, read_strengths, mem_state, top_k):
    """Function for cosine similarity content based reading from memory matrix.

    In the args list, we have the following conventions:
        B: batch size
        M: number of slots in a row of the memory matrix
        R: number of rows in the memory matrix
        H: number of read heads (of the controller or the policy)
        K: top_k if top_k>0
    Args:
        read_keys: the read keys of shape [B, H, M].
        read_strengths: the coefficients used to compute the normalized
            weighting vector of shape [B, H].
        mem_state: the primary memory tensor. Of shape [B, R, M].
        top_k: only use top k read matches, other reads do not go into softmax
            and are zeroed out in the output. top_k=0 (default) means use
            dense reads.
    Returns:
        The memory reads [B, H, M], read weights [B, H, top k], read indices
            [B, H, top k], and read strengths [B, H].
    """
    _assert_compatible_read_from_memory_inputs(read_keys, read_strengths,
                                               mem_state)
    batch_size = read_keys.shape[0]
    num_read_heads = read_keys.shape[1]

    # Scale such that all rows are L2-unit vectors, for memory and read query.
    scaled_read_keys = F.normalize(read_keys, dim=-1, p=2)  # [B, H, M]##each key is an L2-unit vector
    scaled_mem = F.normalize(mem_state, dim=-1, p=2)  # [B, R, M] ##each row in memory is an L2-unit vector

    # The cosine distance is then their dot product.
    # Find the cosine distance between each read head and each row of memory.

    ##[B,H,M] x [B,M,R]
    cosine_distances = torch.matmul(
        scaled_read_keys, scaled_mem.transpose(1, 2))  # [B, H, R] ##similarity of each key with each mem item

    # The rank must match cosine_distances for broadcasting to work.
    read_strengths = torch.unsqueeze(read_strengths, dim=-1)  # [B, H, 1] ## becomes [B,H,R] when broadcasted
    weighted_distances = read_strengths * cosine_distances  # [B, H, R]

    if top_k:
        # Get top k indices (row indices with top k largest weighted distances).
        # sorted=False seems to have no impact in pytorch=1.7.1
        top_k_output = torch.topk(weighted_distances, top_k, sorted=False)
        read_indices = top_k_output[1]  # [B, H, K]

        # Create a sub-memory for each read head with only the top k rows.
        # Each batch_gather is [B, K, M] and the list stacks to [B, H, K, M].
        topk_mem_per_head = [batch_gather(mem_state, read_indices[:, h, :])
                             for h in range(num_read_heads)]
        topk_mem = torch.stack(topk_mem_per_head, dim=1)  # [B, H, K, M]
        topk_scaled_mem = F.normalize(topk_mem, dim=-1, p=2)  # [B, H, K, M]

        # Calculate read weights for each head's top k sub-memory.
        expanded_scaled_read_keys = torch.unsqueeze(
            scaled_read_keys, dim=2)  # [B, H, 1, M]
        topk_cosine_distances = torch.sum(
            expanded_scaled_read_keys * topk_scaled_mem, dim=-1)  # [B, H, K]
        topk_weighted_distances = (
                read_strengths * topk_cosine_distances)  # [B, H, K]
        read_weights = F.softmax(topk_weighted_distances, dim=-1)  # [B, H, K]

        # For each head, read using the sub-memories and corresponding weights.
        expanded_weights = torch.unsqueeze(read_weights, dim=-1)  # [B,H,K,1]
        memory_reads = torch.sum(expanded_weights * topk_mem, dim=2)  # [B,H,M]
    else:
        read_weights = F.softmax(weighted_distances, dim=-1) ##approx 1 hot (goal is to select correct row in memstate) ; [B, H, R] (same dim as cosine_distances and weighted_distances)

        num_rows_memory = mem_state.shape[1]
        all_indices = torch.arange(num_rows_memory).int()
        all_indices = torch.reshape(all_indices, [1, 1, num_rows_memory])
        read_indices = all_indices.repeat([batch_size, num_read_heads, 1])

        # This is the actual memory access.
        # Note that matmul automatically batch applies for us.
        memory_reads = torch.matmul(read_weights, mem_state) ##each of H rows in memory_reads is a weighted sum of the rows in mem_state, weighted by similarity. in ideal case, should be identical to correct row in memstate

    assert read_keys.shape == memory_reads.shape

    # [B, H, 1] -> [B, H]
    read_strengths = torch.squeeze(read_strengths, dim=-1)

    return memory_reads, read_weights, read_indices, read_strengths


def _assert_compatible_read_from_memory_inputs(read_keys, read_strengths,
                                               mem_state):
    assert len(read_keys.shape) == 3
    b_shape, h_shape, m_shape = read_keys.shape
    assert len(mem_state.shape) == 3
    r_shape = mem_state.shape[1]

    assert read_strengths.shape == (b_shape, h_shape)
    assert mem_state.shape == (b_shape, r_shape, m_shape)


def _assert_compatible_memory_reader_input(input_tensors):
    """Asserts MemoryReader's _build has been given the correct shapes."""
    assert len(input_tensors) == 2
    _, mem_state = input_tensors
    assert len(mem_state.shape) == 3


class Memory_old(nn.Module):
    """Thin wrapper of a memory reader and writer."""

    def __init__(self,
                 input_size,
                 batch_size,
                 memory_word_size,
                 num_rows_memory,
                 num_read_heads=1,
                 top_k=0,
                 ):
        super().__init__()

        self._memory_reader = MemoryReader(
            input_size=input_size, memory_word_size=memory_word_size,
            num_read_heads=num_read_heads, top_k=top_k
        )
        self._memory_writer = MemoryWriter(
            mem_shape=(num_rows_memory, memory_word_size))

        self.mem_state = torch.zeros(
            (batch_size, num_rows_memory, memory_word_size))
        self.mem_writer_state = torch.tensor(0)

    def forward(self, inputs):
        """

        Args:
             inputs: tuple (reader_input, writer_input)
                reader_input (batch_size, input_size)
                writer_input (batch_size, memory_word_size)

        Returns:
            mem_reads: read out (B, H * M)
            read_info: named tuple
        """
        reader_input, writer_input = inputs
        mem_reads, read_info = self._memory_reader(
            (reader_input, self.mem_state))
        self.mem_state, self.mem_writer_state = self._memory_writer(
            (writer_input,  self.mem_state), self.mem_writer_state)

        return mem_reads, read_info


class Memory(nn.Module):
    """Thin wrapper of a memory reader and writer."""

    def __init__(self,
                 input_size,
                 batch_size,
                 memory_word_size,
                 num_rows_memory,
                 num_read_heads=1,
                 top_k=0,
                 ):
        super().__init__()

        self._memory_reader = MemoryReader(
            input_size=input_size, memory_word_size=memory_word_size,
            num_read_heads=num_read_heads, top_k=top_k,
            keys_and_read_strengths_mode='linear_with_softplus_orig_TVT')
        self._memory_writer = MemoryWriter(
            mem_shape=(num_rows_memory, memory_word_size))

        self.mem_state = torch.zeros(
            (batch_size, num_rows_memory, memory_word_size))
        self.mem_writer_state = torch.tensor(0)

    def forward(self, inputs, modu_input, target=None):
        """
        Args:
            inputs: (seq_len, batch_size, input_size)

            B: batch size
            M: number of slots in a row of the memory matrix
            R: number of rows in the memory matrix
            H: number of read heads (of the controller or the policy)
            K: top_k if top_k>0

        Returns:
            output: (seq_len, batch_size, output_size)
            rnn_state: None
        """
        assert target is None #target param only for method signature compatibility

        self.mem_writer_state = torch.tensor(0)

        output = list()
        self.mem_state = torch.zeros(self.mem_state.shape)
        inc = 0
        for input_t in inputs: ##for each of seq_len inputs
            # TODO: temporarily assuming reader_input = writer_input
            # reader_input should be (batch_size, input_size)
            reader_input = input_t  # (batch_size, input_size)

            # writer_input should be (batch_size, memory_word_size)
            if modu_input[inc].item() == 1:
                writer_input = input_t  # (batch_size, input_size)
                self.mem_state, self.mem_writer_state = self._memory_writer(
                    (writer_input,  self.mem_state), self.mem_writer_state)

            # mem_reads: (B, H * M), read_info: named tuple
            mem_reads, read_info = self._memory_reader( ##when debugging: check memstate at this line
                (reader_input, self.mem_state))


            output.append(mem_reads)
            inc += 1

        output = torch.stack(output, 0)  # (seq_len, batch_size, H * M)

        return output, None
