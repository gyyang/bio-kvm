import unittest

import datasets.recall as recall

class TestRecall(unittest.TestCase):

    def test_Recall(self):
        T = 10
        p_recall = 0.5
        n_repeat = 1
        stim_dim = 5

        dataset = recall.RecallDataset(
            stim_dim=stim_dim,
            T_min=T,
            T_max=T,
            p_recall=p_recall,
            n_repeat=n_repeat
        )
        data = dataset.generate()

        T_total = int(T * (1 + p_recall)) * n_repeat
        assert data['input'].shape == (T_total, stim_dim)
        assert data['modu_input'].shape == (T_total, 1)
        assert data['target'].shape == (T_total, stim_dim)
        assert data['mask'].shape == (T_total,)

    def test_RecallRepeat(self):
        T = 10
        p_recall = 0.5
        n_repeat = 2
        stim_dim = 5

        dataset = recall.RecallDataset(
            stim_dim=stim_dim,
            T_min=T,
            T_max=T,
            p_recall=p_recall,
            n_repeat=n_repeat
        )
        data = dataset.generate()

        T_total = int(T * (1 + p_recall)) * n_repeat
        assert data['input'].shape == (T_total, stim_dim)
        assert data['modu_input'].shape == (T_total, 1)
        assert data['target'].shape == (T_total, stim_dim)
        assert data['mask'].shape == (T_total,)


if __name__ == '__main__':
    unittest.main()