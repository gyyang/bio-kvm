from datasets.recall import RecallDataset, EcstasyRecall
from datasets.copypaste import CopyPasteDataset
from datasets.seqrecall import SequenceRecallDataset
import tools


def get_dataset(config, verbose=True):
    if config['name'] == 'recall' or config['name'] == 'balanced_recall':
        dataset = RecallDataset(**config)
    elif config['name'] == 'ecstasy':
        dataset = EcstasyRecall(**config)
    elif config['name'] == 'copypaste':
        dataset = CopyPasteDataset(**config)
    elif config['name'] == 'seqrecall':
        dataset = SequenceRecallDataset(**config)
    else:
        raise ValueError('Unknown dataset; implement dataset.get_dataset')
    if verbose:
        print(dataset)
    return dataset


def visualize_dataset(path_or_dataset):
    """Visualize dataset from path.

    Visualize all unique datasets.
    """

    if isinstance(path_or_dataset, str):
        # TODO: Add automatic check
        path = path_or_dataset
        modeldirs = tools.get_modeldirs(path)
        unique_datasets = []
        for i, modeldir in enumerate(modeldirs):
            config = tools.load_config(modeldir)
            dataset_config = config['dataset']

            if dataset_config not in unique_datasets:
                unique_datasets.append(dataset_config.copy())
                dataset = get_dataset(dataset_config)
                figname = 'dataset_' + tools.get_model_name(modeldir)
                dataset.visualize(figpath=path, figname=figname)
    else:
        path_or_dataset.visualize()
