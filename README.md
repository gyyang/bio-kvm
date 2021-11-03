# bio-kvm

[Biological learning in key-value memory networks](https://arxiv.org/abs/2110.13976#)

### Recreating Figures
All figures from the paper can be recreated by running `notebooks/plot_all_figs.ipynb`.
Some figures require pre-trained models (see directions on training and analysis below).
Below are the figures that require pre-trained models and the names of the
corresponding experiments that need to be run:

| Figure | Experiments to Run |
| --- | ----------- |
| 2b | tvt |
| 3a | train\_random\_capacity |
| 3c | train\_prepost\_zero\_init |
| 4a | train\_continual |
| 4c | train\_corr |
| 5a | heteroassociative\* |
| 5b | seqrecall\* |
| 5c | copy\* |

### Training experiments
The functions found in the `experiments.py` file correspond to the name of
each experiment. Experiments can be run with:
`python main.py --train {experiment name}`

Generic analysis of training progress (or specialized analysis for some experiments)
can be done by running the following after training:
`python main.py --analyze {experiment name}`

