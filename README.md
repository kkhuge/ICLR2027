# Understanding Generalization in Client-Level DP-FL: A Hierarchical Stability Analysis

This repository contains the code for the experiments and figures in *Understanding Generalization in Client-Level DP-FL: A Hierarchical Stability Analysis*.

## Generating the data

Run the following command to download the required public datasets and generate the client partitions:

```bash
python reproduce.py --stage data --group all
```

The data scripts are in `data/`: `femnist/prepare_client_level_hierarchical.py` creates the natural and IID TFF FEMNIST splits; the scripts under `data/fmnist/` create the Fashion-MNIST heterogeneity, local-size, and IID splits; and `cifar10/generate_cifar_iid.py` creates the CIFAR-10 IID partition. `download_femnist.py` obtains the official TFF FEMNIST source files.

## Running the experiments and plotting the figures

From this directory, run:

```bash
python reproduce.py --stage train --group all
python reproduce.py --stage plot --group all
```

Use `--group` to run only one experiment family:

| Group | Figures |
| --- | --- |
| `femnist` | Training, within-client test, and unseen-client test loss and accuracy under a natural non-IID partition |
| `heterogeneity` | Within-client and unseen-client generalization gaps |
| `local_size` | Within-client and unseen-client gaps versus local dataset size |
| `learning_rate` | Training loss and generalization gap under learning-rate decay |
| `privacy` | Generalization gaps under fixed and varying privacy budgets |
| `cifar_sigma` | Accuracy and gap versus noise multiplier |
| `cifar_clip` | Accuracy and gap versus clipping threshold |
| `cifar_participation` | Accuracy and gap versus client participation |
| `cifar_tau` | Accuracy and gap versus local training steps |

For example:

```bash
python reproduce.py --group heterogeneity --stage all
```

The plotting scripts are listed in `reproduce.py`, and the generated figures are saved as PDFs in `figure_output/`.

## Dependencies

- Python >= 3.10
- PyTorch = 2.9.0
- torchvision = 0.24.0
- NumPy = 2.1.2
- Matplotlib = 3.10.7
- h5py = 3.16.0
- Pillow >= 10.0
- tensorboardX
