# Higgs CP Structure Identification using Tau-Tau Decay

This repository contains the code and models for identifying the CP structure of the Higgs boson through its decay into tau-tau pairs

## Prerequisites

Before you begin, ensure you have git installed on your machine to clone this repository. If git is not installed, you can download it from [Git's official site](https://git-scm.com/downloads).

## Installation

Follow these steps to set up your environment and start analyzing the Higgs boson's CP structure.

### Step 1: Clone the Repository

Clone this repository to your local machine using the following command:

```bash
git clone https://github.com/wesmail/HiggsCP.git
cd HiggsCP
```

### Step 2: Install Conda

If you do not have Miniconda or Anaconda installed, download and install it from [Miniconda](https://docs.conda.io/en/latest/miniconda.html) or [Anaconda](https://www.anaconda.com/products/individual) respectively.

### Step 3: Set Up Your Environment

This project relies on several dependencies listed in `environment.yml`, including libraries such as NumPy, Pandas, Matplotlib, tqdm, h5py, scikit-learn, PyTorch, PyTorch Geometric, PyTorch Lightning, and Torchmetrics.

To install all dependencies at once and create a Conda environment named `h2ttbar`, run the following command in your terminal:

```bash
conda env create -f environment.yml
```

### Step 4: Activate the Environment
```bash
conda activate h2ttbar
```

## Usage

### Download the Data
You need first to download the data and store it in the `files/` directory. The data is stored in [Google Drive](https://drive.google.com/drive/folders/1Sba8uLfluBHdNO2tnSuCot5lKS06B0tB?usp=sharing).

### Training the Model

You can train the model using the `run.sh` script provided in the repository. This script supports running the training process for each angle individually.

To see how to use the script, you can type:

```bash
./run.sh -h
```
For example, to train the model on angle `0`, you can use the following command:

```bash
./run.sh --mode "train" --angle "0"
```
This command will create an HDF5 file named `data_0.hdf5` containing the signal and background data, which will be used to train the heterogeneous Graph Neural Network. The training results, including the saved model, hyperparameters, and training progress, will be stored in a directory named `h2tt_angle_0_results`.

### Configuration
The training (and testing) hyperparameters, such as the number of epochs, learning rate, and size of the network, are stored in the `train.yaml` file located in the `config/` directory. You can override any of these parameters by modifying this file before running the training or testing commands.

### Training All Angles
If you wish to train models for all angles at once, you can use the `train_all_angles.sh` script:

```bash
./train_all_angles.sh
```

### Testing the Trained Model
To test a trained model, you need to provide the path to the trained model's checkpoint file to the `run.sh` script. For example, to test a model trained on angle `0` can be something like the following line, where you need to adjust the `ckpt_path` where the trained model is:

```bash
./run.sh --mode "test" --ckpt_path "h2tt_angle_0_results/version_0/checkpoints/epoch=0-step=109.ckpt" --h5_file "files/data_0.hdf5"
```
This command will perform the testing on the specified model and data file. The results, including **ROC** and **angular distribution** $\phi^{*}$ plot, will be saved under the `h2tt_angle_0_results` directory.



