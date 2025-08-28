# catechol_solvent_selection (Forked for the SoDaDE model testing)

Repository for the code and data on the catechol dataset for solvent selection and machine learning.

The SoDaDE model has been added to this benchmark for testing, with the testing and loading code which has been set up by the authors of the paper. 

## Added SoDaDE README

### SoDaDE Requirements
The entire catechol `requirements.txt` is not necessary to run the SoDaDE model finetuning. For just the SoDaDE model testing, `SoDaDE_requirements.txt` and `Python==3.11.13` were used.

To test this functionality, install the requirements using:
```bash
pip install -r SoDaDE_requirements.txt
```

### The relevant additions to the code base are the following:
-`fine_tune.py`: This file finetunes the SoDaDE model on a selected dataset through the `SoDaDE_regression.py` file. The full data task is pre-set but can be changed to the 'single solvent' task easily. Optimal parameters are present and it saved a .csv of the results in the `results` directory. The folder the .csv is saved into differs according to the task `full_data` or `single_solvent` and whether the SoDaDE model was frozen during finetuning `decoderFalse` (SoDaDE learning happened) or `decoderTrue` (SoDaDE model weights were frozen).

-`SoDaDE_regression.py`: This model calls either `decoder_full_yields.py` or `decoder_single_solvent.py` depending on the desired task. Parameters can be manually specified for parameter searching.

- `decoder_full_yields.py` which is the SoDaDE model adapted to the full_data dataset. The data loader loads the `data\full_data\catechol_full_data_yields.csv` and goes through 'leave one out splits' according to ramp number and solvent. 

- `decoder_single_solvent.py` is the SoDaDE model adapted to the single_solvent dataset. The data loader loads the data `data\single_solvent\catechol_single_solvent_yields.csv` and runs 'leave one out' splits according to solvent name.

- `SoDaDE_DM_64_TL_5_heads_16.pth` is the pretrained model from the SoDaDE repository. This is the model that achieved the lowest MSE on the property prediction task and is being fine-tuned on either the 'full_data' or 'single_solvent' tasks. 

These scripts have been saved in the main directory for easy use and reference. 

### Additional directories which have been added to the repository are the following:

- `SoDaDE_results` contains the results of large batch parameter grid search jobs completed on a computing cluster. For each task, 'full_data' or 'single_solvent', and whether the SoDaDE model was frozen a batch job was done and each result was saved into their respective directories (see description of `fine_tune.py`. The performance on each ramp or solvent is saved as a `.json` file in the directory. )

-`batch_job_results\results_analysis.ipynb` is a jypter notebook which collates each large folder of .csv files into 4 .csv files for easier analysis. In the 4 following cells, it shows the average MSE's of the best performing parameters for each task and whether the SoDaDE model was frozen or not. 

- `models_for_plotting_solvent_embeddings` is a directory containing code to train an 'illustrative' model on the entire single solvent dataset to show how the solvent embeddings change coming out of the SoDaDE model and after the first layer of the neural network. These models are purely for illustrative purposes and extract embeddings for each epoch or batch trained.

## Standard Catechol Solvent Selection README

### Installation

The requirements for this project can be installed using:
```bash
pip install -e .
```
or
```bash
pip install -r requirements.txt --no-deps
```

We recommend installing these in a fresh virtual environment, with Python version
3.11 or greater. We use [`uv`](https://docs.astral.sh/uv/) to manage dependencies, 
but you don't need to install this yourself. If you wish to add more packages to
the requirements, and you don't have `uv` installed, create a PR/issue.


## Dataset

Here, we provide a brief overview of the dataset in this repo.

The dataset is divided into two:
- `catechol_single_solvent_yields` contains only the single-solvent data
- `catechol_full_data_yields` contains the full data set with mixture solvents

We also provide some pre-computed featurizations, which can be looked up with the 
`SOLVENT NAME` column.

### Single solvent columns
Below is a table of all the columns in the `catechol_single_solvent_yields` csv:

| Name | Type | Description |
|--------|--------|--------|
| `EXP NUM` | int| Experiment index; all rows with the same `EXP NUM` will use the same solvent|
| `Residence Time` | float | Time (in minutes) of the reaction|
| `Temperature`| float | Temperature (in Celsius) of the reaction|
| `SM` | float | Quantity of starting material measured (yield %)|
| `Product 2` | float | Quantity of product 2 measured (yield %)| 
| `Product 3` | float | Quantity of product 3 measured (yield %)| 
| `SOLVENT_NAME` | str | Chemical name of the solvent; used as a key when looking up featurizations| 
| `SOLVENT_RATIO` | list[float] | Ratio of component solvents [1]|
| `{...} SMILES` | str | SMILES string representation of a molecule|

[1] This is different than the ratios in the solvent ramp experiments. Here, a single solvent has two component molecules, eg. the solvent "Acetonitrile.Acetic Acid" has two compounds. The `SOLVENT_RATIO` gives the ratio between these compounds. Most solvents consist of only a single compound, so the ratio will be `[1.0]`.

**Inputs**: `Residence Time`, `Temperature`, `SOLVENT NAME`

**Outputs**: `SM`, `Product 2`, `Product 3` 

### Full data columns

The full data contains some additional columns, since these experiments ramp between
two solvents:

| Name | Type | Description |
|--------|--------|--------|
| `SolventB%` | float | Percent concentration of solvent B; the rest of the solvent is made up of solvent A|
| `SOLVENT {A/B} NAME` | str | Chemical name of the solvents; used as a key when looking up featurizations|

**Inputs**: `Residence Time`, `Temperature`, `SOLVENT A NAME`, `SOLVENT B NAME`, `SolventB%`

**Outputs**: `SM`, `Product 2`, `Product 3` 


## Running experiments

All of the scripts used to carry out the experiments are in `scripts/`, including
- `eval_solvent_ramps.py`
- `eval_single_solvents.py`
- `eval_transfer_learning.py`
- `eval_active_learning.py`
- `eval_bayes_opt.py`

Each of these files takes, as argument, a model and a featurization, as well
as an additional configuration string. See the individual scripts for usage examples.