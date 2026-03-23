# predict-element-instead-of-sequence

This is the code repository for the article ["Predict the element instead of the sequence: ResNet surrogate method for very accurate predictions of hourly building energy"](https://doi.org/10.1016/j.apenergy.2026.127739) in the Applied Energy journal.

## Installation

To install and test the code yourself, follow these steps:
1. Clone this repository to your local disk.
2. Install [EnergyPlus V24-1-0](https://github.com/NatLabRockies/EnergyPlus/releases/tag/v24.1.0). 
    **NOTE:** Make sure that the path to the EnergyPlus executable is correctly defined in the `run` method of the `Building` class in `pred_elem_seq/simulation.py`. The path is set to `C:/EnergyPlusV24-1-0/energyplus.exe` by default.
3. Install the [uv](https://docs.astral.sh/uv/) Python package and project manager.
4. Run the following two commands in the `predict-element-instead-of-sequence` root directory: 
    1. `uv venv`
    2. `uv pip install -r pyproject.toml`.

## Running the code

Even though it is not common practice, we have organized the various implementations of our method into `pytest` functions.
The arguments of the `pytest` functions are organized in the `tests/conftest.py` module as `pytest` fixtures. Feel free to modify the values
of the fixtures to test different arguments (such as surrogate model hyperparameters).
Use your favourite IDE (or use the command line) to run the main `pytest` functions explained below.

### Main `pytest` functions

1. `tests/test_surrogate.py`:
    1. `test_get_surrogate`
    
    This function uses the `ann` and `ann_datasets_train` `pytest` fixtures to train, test, and save a surrogate model of a building energy model (BEM) with only **one** weather file in the design space.
    
    The `ann` fixture is an instance of the `AnnConfig` dataclass that contains all the metadata necessary to clearly define the type and characteristics of the surrogate model (*e.g.*, number of hidden layers), the details of the BEM (*e.g.*, building parameters in the surrogate design space), and details of the surrogate model training and testing datasets (*e.g.*, number of samples).

    The `ann_datasets_train` fixture is an instance of the `AnnDatasets` class that is used to generate the surrogate model training and testing  by simulating the BEM through a design of experiment.

    2. `test_get_surrogate_weather`

    This function uses the `ann` and `ann_datasets_weather_train` `pytest` fixtures to train, test, and save a surrogate model of a building energy model (BEM) with **multiple** weather files in the design space.

     The `ann_datasets_weather_train` fixture is also an instance of the `AnnDatasets` class, like `ann_datasets_train`, but it also uses `weather_list` as an argument, which is a list of .epw file paths that will be randomly selected during the simulations of the BEM.