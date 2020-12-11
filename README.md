# Data Generating Process to Evaluate Causal Discovery Techniques for Time Series Data
This repository captures source code and data sets for our paper "Data Generating Process to Evaluate Causal Discovery Techniques for Time Series Data" at the [Causal Discovery &amp; Causality-Inspired Machine Learning Workshop](https://www.cmu.edu/dietrich/causality/neurips20ws/) at Neural Information Processing Systems (NeurIPS) 2020. For ease, the [PDF](https://github.com/causalens/cdml-neurips2020/blob/main/dgp_causal_discovery_time_series.pdf) of the paper is provided in this repository, but can also be found on the workshop website [here](https://www.cmu.edu/dietrich/causality/CameraReadys-accepted%20papers/51%5cCameraReady%5ccamera_ready_submission.pdf).

If you utilise our data generating process, please cite:
```
@article{dgp_causal_discovery_time_series,
  title={Data Generating Process to Evaluate Causal Discovery Techniques for Time Series Data},
  author={Lawrence, Andrew R. and Kaiser, Marcus and Sampaio, Rui and Sipos, Maksim},
  journal={Causal Discovery & Causality-Inspired Machine Learning Workshop at Neural Information Processing Systems},
  year={2020}
}
```

## Usage

### Setup
The code was developed and tested using Python 3.7.8. The required Python packages, and their versions, are provided in `requirements.txt`. After cloning the repository locally, create a virtual environment by running the following commands:
```
python3 -m venv <env_name>
source <env_name>/bin/activate
pip install -r <path_to_local_repo>/requirements.txt
deactivate
```

### Example Script
A simple example script is provided in the `dgp` folder. Running `example_script.py` will generate some synthetic data, plot it, and plot the causal diagram of the underlying Structural Causal Model (SCM) from which the data was generated. Looking at the unit tests in `dgp/tests` provide further insight into how the code works.

## Experiment Data Sets

The data sets used for the experiments in our NeurIPS 2020 workshop paper are too large to maintain on GitHub. However, they can be downloaded from [here](https://drive.google.com/file/d/1rQ0DlfxH-Ec5KXClH15Bz6Mz8TJp_Vm0/view?usp=sharing).

All the data sets reside in a single zip file. When extracted, there is a *pickle* file for each experiment:
1. Causal Sufficiency: *causal_sufficiency_data.pickle*
2. Non-Linear Dependencies: *linear_data.pickle*
3. Instantaneous Effects: *instantaneous_effects_data.pickle*
4. IID Data: *iid_data.pickle*
5. Non-Gaussian Noise: *increasing_non_gaussian_noise_data.pickle*

The following code snippet demonstrates how to load the causal sufficiency data:
```
import dill as pickle

filename = '<path_to_file>/causal_sufficiency_data.pickle'

with open(filename, 'rb') as f:
    data = pickle.load(f)
```

The `data` variable will be `List[List[Tuple[pandas.DataFrame, networkx.DiGraph]]]`, where the outer list captures each unique configuration under test, which are described further below. Therefore, the length of the outer list is different for each experiment as a different number of configurations were used for each experiment. The inner list always has `200` elements as we tested the causal discovery methods on data produced from `200` randomly generated SCMs for each configuration. Please see the paper for more information.
1. Causal Sufficiency: `13` configurations, where number of latent variables is `[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 15, 20]` and the number of feature variables are kept constant at `10`.
2. Non-Linear Dependencies: `11` configurations where likelihood of linear functions vs. other functional forms goes from 100% to 0% in 10% increments.
3. Instantaneous Effects: `2` configurations where the minimum possible lag is `[1, 0]`. 
4. IID Data: For IID data, `min_lag = max_lag = 0`. There are `6` configurations, where the number of samples is `[50, 100, 200, 500, 1000, 2500]`.
5. Non-Gaussian Noise: `7` configurations where likelihood of Gaussian noise vs. other possible noise distributions goes from 100% to 10% in 15% increments.
