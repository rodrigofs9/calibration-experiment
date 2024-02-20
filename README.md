# Calibration Experiment

This repository contains the code for the calibration experiment in post-processing and in-processing in recommendation systems. The system has been configured to allow the execution of 8 types of calibration: genres calibration, popularity calibration, double calibration, custom calibration, pairwise calibration, BPR, and a modification of BPR. Additionally, three datasets are integrated for experiment execution.

## Repository Structure

- **main.py:**
  - Contains the main code for executing the experiment.
  - Allows the user to select the fold, calibration type, and dataset.
  - Groups users and items from the dataset, considering popularity aspects.
  - Executes the experiment based on the specified parameters.

- **dataset.py:**
  - Contains configurations for datasets.
  - Removes items and users with few interactions to clean the dataset.

- **calibration.py:**
  - Contains the code responsible for reclassifying the recommendation list.
  - Calculates divergences, either KL or Jansen divergence.

- **metrics_calculation.py:**
  - Contains the code that calculates and records all the metrics analyzed during the experiment.

- **metrics.py:**
  - Contains the code with the calculation for each of the metrics implemented in `metrics_calculation.py`.

- **popularity_calculation.py:**
  - Contains the code that groups items based on the number of interactions in each popularity group.

- **rerank_algo.py:**
  - Contains the code responsible for reclassifying the recommendation list according to the selected calibration type.

- **datasets/:**
  - Contains files related to the datasets used in the experiment.

- **baselines/:**
  - Contains files related to baseline algorithms, such as pairwise and others.

- **bpr_files/:**
  - Contains the code for the BPR (Bayesian Personalized Ranking) algorithm and its modified version.

## How to Run the Project

1. **Prerequisites:**
   - Make sure you have Python installed. We recommend version 3.7 or higher.

2. **Running the Experiment:**
   - Inside the project package execute the following command to start the experiment:
     ```bash
     python main.py
     ```

3. **Results:**
   - The results will be saved in the /results directory.

### Running a Quick Version of the Experiment

To run a quick and simple version of the experiment, replace the line:
```python
exp_results = exp.map(f, set(test["user"]))
```

with
```python
exp_results = exp.map(f, list(islice(test["user"], 10)))
```
