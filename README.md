# Norwegian Coast Extreme Waves

Code for my master's thesis:

**Bias Correction Framework for Extreme Wave-Height Estimation on the Norwegian Coast Using NORA3 Hindcast and Buoy Observations**

This repository contains the code used to correct bias in NORA3 significant wave height (`Hs`) data along the Norwegian coast. The focus is on comparing NORA3 hindcast data with buoy observations, applying bias-correction methods, combining models with a Mixture of Experts approach, and studying how the correction affects extreme wave-height return levels.

## Overview

The workflow has four main steps:

1. Prepare paired NORA3 and buoy-observation datasets.
2. Apply bias-correction methods to reduce systematic error in NORA3 `Hs`.
3. Combine correction methods using a Mixture of Experts model.
4. Estimate extreme wave-height return levels before and after correction.

<p align="center">
  <table>
    <tr>
      <td align="center" bgcolor="#ffffff" style="background-color:#ffffff; padding:24px;">
        <img src="https://github.com/user-attachments/assets/345d19ff-9c64-437c-aed8-566d06ddfb41" width="900">
      </td>
    </tr>
  </table>
</p>

## Study area

The project uses NORA3 hindcast data and wave-buoy observations from selected Norwegian coastal locations.

<p align="center">
  <img src="https://github.com/user-attachments/assets/a4283f0e-7847-4a38-a40b-c4f1d6dada1c" width="620">
</p>

<p align="center">
  <img src="https://github.com/user-attachments/assets/2da5dfb0-9d6e-45d4-b175-dbdaafae6710" width="620">
</p>

The main buoy locations are Fauskane and Fedjeosen. Vestfjorden is used for external validation, while Bergen, Stavanger, and Kristiansund are included as study-area locations.

## Technical approach

The code tests several bias-correction methods for NORA3 `Hs`. These include both statistical and machine-learning methods.

The main correction methods are:

- Linear correction
- Parametric Quantile Mapping
- Directional Adjusted Gumbel Quantile Mapping
- Gaussian Process Regression
- XGBoost
- Transformer-based correction

The machine-learning methods use NORA3 wave and wind variables as input features. Hyperparameters are tuned with Optuna, with emphasis on improving the upper tail of the wave-height distribution.

The corrected outputs are then combined using a **Mixture of Experts (MoE)** model. In this step, an XGBoost-based gating model assigns weights to the individual correction methods. This allows the final correction to depend on the sea state instead of relying on one fixed method for all conditions.

Two validation setups are used:

1. **Local correction**  
   A model is trained and tested at the same buoy location using cross-validation.

2. **Transfer correction**  
   A model trained at one buoy location is applied to another location. This is used to test whether bias correction can be transferred spatially along the coast.

## Bias in NORA3

NORA3 provides long hindcast time series, but it can differ systematically from local buoy observations. This is especially important for high waves, since extreme-value estimates depend strongly on the upper tail of the wave-height distribution.

The figure below summarizes the raw NORA3 RMSE across locations and quantiles. Errors increase toward the upper tail, especially at Fauskane and Fedjeosen.

<p align="center">
  <img src="https://github.com/user-attachments/assets/7b79787d-3122-4e08-852b-1a4588dbb3f6" width="900">
</p>

The figure below shows the largest observed storm event at Fauskane during the overlap period. NORA3 captures the event evolution, but underestimates parts of the peak.

<p align="center">
  <img src="https://github.com/user-attachments/assets/5e44353f-5f8d-42d7-a0fb-85fb9e146e61" width="900">
</p>

## Bias correction results

Several correction methods improve the agreement between NORA3 and buoy observations. The figure below shows an upper-tail QQ comparison after correction at Fauskane. The corrected models are closer to the 1:1 line than raw NORA3, especially in the upper tail.

<p align="center">
  <img src="https://github.com/user-attachments/assets/be60b3ef-abb1-4ae3-aafa-dba9eb2ad04b" width="620">
</p>

The MoE model is used to combine the correction methods. The figure below summarizes the relative RMSE change compared with raw NORA3. Negative values indicate improvement, while positive values indicate higher error than raw NORA3.

<p align="center">
  <img src="https://github.com/user-attachments/assets/36fa1b37-6ece-4861-818e-a5bb250af9ca" width="900">
</p>

## Extreme wave-height results

The corrected time series are used for extreme value analysis. Return levels are estimated from raw NORA3 and corrected `Hs` series to evaluate how bias correction changes extreme-wave estimates.

<p align="center">
  <img src="https://github.com/user-attachments/assets/e3d82e17-ccb1-40c2-ab57-744a5458ea92" width="650">
</p>

## Repository structure

The repository is organized around the main parts of the thesis workflow:

- `src/bias_correction/` — bias correction methods and pipeline
- `src/ensemble/` — Mixture of Experts model
- `src/optuna_parameter_search/` — hyperparameter tuning
- `experiments/` — scripts for running experiments
- `results/` — generated outputs, figures, and validation results

## Running the code

The main workflow used in the thesis was:

1. Prepare paired NORA3 and buoy-observation datasets.
2. Optionally tune XGBoost and Transformer with Optuna.
3. Update the selected hyperparameters in `src/model_profiles.py`.
4. Run bias correction for each location.
5. Optionally tune the Mixture of Experts model.
6. Run the ensemble model.
7. Run extreme value modelling.

## Author

Per Simen Reed  
Master's thesis, University of Agder
