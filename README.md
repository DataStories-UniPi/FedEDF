# FedEDF
Data and source code for the paper "On Electric Vehicle Energy Demand Forecasting and the Effect of Federated learning"


# Table of Contents
- Overview
- Prerequisites & installation
- Data preparation
    - EVSE transaction processing
    - Metadata processing
    - Feature engineering
- Model training
    - Centralized learning (baseline; ARIMA, SARIMA, SARIMAX)
    - Centralized learning (ML-based; XGBoost)
    - Centralized learning (ML-based; [Bi-]LSTM/GRU)
    - Federated learning (ML-based; XGBoost)
    - Federated learning (ML-based; [Bi-]LSTM/GRU)
- Inference & visualization
- Contributors
- Acknowledgement


# Overview 
This repository provides a complete end‑to‑end pipeline for Electric Vehicle Supply Equipment (EVSE) demand forecasting. The pipeline covers:
- Raw data ingestion & cleaning – outlier filtering, timeseries creation.
- Metadata enrichment – location (geocoding), charger type, nominal power output.
- Feature engineering – resampling, lag variables, cyclical encodings, rolling statistics, extrapolation.
- Centralised models – statistical baselines (ARIMA, SARIMA, SARIMAX), XGBoost, RNN ([Bi-]LSTM/GRU).
- Federated learning – privacy‑aware training of RNNs and XGBoost across multiple "EVSE hubs" using Flower.
- Evaluation & visualisation – performance tables (MASE, SMAPE, RMSE, R²) and geographic maps of EVSE locations.

All scripts are written in Python 3.10. For the sake of reproducibility, the `/data` directory includes the preprocessed datasets, along with pretrained models.


# Prerequisites & installation
1. Python 3.10
1. System libraries – git, curl, wget (for dataset download)
1. (Optional) GPU – A CUDA‑enabled GPU speeds up model training; otherwise the CPU fallback works out‑of‑the‑box.
1. Python packages (cf., requirements.txt)
```bash
pip install -r requirements.txt
```

**Note** – The code uses the Nominatim OpenStreetMap API for geocoding. No API key is required, but respect the usage policy (https://operations.osmfoundation.org/policies/nominatim/). The original EVSE datasets can be found at: https://github.com/yvenn-amara/ev-load-open-data


# Data preparation
## EVSE transaction processing
```bash
python 0-data-processing.py \
    --dataset {dundee,boulder,paloalto} \
    --sr_freq 12H \         # resampling frequency
    --min_pts 100           # minimum transactions per EVSE
```

## Build Metadata
```bash
python 1-metadata-preprocessing-dundee-v3.py
python 1-metadata-preprocessing-boulder-v3.py
python 1-metadata-preprocessing-paloalto-v3.py
```


## Feature Engineering
Open and execute all cells in the notebook `1.5-feature-engineering-v3.ipynb` for each dataset.


# Model training
## Centralized learning (baselines; ARIMA, SARIMA, SARIMAX)
```bash
python 2-statistical-training-v4.py --dataset {dundee, boulder, paloalto} --sr_freq 12H --min_pts 100
```

## Centralized learning (ML-based; XGBoost)
Open and execute all cells in the notebook `3-xgboost-training-v5.ipynb` for each dataset.

## Centralized learning (ML-based; [Bi-]LSTM/GRU)
```bash
python 4-rnn-training-v3.py \
    --data {dundee,boulder,paloalto} \
    --njobs 96 \
    --hidden_size 24 \
    --rnn_cell {lstm,gru} 
    [--bi] \
    --num_layers 1 \
    --fc_layers 24 \
    --sr_freq 12H \
    --min_pts 100 \
    --bs 32 \
    --length 48 \
    --stride 1 \
    --n_epochs 100 \
    --patience 10 \
```

## Federated learning (ML-based; XGBoost)
```bash
# FedXGBllr (Average)
python fededf_server.py \
    --federation {dundee,boulder,paloalto} \
    --num_rounds 40 \
    --local_epochs 10 \
    --early_stop \
    --patience 3 \
    --port 28080 \
    --mu 0.0 \
    --clients 8 \
    --conv_channels 16 \
    --dropout_rate 0.13 \
    --strategy fedxgbllr

# FedXGBllr (Proximal)
python fededf_server.py \
    --federation {dundee,boulder,paloalto} \
    --num_rounds 40 \
    --local_epochs 10 \
    --early_stop \
    --patience 3 \
    --port 28080 \
    --mu 1e-1 \
    --clients 8 \
    --conv_channels 16 \
    --dropout_rate 0.13 \
    --strategy fedxgbllr

# Launch federation
bash launch_fedxgb_clients.sh \
    --federation {dundee,boulder,paloalto} \
    --n_estimators 37 \
    --sr_freq 12H \
    --min_pts 100 \
    --bs 64 \
    --port 28080
```

## Federated learning (ML-based; [Bi-]LSTM/GRU)
```bash
# FedProx
python fededf_server.py \
    --federation {dundee,boulder,paloalto} \
    --num_rounds 50 \
    --local_epochs 5 \
    --early_stop \
    --patience 2 \
    --port 28080 \
    --mu 1e-1 \
    --clients 8 \
    --strategy fedprox

# Launch federation
bash launch_fedrnn_clients.sh \
    --federation {dundee,boulder,paloalto} \
    --rnn_cell {lstm, gru} \
    [--bi] \
    --hidden_size 24 \
    --num_layers 1 \
    --fc_layers 24 \
    --sr_freq 12H \
    --min_pts 100 \
    --bs 32 \
    --length 48 \
    --stride 1 \
    --port 28080
```

**Note** – Before running any FL-related experiments, use notebook `6-data-federation.ipynb` to generate the data for each client in the federation.

# Inference & visualization
After training, the server stores the global checkpoint under `/data/pth/`. 

To obtain model forecasts, execute the code in notebooks `4.1-rnn-inference-v3.ipynb`, `6.1-federated-rnn-inference.ipynb` and ``6.2-federated-xgb-inference.ipynb``. 

Use notebooks `5-centralized-edf-evaluation.ipynb` and `6.3-federated-edf-evaluation.ipynb` to generate the comparison tables.

# Contributors
- Andreas Tritsarolis; Department of Informatics, University of Piraeus
- Gil Sampaio; Center for Power and Energy Systems, INESC TEC
- Nikos Pelekis; Department of Statistics and Insurance Science, University of Piraeus
- Yannis Theodoridis; Department of Informatics, University of Piraeus

# Acknowledgement
This work was supported in part by the Horizon Europe Research and Innovation Programme of the European Union under grant agreement No. 101070416 (Green.Dat.AI; https://greendatai.eu). In this work, INESC TEC provided the requirements of the business case, as well as access to the FEUP dataset and their Energy benchmarking tool.
