import os
import sys
import json
import pathlib
import argparse
import logging
from datetime import datetime

import numpy as np 
import xgboost as xgb
from sklearn.metrics import root_mean_squared_error

import torch
from torch.optim import Adam
from torch.utils.data import DataLoader

import flwr as fl
from flwr.common import (
    Code,
    EvaluateIns,
    EvaluateRes,
    FitIns,
    FitRes,
    GetParametersRes,
    GetParametersIns,
    Status,
    Parameters,
    ndarrays_to_parameters,
    parameters_to_ndarrays,
)

import helper as hl
import models as ml
import train as tr


# SIMULATION ONLY - DELETE ON PROD. VERSION
# Set custom number of threads (e.g., equal to physical cores)
NUM_THREADS = 12
torch.set_num_threads(NUM_THREADS)
torch.set_num_interop_threads(NUM_THREADS)

# Define global variables
CFG_DATA_DIR = './data'
CFG_FIGURES_DIR = os.path.join(CFG_DATA_DIR, 'fig')
CFG_PICKLE_DIR = os.path.join(CFG_DATA_DIR, 'pkl')
CFG_LOGFILE_NAME = os.path.join(
    CFG_DATA_DIR, 
    'logs', 
    f"fededf_xgboost_client_{sys.argv[2]}_{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}.flwr.log"
)

# Configure logging parameters
# Create file handler — logs INFO and above
file_handler = logging.FileHandler(CFG_LOGFILE_NAME)
file_handler.setLevel(logging.INFO)
file_handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))

# Create console handler — logs ERROR and above
console_handler = logging.StreamHandler(sys.stderr)
console_handler.setLevel(logging.ERROR)
console_handler.setFormatter(logging.Formatter('%(levelname)s: %(message)s'))

# Set up root logger manually
logger = logging.getLogger(f'fededf_xgboost_client_{sys.argv[2]}')
logger.setLevel(logging.INFO)  # Pass everything to handlers
logger.addHandler(file_handler)
logger.addHandler(console_handler)


class CrossSilo_EDF_XGBoost(fl.client.Client):
    def __init__(self, logger, federation_name, cluster_id, device, dataset_params, feat_eng_params, xgb_params, cnn_params, evaluate_fun, evaluate_fun_params):
        # Related to event logging
        self.logger = logger
        
        # Related to data fetching / processing
        self.federation_name = federation_name
        self.cluster_id = cluster_id
        self.dataset_params = dataset_params
        self.feat_eng_params = feat_eng_params

        # Related to model instantiation / training        
        self.xgb_params, self.cnn_params = xgb_params, cnn_params

        # Designated PyTorch device to send model to for training purposes
        self.device, self.evaluate_fun, self.evaluate_fun_params = device, evaluate_fun, evaluate_fun_params
        
        # Get the XGBoost training dataset
        (
            self.xgb_train_dataset, 
            self.xgb_dev_dataset, 
            self.xgb_test_dataset, 
            self.evse_demand_norm_consts
        ) = hl.evse_process_data_xgboost(
            **{
                'dataset_name': self.federation_name,
                'cluster_id': self.cluster_id,
                'dataset_params': self.dataset_params,
                'feat_eng_params': self.feat_eng_params,
                'logger':logger,
            }
        )
        self.logger.info(f'[init] Loaded EVSE train/dev/test dataset...')
                
        # Train XGBoost model
        self.xgb_model = xgb.XGBRegressor(
            **xgb_params,
        )
        self.xgb_models = []    # After Round #0 it will be a list of ```xgb_model```

        self.logger.info(f'[init] Training XGBoost model...')
        self.xgb_model.fit(
            **self.xgb_train_dataset,
            eval_set=[
                self.xgb_train_dataset.values(),
                self.xgb_dev_dataset.values(),
            ], 
            verbose=True
        )

        self.xgb_model_score = self.xgb_model.score(**self.xgb_test_dataset)
        self.logger.info(f'[init] XGBoost model R^2 score = {self.xgb_model_score}...')
        
        # Prepare parameters for FedXGBllrCNN; currently we know only the #estimators
        self.cnn_model_params, self.cnn_model = cnn_params, None

        # After Round #0, they will be created from ```self.xgb_*_dataset```
        self.cnn_train_dataloader, self.cnn_dev_dataloader, self.cnn_test_dataloader = None, None, None
        
        self.criterion = tr.PinballLoss(self.xgb_params['quantile_alpha'])
        self.model_name = f'fededf_{"xgb"}_{{0}}_{self.cluster_id}.flwr_local.round{{1}}.epoch{{2}}.pth'
    
    def get_xgb_parameters(self):
        return Parameters(tensors=[bytes(self.xgb_model.get_booster().save_raw()), self.cluster_id.zfill(15).encode('utf-8')], tensor_type='bytearray') 
    
    def get_cnn_parameters(self):
        return ndarrays_to_parameters(hl.get_parameters(self.cnn_model))

    def get_parameters(self, ins:GetParametersIns):
        # If it's the first FL round, ```send``` the ```tree```, along with the ```identifier``` of the client...
        if self.cnn_model is None:
            self.logger.warning(f'The FedXGBllr 1D-CNN model is not yet initialized! First, we need to send the XGBoost models...')
            return GetParametersRes(
                status=Status(Code.OK, ''),
                parameters=Parameters(
                    tensors=[], 
                    tensor_type='numpy.ndarray'
                ) 
            )
        
        # ...otherwise, ```send``` the parameters of the ```1D CNN```
        self.logger.info(f'[get_parameters][Client #{self.cluster_id}] Getting parameters of local FedXGBllr 1D-CNN...')
        return GetParametersRes(
            status=Status(Code.OK, ''),
            parameters=self.get_cnn_parameters()
        )

    def set_xgb_parameters(self, parameters):
        self.logger.info(f'[set_parameters] Setting aggregated XGBoost models...')
        
        aggregated_trees = parameters.tensors
        for tree_bytes, cluster_id in zip(aggregated_trees[::2], aggregated_trees[1::2]):
            tree = xgb.XGBRegressor()
            tree.load_model(bytearray(tree_bytes))
            self.xgb_models.append(
                (
                    tree, 
                    int(cluster_id.decode('utf-8'))
                )
            )
        
        # Sanity Check - Ensure that XGBoost trees are sorted w.r.t. client ID
        is_sorted_xgb_models_ids = np.all(
            np.diff(
                [tree_id for _, tree_id in self.xgb_models]
            ) > 0
        )
        logger.info(f'[set_xgb_parameters] Client #{self.cluster_id} - Received {len(self.xgb_models)} trees; Sorted (w.r.t. ID): {is_sorted_xgb_models_ids}')

    def set_cnn_parameters(self, parameters):
        # Load parameters
        parameters_ndarrays = parameters_to_ndarrays(parameters)

        self.logger.info(f'[set_parameters] Setting global FedXGBllr 1D-CNN parameters...')
        return hl.set_parameters(self.cnn_model, parameters_ndarrays)

    def prox_loss(self, local_model, xb, yb, criterion, *args, **kwargs):
        y_pred = local_model(xb, *args).float()
        
        # proximal_term = 0.0
        proximal_term = torch.tensor(0.0, device=self.device).float()
        
        # Compute the proximal term as the squared L2 norm of weight differences
        for local_weights, global_weights in zip(local_model.parameters(), kwargs['global_model_parameters']):
            proximal_term += (local_weights - global_weights).norm(2).pow(2)
                
        # Calculate total loss with the proximal term scaled by proximal_mu
        loss = criterion(y_pred, yb) + (kwargs['proximal_mu'] / 2) * proximal_term
        return y_pred, loss
    
    def fit(self, ins:FitIns):
        # Unpack FitIns (model parameters, configution values)
        parameters, config = ins.parameters, ins.config
        self.logger.info(f'[fit] {config=}')

        # If it's the first FL round...
        if config['round'] == 1:
            self.logger.info(f'[fit] This is the first round of the federation.')

            self.xgb_model.save_model(
                os.path.join(
                    config['save_path'],
                    f'fededf_{"xgb"}_{self.cluster_id}.flwr_local.round0.json'
                )
            )  # Saves as JSON
            self.logger.info(f'[init] Saving local XGBoost model...')

            # Download FedXGBllr 1D-CNN configuration from the server
            self.cnn_params.update({
                'num_clients':config['num_clients'],
                'in_channels':config['in_channels'],
                'conv_channels':config['conv_channels'],
                'out_channels':config['out_channels'],
            })

            # Instantiate 1D-CNN 
            self.cnn_model = ml.FedXGBllrCNN(**self.cnn_model_params)
            self.logger.info(f'[fit] Created FedXGBllrCNN instance! \n{self.cnn_model=}')

            # Send XGBoost parameters
            return FitRes(
                status=Status(
                    Code.OK, 
                    f'[fit] Client #{self.cluster_id} - Sending XGBoost model to server...'
                ), 
                parameters=self.get_xgb_parameters(), 
                num_examples=len(self.xgb_train_dataset), 
                metrics={
                    'train_loss':float(self.xgb_model.evals_result_['validation_0']['rmse'][-1]),
                    'dev_loss':float(self.xgb_model.evals_result_['validation_1']['rmse'][-1])
                }
            )
        
        if config['round'] > 2:
            # Initialize local model with the parameters of the global model
            self.set_cnn_parameters(parameters)
        
        model_save_path = os.path.join(
            config['save_path'],
            self.model_name.format(self.federation_name, config['round'], '{0}')
        )
        self.logger.info(f'[fit] Saving model at: {model_save_path}')

        # Perform training for K epochs
        global_model_parameters = [parameter.detach().clone() for parameter in self.cnn_model.parameters()]
        
        # optimizer_fun = Adam(self.cnn_model.parameters(), lr=1e-4)  # As used in the RNN-based experiments - for the sake of consistency
        optimizer_fun = Adam(self.cnn_model.parameters(), lr=1e-4, betas=(0.5, 0.999), weight_decay=1e-3)   # Adapted from the FedXGBllr paper
        criterion_fun, criterion_fun_params, save_current_params, early_stop_params = self.prox_loss, {
            'global_model_parameters':global_model_parameters,
            'proximal_mu':config['proximal_mu']
        }, {
            'path':model_save_path,
            'round':config['round'],
        }, {
            'patience':config['patience'],
            'save_best':False,
        }

        # Proceed with FL training of FedXGBllr 1D-CNN model, as usual...
        train_losses, dev_losses = tr.train_model(
            self.cnn_model, self.device, 
            self.criterion, optimizer_fun, config['local_epochs'],
            self.cnn_train_dataloader, self.cnn_dev_dataloader, 
            criterion_fun=criterion_fun, criterion_fun_params=criterion_fun_params, 
            evaluate_cycle=-1, early_stop=config['early_stop'], save_current=True,
            evaluate_fun=self.evaluate_fun, evaluate_fun_params=self.evaluate_fun_params, 
            save_current_params=save_current_params, early_stop_params=early_stop_params
        )

        fit_tldr = dict(
            train_loss=float(train_losses[-1]),
            **(
                {'dev_loss':float(dev_losses[-1])} if dev_losses else {}
            ),
            **{
                f'local_{k}':float(v) for k, v in self.evse_demand_norm_consts.items()
            }
        )

        self.logger.info(f'[fit] Client #{self.cluster_id} - Round {config["round"]} - {train_losses = }')
        self.logger.info(f'[fit] Client #{self.cluster_id} - Round {config["round"]} - {dev_losses = }')

        # Send updated model/metrics to the aggregation server
        return FitRes(
            status=Status(Code.OK, f''), 
            parameters=self.get_cnn_parameters(), 
            num_examples=len(self.cnn_train_dataloader.dataset), 
            metrics=fit_tldr
        )
    
    def evaluate(self, ins:EvaluateIns):   
        # Unpack EvaluateIns (model parameters, configution values)
        parameters, config = ins.parameters, ins.config
        self.logger.info(f'[evaluate] {config=}')
    
        # If it's the first FL round...
        if config['round'] == 1:
            # Receive XGBoost models 
            self.logger.info(f'[evaluate] Round {config["round"]} - Receiving aggregted XGBoost trees...')
            self.set_xgb_parameters(parameters)

            # Create XGB Train / Dev / Test datasets
            logger.info(f'[evaluate] Round {config["round"]} - Creating 1D-CNN train dataset...')
            cnn_train_dataset = hl.fedxgbllr_cnn_create_dataloader(
                self.xgb_models,
                self.xgb_train_dataset,
                self.logger,
                **{
                    'enable_categorical': self.xgb_params['enable_categorical']
                }
            )
            torch.save(cnn_train_dataset, os.path.join(config['save_path'], f'cnn_train_dataset.client{self.cluster_id}.pth'))

            logger.info(f'[evaluate] Round {config["round"]} - Creating 1D-CNN dev dataset...')
            cnn_dev_dataset = hl.fedxgbllr_cnn_create_dataloader(
                self.xgb_models,
                self.xgb_dev_dataset,
                self.logger,
                **{
                    'enable_categorical': self.xgb_params['enable_categorical']
                }
            )
            torch.save(cnn_dev_dataset, os.path.join(config['save_path'], f'cnn_dev_dataset.client{self.cluster_id}.pth'))
            
            logger.info(f'[evaluate] Round {config["round"]} - Creating 1D-CNN test dataset...')
            cnn_test_dataset = hl.fedxgbllr_cnn_create_dataloader(
                self.xgb_models,
                self.xgb_test_dataset,
                self.logger,
                **{
                    'enable_categorical': self.xgb_params['enable_categorical']
                }
            )
            torch.save(cnn_test_dataset, os.path.join(config['save_path'], f'cnn_test_dataset.client{self.cluster_id}.pth'))

            self.cnn_train_dataloader = DataLoader(cnn_train_dataset, batch_size=self.dataset_params['bs'], shuffle=True)
            self.cnn_dev_dataloader = DataLoader(cnn_dev_dataset, batch_size=self.dataset_params['bs'], shuffle=False)
            self.cnn_test_dataloader = DataLoader(cnn_test_dataset, batch_size=self.dataset_params['bs'], shuffle=False)
            
            # Return metrics of XGBoost models
            return EvaluateRes(
                status=Status(Code.OK, ''),
                loss=float(
                    self.criterion(
                        y_hat := torch.from_numpy(
                            self.xgb_model.predict(self.xgb_test_dataset['X'])
                        ), 
                        y := torch.from_numpy(
                            self.xgb_test_dataset['y'].values
                        )
                    )
                ),
                num_examples=len(self.xgb_test_dataset['X']),
                metrics = dict(
                    test_acc=float(root_mean_squared_error(y, y_hat)),
                    test_acc_binned=json.dumps(
                        np.array(
                            [
                                metrics_fun(y.numpy(), y_hat.numpy()) for metrics_fun in self.evaluate_fun_params['metrics_funs']
                            ]
                        ).astype(float).tolist()
                    )
                )
            )
        
        # Download the parameters of the global FedXGBllr 1D-CNN model
        self.set_cnn_parameters(parameters)

        # Evaluate the global model on the test set of the current FL round -- NEEDS REVISION        
        test_loss, test_avg_err_all, test_avg_err_binned = self.evaluate_fun(
            self.cnn_model, self.device, self.criterion, self.cnn_test_dataloader,
            desc=f'[Client #{self.cluster_id}][Round #{config["round"]}] ADE @ Test Set...', **self.evaluate_fun_params
        )
        eval_tldr = dict(
            test_acc=float(test_avg_err_all),
            test_acc_binned=json.dumps(test_avg_err_binned.astype(float).tolist())
        )
        self.logger.info(f'[evaluate] Client #{self.cluster_id} - Round {config["round"]} - {test_loss = }; {test_avg_err_all = }; {test_avg_err_binned = }')

        # Send updated eval metrics to the aggregation server
        return EvaluateRes(
            status=Status(Code.OK, f''), 
            loss=float(test_loss),
            num_examples=len(self.cnn_test_dataloader.dataset), 
            metrics=eval_tldr
        )


if __name__ == '__main__':
    parser = argparse.ArgumentParser(prog='FedEDF Local Worker')
    # Arguments related to edge device
    parser.add_argument('--federation', help='Select Federated Dataset', choices=['dundee', 'porto', 'boulder', 'paloalto'], type=str, required=True)
    parser.add_argument('--cluster_id', help='Cluster ID', type=str, required=True)
    # # XGBOOST PARAMS
    parser.add_argument('--n_estimators', help='Number of trees (estimators) in the XGBoost model', type=int, default=50)
    # parser.add_argument('--max_depth', help='Maximum depth of a tree', type=int, default=5)
    # # DATA PARAMS
    parser.add_argument('--sr_freq', help='Resampling Frequency (default: 24H)', default='24H', type=str)
    parser.add_argument('--min_pts', help='Minimum number of transactions for constructing EVSEs timeseries (default:20 points)', default=20, type=int)
    parser.add_argument('--bs', help='Batch Size', default=1, type=int)
    # Arguments related to FL connection
    parser.add_argument('--port', help='Server Port', default=8080, type=int, required=False)

    # Prepare parameter dict(s)
    args = parser.parse_args()

    # Set random seed for PyTorch / Hyperparameter selection
    np.random.seed(int(args.cluster_id))
    torch.manual_seed(int(args.cluster_id))
    torch.autograd.set_detect_anomaly(True)

    dataset_params = dict(
        object_name='oid', 
        power_name='power_curr',
        dataset_tag=args.federation,
        sr_freq=args.sr_freq,
        min_pts=args.min_pts,
        njobs=NUM_THREADS,
        data_dir=CFG_DATA_DIR,
        figures_dir=CFG_FIGURES_DIR,
        pickle_dir=CFG_PICKLE_DIR,
        bs=args.bs
    )
    logger.info(f'Dataset parameters: {dataset_params}')

    fedxgbllr__feat_eng_params = dict(
        rolling__window = 6,
        rolling__min_periods = 4,
        time_axis='timestamp',
        X_feats=[
            'power_output_kW',
            'no_of_sessions',
            'charging_time',
            # 
            'week_sin', 'week_cos', 
            'day_sin', 'day_cos',
            'hour_sin', 'hour_cos',
            # 
            'power_curr_lag1',
            'power_curr', 
            'power_curr_logdelta',
            'power_next_step1_extrap', 
            #  
            'power_curr_std', 
            'power_curr_ema', 
            # 
            'power_curr_lag5', 'power_curr_lag48',
            'power_curr_ema_lag24', 'power_curr_ema_lag48',
            'power_curr_std_lag24', 'power_curr_std_lag48',
            #
            f'downtime_scaled',
        ],
        y_feat='power_next',
    )
    logger.info(f'Preprocessing / Feature Engineering parameters: {fedxgbllr__feat_eng_params}')

    # Diverse parameters per client
    fedxgbllr__xgb_params = dict(
        objective="reg:quantileerror", 
        quantile_alpha=0.7, 
        eval_metric=["rmse"],
        n_estimators=args.n_estimators,    # WARNING: The 1D-CNN dataset will consist of ```clients``` * ```n_estimators``` features - be careful on memory consumption
        learning_rate=np.exp(np.random.uniform(np.log(1e-2), np.log(5e-2))),  # Log-uniform distribution
        # max_depth=np.random.randint(3, 5),
        subsample=np.random.uniform(0.6, 1.0),
        colsample_bytree=np.random.uniform(0.5, 1.0),
        min_child_weight = np.random.randint(5, 20),
        reg_lambda=np.exp(np.random.uniform(np.log(1e-1), np.log(10.0))),  # Log-uniform distribution,
        reg_alpha=np.exp(np.random.uniform(np.log(1e-1), np.log(1.0))),  # Log-uniform distribution
        random_state=int(args.cluster_id), 
        early_stopping_rounds=None, 
        enable_categorical=True,
        n_jobs=NUM_THREADS, 
    )

    fedxgbllr__cnn_params = dict(
        trees_per_client=args.n_estimators
    )

    evaluate_fun_params = dict(
        unit='kW',
        metrics_funs=[hl.smape, hl.wape],
    )

    client = CrossSilo_EDF_XGBoost(
        logger=logger,
        federation_name=args.federation,
        cluster_id=args.cluster_id,
        device=torch.device('cpu'),
        dataset_params=dataset_params,
        feat_eng_params=fedxgbllr__feat_eng_params,
        xgb_params=fedxgbllr__xgb_params,
        cnn_params=fedxgbllr__cnn_params,
        evaluate_fun=tr.evaluate_model_multihead_1dcnn,
        evaluate_fun_params=evaluate_fun_params,
    )
    fl.client.start_client(server_address=f"[::]:{args.port}", client=client)
