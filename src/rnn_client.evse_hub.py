import os
import sys
import json
import copy
import argparse
import logging
from datetime import datetime

import torch
from torch import nn
import flwr as fl

import helper as hl
import models as ml
import train as tr

torch.manual_seed(10)
torch.autograd.set_detect_anomaly(True)


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
    f"fededf_client_{sys.argv[2]}_{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}.flwr.log"
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
logger = logging.getLogger(f'fededf_client_{sys.argv[2]}')
logger.setLevel(logging.INFO)  # Pass everything to handlers
logger.addHandler(file_handler)
logger.addHandler(console_handler)


class CrossSilo_EDF_RNN(fl.client.NumPyClient):
    def __init__(self, logger, federation_name, cluster_id, device, process_data_params, model_params, evaluate_fun=tr.evaluate_model_multihead_rnn, evaluate_fun_params={}):
        # Related to event logging
        self.logger = logger
        
        # Related to data fetching / processing
        self.federation_name = federation_name
        self.cluster_id = cluster_id
        self.process_data_params = process_data_params

        # Related to model instantiation / training        
        self.model_params = model_params
        self.device, self.evaluate_fun, self.evaluate_fun_params = device, evaluate_fun, evaluate_fun_params
        
        # Generate the embedding layers / token lookup tables
        (
            self.location_embeddings, self.model_embeddings,
            self.location_tokens, self.location_token_lookup,
            self.model_tokens, self.model_token_lookup
        ) = hl.evse_create_embeddings(
            self.federation_name,
            self.process_data_params['dataset_params']
        )
        self.logger.info(f'[init] Created embedding layers and token lookup tables...')

        # Get the training dataset
        _, evse_demand_seq_windows, self.evse_demand_norm_consts = hl.evse_process_data_rnn(
            **{
                'dataset_name': self.federation_name,
                'cluster_id': self.cluster_id,
                'logger':logger,
                **self.process_data_params,
            }
        )
        self.logger.info(f'[init] Loaded train/dev/test dataset...')

        # Generate the DataLoaders
        self.train_loader, self.dev_loader, self.test_loader = hl.evse_create_dataloader(
            self.process_data_params['dataset_params'], 
            evse_demand_seq_windows, 
            self.location_tokens, 
            self.location_token_lookup, 
            self.model_tokens,
            self.model_token_lookup
        )
        self.logger.info(f'[init] Created DataLoaders...')

        # Instantiate EDF model
        self.model = ml.EnergyDemandForecasting_v2(
            **{
                'location_embeddings':self.location_embeddings, 
                'model_embeddings':self.model_embeddings,
                'scale':None,
                **self.model_params
            }
        )
        self.logger.info(f'[init] Created (local)FedEDF instance...')
        self.logger.info(self.model)

        # Send model to designated PyTorch device, and proceed with training as usual
        self.model.to(self.device)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=1e-4)
        self.criterion = tr.PinballLoss(quantile=0.7)
        # 
        rnn_cell_name = 'lstm' if issubclass(self.model_params['rnn_cell'], nn.LSTM) else 'gru'
        rnn_cell_name_is_bi = 'bi-' if self.model_params['bidirectional'] else ''
        self.model_name = f'fededf_{rnn_cell_name_is_bi}{rnn_cell_name}_{{0}}_{self.cluster_id}.flwr_local.round{{1}}.epoch{{2}}.pth'

        # Related to model evaluation (maintain test sets from previous batches)
        self.cluster_id_test_sets = []
    
    def get_parameters(self, **kwargs):
        self.logger.info(f'[get_parameters] Getting (local)FedEDF parameters...')
        return hl.get_parameters(self.model)

    def set_parameters(self, parameters):
        self.logger.info(f'[set_parameters] Setting (global)FedEDF parameters...')
        return hl.set_parameters(self.model, parameters)

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
    

    def fit(self, parameters, config):
        self.logger.info(f'[fit] {config=}')

        # If the train set is empty, return the global parameters as-is
        if self.train_loader is None:
            self.logger.warn(f'[fit] Client #{self.cluster_id} does not have a train set at FL round #{config["round"]}!')
            return parameters, 0, {} 
        
        # Initialize local model with the parameters of the global model
        self.set_parameters(parameters)
        
        model_save_path = os.path.join(
            config['save_path'],
            self.model_name.format(self.federation_name, config['round'], '{0}')
        )
        self.logger.info(f'[fit] Saving model at: {model_save_path}')

        # Perform training for K epochs
        global_model_parameters = [
            torch.tensor(
                copy.deepcopy(parameter),
                requires_grad=False
            ).to(self.device) for parameter in parameters
        ]
        
        criterion_fun, criterion_fun_params, save_current_params, early_stop_params = self.prox_loss, {
            'global_model_parameters':global_model_parameters,
            'proximal_mu':config['proximal_mu']
        }, {
            'path':model_save_path,
            'round':config['round'],
            **{
                f'local_{k}':float(v) for k, v in self.evse_demand_norm_consts.items()
            }
        }, {
            'patience':config['patience'],
            'save_best':False,
        }

        train_losses, dev_losses = tr.train_model(
            self.model, self.device, self.criterion, self.optimizer, config['local_epochs'],
            self.train_loader, self.dev_loader, criterion_fun=criterion_fun, criterion_fun_params=criterion_fun_params, 
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
        return self.get_parameters(), len(self.train_loader.dataset), fit_tldr

    def evaluate(self, parameters, config):        
        # First, if the test set is empty, then return None
        if self.test_loader is None:
            self.logger.warn(f'[evaluate] Client #{self.cluster_id} does not have a test set at FL round #{config["round"]}!')
            return float(0), 0, {} # Return zero loss, zero number of samples, and empty metrics
        
        # Download the parameters of the global model
        self.set_parameters(parameters)

        # Evaluate the global model on the test set of the current FL round
        test_loss, test_avg_err_all, test_avg_err_binned = self.evaluate_fun(
            self.model, self.device, self.criterion, self.test_loader,
            desc=f'[Client #{self.cluster_id}][Round #{config["round"]}] ADE @ Test Set...', **self.evaluate_fun_params
        )
        eval_tldr = dict(
            test_acc=float(test_avg_err_all),
            test_acc_binned=json.dumps(test_avg_err_binned.astype(float).tolist())
        )
        self.logger.info(f'[evaluate] Client #{self.cluster_id} - Round {config["round"]} - {test_loss = }; {test_avg_err_all = }; {test_avg_err_binned = }')

        return float(test_loss), len(self.test_loader.dataset), eval_tldr


if __name__ == '__main__':
    parser = argparse.ArgumentParser(prog='FedEDF Local Worker')
    # Arguments related to edge device
    parser.add_argument('--federation', help='Select Federated Dataset', choices=['dundee', 'porto', 'boulder', 'paloalto'], type=str, required=True)
    parser.add_argument('--cluster_id', help='Cluster ID', type=str, required=True)
    # Arguments from EDF centralized training code
    # # RNN PARAMS
    parser.add_argument('--bi', help='Use Bidirectional RNN', action='store_true')
    parser.add_argument('--rnn_cell', help='Type of RNN Cell', type=str, default='lstm', choices=['lstm', 'gru'])
    parser.add_argument('--hidden_size', help='RNN Hidden Size', type=int, default=350)
    parser.add_argument('--num_layers', help='RNN Number of Layers', type=int, default=1)
    parser.add_argument('--fc_layers', help='Number/Size of FC Layers', type=str, default='150')
    # # DATA PARAMS
    parser.add_argument('--sr_freq', help='Resampling Frequency (default: 24H)', default='24H', type=str)
    parser.add_argument('--min_pts', help='Minimum number of transactions for constructing EVSEs timeseries (default:20 points)', default=20, type=int)
    parser.add_argument('--length', help='Rolling Window Length (default: 14)', default=14, type=int)
    parser.add_argument('--stride', help='Rolling Window Stride (default: 2)', default=2, type=int)
    parser.add_argument('--bs', help='Batch Size', default=1, type=int)
    # Arguments related to FL connection
    parser.add_argument('--port', help='Server Port', default=8080, type=int, required=False)

    # Prepare parameter dict(s)
    args = parser.parse_args()

    col_names = dict(object_name='oid', power_name='power_curr')

    dataset_params = dict(
        dataset_tag=args.federation,
        bs=args.bs,
        sr_freq=args.sr_freq,
        min_pts=args.min_pts,
        njobs=NUM_THREADS,
        data_dir=CFG_DATA_DIR,
        figures_dir=CFG_FIGURES_DIR,
        pickle_dir=CFG_PICKLE_DIR
    )
    dataset_params.update(**col_names)
    logger.info(f'Dataset parameters: {dataset_params}')

    sequencing_params = dict(
        time_axis='timestamp',
        rnn_feats=[
            'day_sin', 'day_cos',
            'hour_sin', 'hour_cos', 
            'week_sin', 'week_cos', 
            #
            'power_curr_logdelta',
            'power_next_step1_extrap',
            # 
            'power_curr_std', 
            'power_curr_ema',
            #
            f'downtime_scaled',
            #
            'no_of_sessions_scaled',
            'charging_time',
            'power_curr', 
        ],
        # Extra features to include in the Fully Connected (FC) layer (excl. ```building``` and ```model```)
        fc_feats=[
            'power_output_kW', 
            # 'is_holiday',
        ], 
        y_feat='power_curr',
    )
    logger.info(f'Sequencing parameters: {sequencing_params}')

    windowing_params = dict(
        length_min=args.length // 2, 
        length_max=args.length,
        stride=args.stride,
        time_axis='timestamp', 
        # 
        rnn_feats=sequencing_params['rnn_feats'], 
        fc_feats=sequencing_params['fc_feats'], 
        y_feats=['power_next'],
    )
    logger.info(f'Windowing parameters: {windowing_params}')
    
    process_cluster_stream_data_params = dict(
        dataset_params=dataset_params, 
        sequencing_params=sequencing_params, 
        windowing_params=windowing_params
    )

    model_params = dict(
        input_size=len(windowing_params['rnn_feats']),
        bidirectional=args.bi,
        rnn_cell=getattr(nn, args.rnn_cell.upper()),
        num_layers=args.num_layers,
        hidden_size=args.hidden_size,
        misc_size=len(windowing_params['fc_feats']),
        fc_layers=[int(i) for i in args.fc_layers.split(',')],
        output_size=len(windowing_params['y_feats'])
    )
    logger.info(f'Model parameters: {model_params}')

    evaluate_fun_params = dict(
        unit='kW',
        metrics_funs=[hl.smape, hl.wape],
    )
        
    device = torch.device('cpu')
    
    client = CrossSilo_EDF_RNN(
        logger=logger,
        federation_name=args.federation,
        cluster_id=args.cluster_id,
        device=device,
        process_data_params=process_cluster_stream_data_params,
        model_params=model_params,
        evaluate_fun=tr.evaluate_model_multihead_rnn,
        evaluate_fun_params=evaluate_fun_params,
    )
    fl.client.start_client(server_address=f"[::]:{args.port}", client=client.to_client())
