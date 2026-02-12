"""Flower server example."""
import os
import glob
import argparse
import logging
from datetime import datetime

import torch

import flwr as fl
from flwr.common.logger import FLOWER_LOGGER

import strategy as st


# Define global variables
CFG_DATA_DIR = './data'
CFG_FIGURES_DIR = os.path.join(CFG_DATA_DIR, 'fig')
CFG_PICKLE_DIR = os.path.join(CFG_DATA_DIR, 'pkl')
CFG_LOGFILE_NAME = os.path.join(
    CFG_DATA_DIR, 
    'logs', 
    f"fededf_server_{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}.flwr.log"
)


# Create a file handler
file_handler = logging.FileHandler(CFG_LOGFILE_NAME)
file_handler.setLevel(logging.INFO)

# Define log format
formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
file_handler.setFormatter(formatter)

# Add the handler to the logger object of Flower
logger = FLOWER_LOGGER
logger.addHandler(file_handler)

# Avoid duplicated logs if you're adding multiple handlers
logger.propagate = False


if __name__ == "__main__":
    parser = argparse.ArgumentParser(prog='FedEDF Aggregation Server')
    # Arguments related to FL server 
    parser.add_argument('--federation', help='Select Dataset', choices=['dundee', 'porto', 'boulder', 'paloalto'], type=str, required=True)
    parser.add_argument('--load_check', help='Continue from Latest Round', action="store_true")
    # Arguments related to FL clients 
    parser.add_argument('--num_rounds', help='#FL Rounds (default: 30)', default=30, type=int, required=False)
    parser.add_argument('--local_epochs', help='#Local Epochs (default: 3)', default=3, type=int, required=False)
    parser.add_argument('--early_stop', help='Enable Early Stopping mechanism during edge devices\' training', action="store_true")
    parser.add_argument('--patience', help='Patience (#Epochs) for Early Stopping (default: 3)', default=3, type=int)
    # Arguments related to FL connection / aggregation strategy
    parser.add_argument('--conv_channels', help='FedXGBllr 1D-CNN - no. of convolutional channels', default=8, required=False, type=int)
    parser.add_argument('--fc_layers', help='FedXGBllr 1D-CNN - Number/Size of FC Layers', type=str, default='')
    parser.add_argument('--dropout_rate', help='FedXGBllr 1D-CNN - dropout probability', default=0.3, required=False, type=float)
    parser.add_argument('--strategy', help='Aggregation Strategy', choices=['fedprox', 'fedxgbllr'], default='fedprox', required=False, type=str)
    parser.add_argument('--port', help='Server Port', default=8080, type=int, required=False)
    parser.add_argument('--mu', help='Proximal $\mu$', default=0.01, type=float, required=False)
    parser.add_argument('--clients', help='#Clients (default: 2)', default=2, type=int, required=False)
    parser.add_argument('--fraction_fit', help='#clients to train per round (%%)', default=1.0, type=float, required=False)
    parser.add_argument('--fraction_eval', help='#clients to evaluate per round (%%)', default=1.0, type=float, required=False)
    #
    args = parser.parse_args()

    # Generate model names/directories
    global_model_name = f'fededf_{args.federation}.flwr_global.epoch{{0}}.pth'
    model_save_path = os.path.join(
        '.', 'data', 'pth', 
        f'fededf_ver{datetime.now().strftime("%Y-%m-%d_%H-%M-%S")}_{args.federation}_fraction_fit={args.fraction_fit}_fraction_eval={args.fraction_eval}_proximal_mu={args.mu}'
    )
    os.makedirs(model_save_path, exist_ok=True)
    logger.info(f'[server] Saving model at: {model_save_path}')

    # Load latest model (if needed)
    if args.load_check:
        logger.info('[server] Loading from latest checkpoint...')
        latest_global_model_save_path = sorted(
            glob.glob(
                os.path.join(
                    model_save_path, 
                    global_model_name.format('*')
                )
            )
        )[-1]

        model_params = torch.load(
            latest_global_model_save_path
        )
        initial_params = fl.common.ndarrays_to_parameters(
            [val.cpu().numpy() for _, val in model_params['model_state_dict'].items()]
        )

    # Define strategy
    strategy_params = dict(
        save_path = model_save_path,
        model_name = global_model_name,
        load_check = args.load_check,
        early_stop = args.early_stop,
        patience = args.patience,
        logger = logger,
        # 
        model_parameters = model_params if args.load_check else None,
        initial_parameters = initial_params if args.load_check else None,
        # 
        proximal_mu = args.mu,
        local_epochs = args.local_epochs,
        fraction_fit = args.fraction_fit,
        # 
        fraction_evaluate = args.fraction_eval,
        min_available_clients = args.clients,
        min_fit_clients = max(2, int(args.clients * args.fraction_fit)),
        min_evaluate_clients = max(2, int(args.clients * args.fraction_eval))
    )

    if args.strategy == 'fedprox':
        agg_strategy = st.CrossSiloFedProx(
            **strategy_params
        )

        logger.info('[server] Using ```FedProx``` aggregation strategy...')

    elif args.strategy == 'fedxgbllr':
        fedxgbllr__cnn_params = dict(
            num_clients=args.clients,
            in_channels=1,      # The prediction outcomes are treated as a single sequence of values
            conv_channels=args.conv_channels,   
            out_channels=1,     # Represents a single final prediction value, suitable for regression
            fc_layers=args.fc_layers,
            dropout_rate=args.dropout_rate
        )

        agg_strategy = st.FedXGBllr(
            cnn_params=fedxgbllr__cnn_params,
            accept_failures=False,
            **strategy_params
        )

        logger.info('[server] Using ```FedXGBllr``` aggregation strategy...')

    # Start the server
    fl.server.start_server(
        server_address=f"[::]:{args.port}", 
        config=fl.server.ServerConfig(num_rounds=args.num_rounds), 
        strategy=agg_strategy,
    )
