import os
from typing import (
    Any, 
    Dict, 
    List, 
    Optional, 
    Tuple, 
    Union
)

import torch
import numpy as np
import xgboost as xgb
import flwr as fl
from flwr.common import (
    GetParametersIns, 
    FitRes, 
    Parameters, 
    Scalar, 
    EvaluateRes
)
from flwr.server.strategy import FedProx
from flwr.server.client_proxy import ClientProxy

import helper as hl


class CrossSiloFedProx(FedProx):
    def __init__(self, save_path, model_name, load_check, early_stop, patience, logger, model_parameters, local_epochs, ndigits=10, **kwargs):
        self.ndigits = ndigits
        self.save_path = save_path
        self.model_name = model_name
        self.early_stop = early_stop
        self.patience = patience
        self.logger = logger
        #
        (
            self.global_charging_time_norm_const,
            self.global_downtime_norm_const,
            self.global_no_of_sessions_norm_const
        ) = (
            np.nan,
            np.nan,
            np.nan
        )
        # 
        self.train_loss_aggregated = []
        self.dev_loss_aggregated = []
        # 
        self.latest_round = 0
        self.local_epochs = local_epochs
        
        if load_check:
            self.latest_round = model_parameters['round'] + 1
            self.train_loss_aggregated = model_parameters['loss']
            self.dev_loss_aggregated = model_parameters['dev_loss']
            self.global_charging_time_norm_const = model_parameters['global_charging_time_norm_const']
            self.global_downtime_norm_const = model_parameters['global_downtime_norm_const']
            self.global_no_of_sessions_norm_const = model_parameters['global_no_of_sessions_norm_const']

        super().__init__(**{
            **kwargs,
            'on_fit_config_fn': self.fit_config_fn,
            'on_evaluate_config_fn': self.evaluate_config_fn
        })

    def fit_config_fn(self, rnd: int):
        return {
            'round': rnd,
            'save_path': self.save_path,
            'local_epochs': self.local_epochs,
            'early_stop': self.early_stop,
            'patience': self.patience,
            'global_charging_time_norm_const':self.global_charging_time_norm_const,
            'global_downtime_norm_const':self.global_downtime_norm_const,
            'global_no_of_sessions_norm_const':self.global_no_of_sessions_norm_const

        }
    
    def evaluate_config_fn(self, rnd: int):
        return {
            'round': rnd,
            'save_path': self.save_path,
            'global_charging_time_norm_const':self.global_charging_time_norm_const,
            'global_downtime_norm_const':self.global_downtime_norm_const,
            'global_no_of_sessions_norm_const':self.global_no_of_sessions_norm_const
        }

    def save_model(self, rnd, aggregated_weights):
        model_dict = dict({
            'parameters':fl.common.parameters_to_ndarrays(aggregated_weights[0]),
            'round': rnd,
            'loss': self.train_loss_aggregated,
            'dev_loss': self.dev_loss_aggregated,
            'global_charging_time_norm_const':self.global_charging_time_norm_const,
            'global_downtime_norm_const':self.global_downtime_norm_const,
            'global_no_of_sessions_norm_const':self.global_no_of_sessions_norm_const
        })
        torch.save(model_dict, os.path.join(self.save_path, self.model_name.format(rnd)))

        return aggregated_weights

    def aggregate_fit(
        self,
        server_round: int,
        results: List[Tuple[ClientProxy, FitRes]],
        failures: List[Union[Tuple[ClientProxy, FitRes], BaseException]],
    ) -> Tuple[Optional[Parameters], Dict[str, Scalar]]:
        curr_round = self.latest_round + server_round
        
        # Filter results for clients that actually contributed to the federation
        valid_results = [result for result in results if result[1].num_examples > 0]

        # If no clients contribute to the federation, maintain current (global) model parameters
        if not valid_results:
            self.logger.warning(f'[strategy][aggregate_fit] Round {curr_round} - No clients contributed in this round.')
            return None, {}
        
        # Call aggregate_fit from base class (FedProx)
        aggregated_weights = super().aggregate_fit(server_round, valid_results, failures)

        # Update aggregated metrics (train/dev loss)
        self.train_loss_aggregated.append(hl.weighted_sum(valid_results, 'train_loss', logger=self.logger))
        self.dev_loss_aggregated.append(hl.weighted_sum(valid_results, 'dev_loss', logger=self.logger))

        # Update data statistics (global scaling/normalization constants)
        self.global_charging_time_norm_const = hl.weighted_sum(
            valid_results, 
            'local_charging_time_norm_const', 
            logger=self.logger
        )
        
        self.global_downtime_norm_const = hl.weighted_sum(
            valid_results, 
            'local_downtime_norm_const', 
            logger=self.logger
        )
        self.global_no_of_sessions_norm_const = hl.weighted_sum(
            valid_results, 
            'local_no_of_sessions_norm_const', 
            logger=self.logger
        )
    
        self.logger.info(f"[strategy][aggregate_fit] Round {curr_round} - global_charging_time_norm_const = {self.global_charging_time_norm_const}")
        self.logger.info(f"[strategy][aggregate_fit] Round {curr_round} - global_downtime_norm_const = {self.global_downtime_norm_const}")
        self.logger.info(f"[strategy][aggregate_fit] Round {curr_round} - global_no_of_sessions_norm_const = {self.global_no_of_sessions_norm_const}")

        # Save aggregated_weights
        self.logger.info(f'[strategy][aggregate_fit] Round {curr_round} - Saving aggregated_weights...')
        self.save_model(curr_round, aggregated_weights)

        return aggregated_weights
    
    def aggregate_evaluate(
        self,
        server_round: int,
        results: List[Tuple[ClientProxy, EvaluateRes]],
        failures: List[Union[Tuple[ClientProxy, EvaluateRes], BaseException]],
    ) -> Tuple[Optional[float], Dict[str, Scalar]]:
        """Aggregate evaluation losses using weighted average."""
        curr_round = self.latest_round + server_round
        
        # Filter results for clients that actually contributed to the federation
        valid_results = [result for result in results if result[1].num_examples > 0]

        # If no clients contribute to the federation, maintain current (global) model parameters
        if not valid_results:
            self.logger.warning(f'[strategy][aggregate_evaluate] Round {curr_round} - No clients contributed in this round.')
            return None, {}

        # Call aggregate_evaluate from base class (FedProx)
        loss_aggregated, metrics = super().aggregate_evaluate(curr_round, valid_results, failures)

        # Weigh (global/binned) accuracy of each client by number of examples used
        accuracy_aggregated_all = hl.weighted_sum(valid_results, 'test_acc', logger=self.logger)
        accuracy_aggregated_binned = hl.weighted_sum(valid_results, 'test_acc_binned', logger=self.logger)
        self.logger.info(f'[strategy][aggregate_evaluate] Round {curr_round} - {loss_aggregated = }; {accuracy_aggregated_all = }; {accuracy_aggregated_binned = }')

        # Output aggregated metrics
        return round(loss_aggregated, ndigits=self.ndigits), {
            **metrics, 
            'accuracy': np.around(accuracy_aggregated_all, decimals=self.ndigits),
            'accuracy_binned': np.around(accuracy_aggregated_binned, decimals=self.ndigits),
        }


class FedXGBllr(CrossSiloFedProx):
    """Configurable FedXGBllr strategy implementation."""

    def __init__(self, cnn_params, save_path, model_name, load_check, early_stop, patience, logger, model_parameters, local_epochs, ndigits=10, **kwargs) -> None:
        """Federated XGBoost [Ma et al., 2023] strategy.

        Implementation based on https://arxiv.org/abs/2304.07537.
        Forked from https://github.com/adap/flower/blob/main/baselines/hfedxgboost/hfedxgboost/strategy.py
        """
        self.aggregated_trees = []
        self.fedxgbllr__cnn_params = cnn_params
        super().__init__(save_path, model_name, load_check, early_stop, patience, logger, model_parameters, local_epochs, ndigits, **kwargs)

    def __repr__(self) -> str:
        """Compute a string representation of the strategy."""
        rep = f'FedXGBllr(accept_failures={self.accept_failures})'
        return rep

    def fit_config_fn(self, rnd: int):
        fitins = super().fit_config_fn(rnd)

        # In first FL round, send the configuration of 1D-CNN
        if rnd == 1:
            return {
                **fitins,
                **self.fedxgbllr__cnn_params
            }
        
        return fitins

    def save_xgb_models(self):
        reconstructed_trees = []
        
        for tree_bytes, cluster_id in zip(self.aggregated_trees[::2], self.aggregated_trees[1::2]):
            tree = xgb.XGBRegressor()
            tree.load_model(bytearray(tree_bytes))
    
            reconstructed_trees.append(
                (
                    tree, 
                    int(cluster_id.decode('utf-8'))
                )
            )
        
        np.save(
            file=os.path.join(
                self.save_path,
                self.model_name[:-4].format('0.xgb_trees')
            ),
            arr=np.array(
                reconstructed_trees,
                dtype='object'
            )
        )

    def aggregate_fit(
        self,
        server_round: int,
        results: List[Tuple[ClientProxy, FitRes]],
        failures: List[Union[Tuple[ClientProxy, FitRes], BaseException]],
    ) -> Tuple[Optional[Parameters], Dict[str, Scalar]]:
        curr_round = self.latest_round + server_round

        # Filter results for clients that actually contributed to the federation
        valid_results = [result for result in results if result[1].num_examples > 0]

        # If no clients contribute to the federation, maintain current (global) model parameters
        if not valid_results:
            self.logger.warning(f'[strategy][aggregate_fit] Round {curr_round} - No clients contributed in this round.')
            return None, {}

        # If its the first FL round, aggregate the XGBoost trees (i.e., combine them into a list of <XGBRegressor, int>)
        if server_round == 1:
            # Sort XGBoost trees, by client identifers
            valid_results = sorted(valid_results, key=lambda l: int(l[1].parameters.tensors[1].decode('utf-8')))

            for _, fit_res in valid_results:
                self.aggregated_trees.extend(fit_res.parameters.tensors)

            # Update aggregated metrics (train/dev loss)
            self.train_loss_aggregated.append(hl.weighted_sum(valid_results, 'train_loss', logger=self.logger))
            self.dev_loss_aggregated.append(hl.weighted_sum(valid_results, 'dev_loss', logger=self.logger))

            # Save aggregated trees
            self.logger.info(f'[strategy][aggregate_fit] Round {curr_round} - Saving aggregated trees...')
            self.save_xgb_models()

            # Aggregate custom metrics if aggregation fn was provided
            metrics_aggregated = {}
            if self.fit_metrics_aggregation_fn:
                fit_metrics = [(res.num_examples, res.metrics) for _, res in results]
                metrics_aggregated = self.fit_metrics_aggregation_fn(fit_metrics)
            elif server_round == 1:  # Only log this warning once
                self.logger.warning("No fit_metrics_aggregation_fn provided")

            return Parameters(tensors=self.aggregated_trees, tensor_type='bytearray'), metrics_aggregated

        # Call aggregate_fit from base class (FedProx)
        aggregated_weights = super().aggregate_fit(server_round, valid_results, failures)

        # Save aggregated_weights
        self.logger.info(f'[strategy][aggregate_fit] Round {curr_round} - Saving aggregated_weights...')
        self.save_model(curr_round, aggregated_weights)

        return aggregated_weights
