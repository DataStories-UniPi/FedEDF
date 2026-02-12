import pandas as pd
import numpy as np
import tqdm
import time

from sklearn.preprocessing import FunctionTransformer
import torch
import torch.nn as nn

import models as ml


ROUND_DECIMALS = 5


class PinballLoss(nn.Module):
    def __init__(self, quantile: float):
        """
        Initialize the Pinball Loss with a specific quantile.
        
        Args:
        quantile (float): The quantile to be used in the Pinball Loss. 
                          For median, use quantile = 0.5.
        """
        super(PinballLoss, self).__init__()
        self.quantile = quantile

    def forward(self, y_pred, y_true):
        """
        Compute the Pinball Loss between predicted and true values.
        
        Args:
        y_pred (Tensor): Predicted values (output of the model).
        y_true (Tensor): Ground truth values.
        
        Returns:
        Tensor: The computed Pinball Loss.
        """
        # Error is the difference between predicted and actual values
        error = y_true - y_pred
        
        # Pinball loss calculation
        loss = torch.max(self.quantile * error, (self.quantile - 1) * error)
        
        # Return the mean loss
        return torch.mean(loss)

    def __repr__(self):
        return f"PinballLoss_q{int(self.quantile*100)}"


def save_model(model, path, **kwargs):
    torch.save({
        'model_state_dict': model.state_dict(),
        **kwargs
    }, path)


def calc_loss(model, xb, yb, criterion, *args, **kwargs):
    y_pred = model(xb, *args)
    # loss = criterion(y_pred, yb.squeeze(dim=-2))
    loss = criterion(y_pred, yb)
    return y_pred, loss
    

def model_backprop(model, xb, yb, criterion, optimizer, criterion_fun, *args, **kwargs):
    try:
        optimizer.zero_grad()
        _, loss = criterion_fun(model, xb, yb, criterion, *args, **kwargs)

        loss.backward()
        optimizer.step()
    except RuntimeError as err_runtime:
        print(err_runtime)
        # pdb.set_trace()
    return loss


def running_loss(loss, data_loader):
    loss = torch.Tensor(loss).sum()
    loss = loss / len(data_loader)
    return loss


def model_dev_loss(model, device, criterion, dev_loader, criterion_fun, criterion_fun_params):
    model.eval()
    with torch.no_grad():
        dev_loss = []
        for (xb, yb, *args) in (pbar := tqdm.tqdm(dev_loader, leave=False, total=len(dev_loader), dynamic_ncols=True)):
            xb = xb.to(device).float()
            yb = yb.to(device).float()
            
            if isinstance(model, ml.FedXGBllrCNN):
                _, loss = criterion_fun(model, xb, yb, criterion, *args, **criterion_fun_params)
            else:
                lb, *args_rest = args
                args_rest = (arg.to(device) for arg in args_rest)
                _, loss = criterion_fun(model, xb, yb, criterion, lb, *args_rest, **criterion_fun_params)
            
            dev_loss.append(loss)
            pbar.set_description(f'Dev Loss: {loss:.{ROUND_DECIMALS}f}')

    return running_loss(dev_loss, dev_loader)


def train_step(model, device, criterion, optimizer, train_loader, criterion_fun, criterion_fun_params):
    model.train()

    train_loss = []
    for j, (xb, yb, *args) in (pbar := tqdm.tqdm(enumerate(train_loader), leave=False, total=len(train_loader), dynamic_ncols=True)):
        xb = xb.to(device).float()
        yb = yb.to(device).float()

        if isinstance(model, ml.FedXGBllrCNN):
            tr_loss = model_backprop(model, xb, yb, criterion, optimizer, criterion_fun, *args, **criterion_fun_params)
        else:
            lb, *args_rest = args
            args_rest = (arg.to(device) for arg in args_rest)
            tr_loss = model_backprop(model, xb, yb, criterion, optimizer, criterion_fun, lb, *args_rest, **criterion_fun_params)
        
        train_loss.append(tr_loss)
        pbar.set_description(f'Train Loss: {tr_loss:.{ROUND_DECIMALS}f}')

    return running_loss(train_loss, train_loader)


def early_stopping(n_epochs_stop, min_loss, curr_loss, patience=5, min_delta=1e-4, save_best=False, **kwargs):
    if (min_loss - curr_loss) > min_delta:
        if save_best:
            print(f'Loss Decreased ({min_loss:.{ROUND_DECIMALS}f} -> {curr_loss:.{ROUND_DECIMALS}f}). Saving Model...', end=' ')
            save_model(**kwargs)
            print('Done!')

        return 0, curr_loss, False

    print(f'Loss Increased ({min_loss:.{ROUND_DECIMALS}f} -> {curr_loss:.{ROUND_DECIMALS}f}).')
    n_epochs_stop_ = n_epochs_stop + 1
    return n_epochs_stop_, min_loss, n_epochs_stop_ == patience


def evaluate_model_multihead_rnn(model, device, criterion, test_loader, display_acc=True,
                                #   bins=np.arange(0, 1801, 300),
                                 metrics_funs=[], desc=None, **kwargs):
    y_trues, y_preds = [], []
    errs, losses = [], []

    model.eval()
    with torch.no_grad():
        for xb, yb, *args in (pbar := tqdm.tqdm(test_loader, leave=False, desc=desc, total=len(test_loader), dynamic_ncols=True)):
            # print(f'{xb.shape=}\t {yb.shape=}\t {lb.shape=}')
            xb, yb = xb.to(device), yb.to(device)    # Model Inference

            lb, *args_rest = args
            args_rest = (arg.to(device) for arg in args_rest)
            y_pred = model(xb.float(), lb, *args_rest).detach()

            # pdb.set_trace()
            y_trues.append(yb.detach().cpu())
            y_preds.append(y_pred.cpu())
            
            errs.append(
                pd.DataFrame(
                    np.linalg.norm(yb.cpu() - y_pred.cpu(), axis=-1), 
                    columns=range(1, y_pred.shape[1]+1) 
                )
            )

            losses.append(eval_loss := criterion(y_pred, yb))
            pbar.set_description(f'{desc}: {eval_loss:.{ROUND_DECIMALS}f}')

    test_loss = running_loss(losses, test_loader)
    avg_disp_err = pd.concat(errs).describe().loc['mean']

    # If y_trues is 2D (<#samples, #steps>), then it is implied that it consists of 1 feature, 
    # therefore it is transformed into <#samples, #steps, 1>
    y_trues, y_preds = np.concatenate(y_trues), np.concatenate(y_preds)

    if len(y_trues.shape) == 2:
        y_trues, y_preds = (
            np.expand_dims(y_trues, axis=-1),
            np.expand_dims(y_preds, axis=-1)
        )

    if display_acc:
        # Avg. Loss | Avg. Disp. Error (@look-ahead)
        print(f'Loss: {test_loss:.{ROUND_DECIMALS}f} | ', f'Accuracy (RMSE): {np.average(avg_disp_err):.{ROUND_DECIMALS}f} |', '; '.join(f'{i:.{ROUND_DECIMALS}f}' for i in avg_disp_err.values.tolist()), f'{kwargs.pop("unit", "m")}')

        metrics_fun_res = []
        if metrics_funs:
            for metric_fun in metrics_funs:
                metric_fun_res = metric_fun(y_trues, y_preds).squeeze(-1)
                metrics_fun_res.append(metric_fun_res)
                print(f'{metric_fun.__name__.upper()}: {np.mean(metric_fun_res):.{ROUND_DECIMALS}f} | ', '; '.join(f'{i:.{ROUND_DECIMALS}f}' for i in metric_fun_res))

    return test_loss, np.average(avg_disp_err), np.array(metrics_fun_res)


def evaluate_model_multihead_1dcnn(model, device, criterion, test_loader, display_acc=True,
                                #   bins=np.arange(0, 1801, 300),
                                 metrics_funs=[], desc=None, **kwargs):
    y_trues, y_preds = [], []
    errs, losses = [], []

    model.eval()
    with torch.no_grad():
        for xb, yb, *args in (pbar := tqdm.tqdm(test_loader, leave=False, desc=desc, total=len(test_loader), dynamic_ncols=True)):
            # print(f'{xb.shape=}\t {yb.shape=}\t {lb.shape=}')
            xb, yb = xb.to(device), yb.to(device)    # Model Inference
            y_pred = model(xb.float(), *args).detach()

            y_trues.append(yb.unsqueeze(0).detach().cpu())
            y_preds.append(y_pred.unsqueeze(0).cpu())
            
            # pdb.set_trace()
            errs.append(
                pd.DataFrame(
                    np.linalg.norm(y_trues[-1] - y_preds[-1], axis=0), 
                    columns=range(1, y_pred.shape[1]+1)
                )
            )

            losses.append(eval_loss := criterion(y_pred, yb))
            pbar.set_description(f'{desc}: {eval_loss:.{ROUND_DECIMALS}f}')
    
    test_loss = running_loss(losses, test_loader)
    avg_disp_err = pd.concat(errs).describe().loc['mean']

    # Merge input arrays (#batches, #samples, #features) 
    # and squeeze to 2D (#samples, #features) for metric calculation
    # pdb.set_trace()
    y_trues, y_preds = np.hstack(y_trues).squeeze(axis=0), np.hstack(y_preds).squeeze(axis=0)
    
    if display_acc:
        # Avg. Loss | Avg. Disp. Error (@look-ahead)
        print(f'Loss: {test_loss:.{ROUND_DECIMALS}f} | ', f'Accuracy (RMSE): {np.average(avg_disp_err):.{ROUND_DECIMALS}f} |', '; '.join(f'{i:.{ROUND_DECIMALS}f}' for i in avg_disp_err.values.tolist()), f'{kwargs.pop("unit", "m")}')

        metrics_fun_res = []
        if metrics_funs:
            for metric_fun in metrics_funs:
                # pdb.set_trace()
                metric_fun_res = metric_fun(y_trues, y_preds)
                metrics_fun_res.append(metric_fun_res)
                print(f'{metric_fun.__name__.upper()}: {np.mean(metric_fun_res):.{ROUND_DECIMALS}f} | ', '; '.join(f'{i:.{ROUND_DECIMALS}f}' for i in metric_fun_res))

    return test_loss, np.average(avg_disp_err), np.array(metrics_fun_res)


def train_model(model, device, criterion, optimizer, n_epochs,
                train_loader, dev_loader, criterion_fun=calc_loss, criterion_fun_params={}, evaluate_cycle=5, early_stop=True, save_current=True,
                evaluate_fun=evaluate_model_multihead_rnn, evaluate_fun_params={}, early_stop_params={}, save_current_params={}, trial=None):
    train_losses, dev_losses = [], []

    # Early Stopping Initial Param. Values
    min_loss, n_epochs_stop, stop = early_stop_params.pop('min_loss', torch.tensor(float("Inf"))), 0, False

    if save_current:
        save_path_template = save_current_params['path']

    # training loop
    for i in range(n_epochs):
        t_start = time.process_time()
        train_loss = train_step(model, device, criterion, optimizer, train_loader, criterion_fun, criterion_fun_params)
        dev_loss = model_dev_loss(model, device, criterion, dev_loader, criterion_fun, criterion_fun_params)
        t_end = time.process_time() - t_start

        train_losses.append(train_loss.numpy())
        dev_losses.append(dev_loss.numpy())

        epoch_summary = {
            'model': model,
            'epoch': i,
            # Save the scaler only if it exists, otherwise save as scaler the identity function
            'scaler': getattr(train_loader.dataset, 'scaler', FunctionTransformer(None)),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': train_losses,
            'dev_loss': dev_losses
        }

        if early_stop:
            early_stop_params.update(epoch_summary)
            n_epochs_stop, min_loss, stop = early_stopping(n_epochs_stop, min_loss, dev_loss, **early_stop_params)
        
        if save_current:
            save_current_params.update(epoch_summary)
            save_current_params['path'] = save_path_template.format(i)
            save_model(**save_current_params)

        print(f'Epoch #{i+1}/{n_epochs} | '
              f'Train Loss: {train_loss:.{ROUND_DECIMALS}f} | '
              f'Validation Loss: {dev_loss:.{ROUND_DECIMALS}f} | '
              f'Time Elapsed: {t_end:.{ROUND_DECIMALS}f}')

        if evaluate_cycle != -1 and i % evaluate_cycle == 0 and evaluate_fun is not None:
            _ = evaluate_fun(
                model, device, criterion, dev_loader,
                desc='Eval. @ Dev Set...', **evaluate_fun_params
            )

        if stop:
            print(f'Training Stopped at Epoch #{i+1}')
            break

    return train_losses, dev_losses
