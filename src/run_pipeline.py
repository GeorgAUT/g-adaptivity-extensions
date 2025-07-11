import os
import time
import yaml
import torch
import wandb

from src.params import get_params
from src.utils_data import make_data_name
from src.utils_eval import evaluate_model
from src.utils_eval_plotting import plot_trained_dataset_2d, plot_trained_dataset_3d
from src.run_GNN import main as run_GNN, get_data
from models.mesh_adaptor_model import Mesh_Adaptor
from models.direct_optimisation import backFEM_2D


def recover_config(run):
    # Get config directly from the run object
    raw_config = {k: v for k, v in run.config.items()}

    # Extract actual values from the nested config structure
    opt = {k: v['value'] if isinstance(v, dict) and 'value' in v else v
           for k, v in raw_config.items()}

    print(f"Recovered config:")
    print(yaml.dump(opt))
    return opt

def load_model_from_path(old_opt, model_name="model_best.pt"):

    # Load model state dict
    if old_opt['wandb_load_model'] == 'wandb':
        model_path = old_opt['wandb_model_path']
        # Load from wandb
        api = wandb.Api()
        run = api.run(model_path)
        model_file = run.file(model_name).download(replace=True)
        state_dict = torch.load(model_file.name)
        # Always get config from wandb when run_id is provided
        opt = recover_config(run)

    elif old_opt['wandb_load_model'] == 'local':
        model_path = old_opt['local_model_path']

        # Load from local path
        state_dict = torch.load(model_path)
        # Load config from local yaml
        config_path = os.path.join(os.path.dirname(model_path), 'config.yaml')
        if os.path.exists(config_path):
            with open(config_path, 'r') as f:
                opt = yaml.safe_load(f)
        else:
            raise ValueError(
                f"No config.yaml found in {os.path.dirname(model_path)}. Cannot load model without configuration.")

    dim = len(old_opt['mesh_dims'])
    opt['wandb'] = False
    model = Mesh_Adaptor(opt, gfe_in_c=dim + 1, lfe_in_c=dim + 1, deform_in_c=dim + 1).to(opt['device'])

    model.load_state_dict(state_dict)

    return model, opt

def get_model_opt(opt):
    if opt['wandb_load_model']:
        model, opt = load_model_from_path(opt, model_name="model_best.pt")
        
    elif opt['model'] in ['fixed_mesh_1D', 'backFEM_1D', 'fixed_mesh_2D', 'backFEM_2D']:
        if opt['model'] == 'fixed_mesh_1D':
            raise NotImplementedError(f"Model {opt['model']} is no longer supported.")
        elif opt['model'] == 'backFEM_1D':
            raise NotImplementedError(f"Model {opt['model']} is no longer supported.")
        elif opt['model'] == 'fixed_mesh_2D':
            raise NotImplementedError(f"Model {opt['model']} is no longer supported.")
        elif opt['model'] == 'backFEM_2D':
            model = backFEM_2D(opt)
    else:
        model = run_GNN(opt)

    return model, opt


def main(opt):
    if torch.backends.mps.is_available():
        opt['device'] = torch.device('cpu') #mps
    elif torch.cuda.is_available():
        opt['device'] = torch.device('cpu')
    else:
        opt['device'] = torch.device('cpu')

    # get trained model and dataset
    opt = make_data_name(opt, train_test="train")

    if opt['wandb']:
        if opt['wandb_offline']:
            os.environ["WANDB_MODE"] = "offline"
        else:
            os.environ["WANDB_MODE"] = "run"

        if 'wandb_run_name' in opt.keys():
            wandb_run = wandb.init(entity=opt['wandb_entity'], project=opt['wandb_project'], group=opt['wandb_group'],
                                   name=opt['wandb_run_name'], reinit=True, config=opt, allow_val_change=True)
        else:
            wandb_run = wandb.init(entity=opt['wandb_entity'], project=opt['wandb_project'], group=opt['wandb_group'],
                                   reinit=True, config=opt, allow_val_change=True)
        opt = wandb.config

    else:
        os.environ["WANDB_MODE"] = "disabled"

    test_dataset = get_data(opt, train_test="test")

    start_train_time = time.time()
    model, opt = get_model_opt(opt)

    model.eval()
    end_train_time = time.time()

    start_eval_time = time.time()

    results_df, times_df, df_metrics = evaluate_model(model, test_dataset, opt)
    end_eval_time = time.time()

    if opt['pde_type'] == "Poisson" and (opt['show_dataset_plots'] or opt['wandb_log_plots']):
        if len(opt['mesh_dims']) == 3:
            plot_trained_dataset_3d(test_dataset, model, opt, show_mesh_evol_plots=opt['show_mesh_evol_plots'])
        else:
            plot_trained_dataset_2d(test_dataset, model, opt, show_mesh_evol_plots=opt['show_mesh_evol_plots'])

    if opt['wandb']:
        results_table = wandb.Table(dataframe=results_df.round(4))
        results_table_describe = wandb.Table(dataframe=results_df.describe().reset_index().round(4))
        times_table = wandb.Table(dataframe=times_df.round(6))
        times_table_describe = wandb.Table(dataframe=times_df.describe().reset_index().round(6))
        metrics_table = wandb.Table(dataframe=df_metrics.round(4))
        metrics_table_describe = wandb.Table(dataframe=df_metrics.describe().reset_index().round(4))

        wandb.log({"results_table": results_table,
                    "results_table_describe": results_table_describe,
                    "times_table": times_table,
                    "times_table_describe": times_table_describe,
                    "metrics_table": metrics_table,
                    "metrics_table_describe": metrics_table_describe})

        #average results over rows, convert to dict using only the suffix of the column name
        results_dict = {f"{key.split('.')[-1]}": results_df.mean()[key] for key in results_df.columns}
        times_dict = {f"{key.split('.')[-1]}": times_df.mean()[key] for key in times_df.columns}
        metrics_dict = {f"{key.split('.')[-1]}": df_metrics.mean()[key] for key in df_metrics.columns}
        headline_results = {**results_dict,
                            **times_dict,
                            **metrics_dict,
                       "train_time": end_train_time - start_train_time,
                       "eval_time": end_eval_time - start_eval_time}
        wandb.log(headline_results)

    if opt['wandb']:
        wandb.finish()

    return results_df, times_df, df_metrics


if __name__ == "__main__":
    opt = get_params()
    main(opt)
