import argparse
import os
import numpy as np
import random
import torch
import yaml


def run_params(opt):
    """Set run parameters from YAML config.
    exp_config will overwrite base_config.
    """
    yaml_opts = load_yaml_config(opt['exp_config'], opt['base_config'])
    opt.update(yaml_opts)

    return opt

def t_or_f(tf_str):
    if tf_str == "True" or tf_str == "true" or (type(tf_str) == bool and tf_str):
        return True
    elif tf_str == "False" or tf_str == "false" or (type(tf_str) == bool and not tf_str):
        return False
    else:
        return tf_str

def tf_sweep_args(opt):
    for arg in list(opt.keys()):
        str_tf = opt[arg]
        bool_tf = t_or_f(str_tf)
        opt[arg] = bool_tf
    return opt

def set_seed(seed: int = 42) -> None:
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ["PYTHONHASHSEED"] = str(seed)
    print(f"Random seed set as {seed}")

def get_arg_list(arg_list):
    print(arg_list, len(arg_list), type(arg_list))
    if type(arg_list[0]) == int:
        pass
    else:
        arg_list = eval(arg_list[0]) #fix to deal with list of args input
    return arg_list

def load_yaml_config(config_name: str = None, base_config_name: str = 'base_config') -> dict:
    config = {}
    
    def _load_config(name):
        config_path = name
        
        if not os.path.exists(config_path):
            print(f"Warning: Config file not found: {config_path}")
            return {}
        
        with open(config_path, 'r') as f:
            yaml_config = yaml.safe_load(f)
            
        if yaml_config is None:
            print(f"Warning: Config file is empty: {config_path}")
            return {}
            
        flat_config = {}
        for section in yaml_config:
            if isinstance(yaml_config[section], dict):
                for key, value in yaml_config[section].items():
                    flat_config[key] = value
            else:
                flat_config[section] = yaml_config[section]
                
        return flat_config
    
    # Load base config first
    if base_config_name:
        config.update(_load_config(base_config_name))
    
    # Then load experiment-specific config
    if config_name:
        config.update(_load_config(config_name))
    
    return config


def get_params():
    parser = argparse.ArgumentParser()
    
    # Config path arguments
    parser.add_argument('--base_config', type=str, default='./configs/base_config.yaml',
                       help='Base YAML config file (without .yaml extension)')
    parser.add_argument('--exp_config', type=str, default=None, 
                       help='Experiment-specific YAML config file (without .yaml extension)')

    #data params
    parser.add_argument('--data_dir', type=str, default='./data', help="data directory")
    parser.add_argument('--dataset', type=str, default='grid', choices=['fd_mmpde_1d','fd_ma_2d'], help="high level data type")
    parser.add_argument('--data_type', type=str, default='randg_mix', choices=['all', 'structured', 'randg', 'randg_mix', 'RBF'], help="data desriptor")
    parser.add_argument('--data_name', type=str, default='test', help="data path desriptor")
    parser.add_argument('--num_train', type=int, default=100, help="number of training data points")
    parser.add_argument('--num_test', type=int, default=25, help="number of test data points")

    #mixed res data params
    parser.add_argument('--mesh_dims_train', nargs='+', default=[[15, 15], [20, 20]], help='dimensions of mesh - width, height')
    parser.add_argument('--mesh_dims_test', nargs='+', default=[[i, i] for i in range(12, 24, 1)], help='dimensions of mesh - width, height')
    parser.add_argument('--num_gauss_range', nargs='+', default=[2, 3], help='number of Gaussians in u')
    parser.add_argument('--test_frac', type=float, default=None, help="fraction of test data")
    parser.add_argument('--use_patches', type=bool, default=False, help="use patches")

    #mesh params
    parser.add_argument('--mesh_geometry', type=str, default='rectangle')#, choices=['rectangle','cylinder_100','polygon_010','cylinder015','cylinder010','cylinder_100_025','cylinder_100_050'])
    parser.add_argument('--mesh_dims', nargs='+', default=[15, 15], help='dimensions of mesh - width, height')
    parser.add_argument('--mesh_file_type', type=str, default='bin', help="mesh file type", choices=['hdf5', 'bin'])
    parser.add_argument('--monitor_type', type=str, default='monitor_hessian', help="monitor type")
    parser.add_argument('--fix_boundary', type=str, default="True", help="fix boundary nodes")
    parser.add_argument('--mon_reg', type=float, default=0.1, help="regularisation term in MMPDE5")
    parser.add_argument('--mon_power', type=float, default=0.2, help="power term in MMPDE5")
    parser.add_argument('--monitor_alpha', type=int, default=5, help='monitor_alpha')
    parser.add_argument('--mesh_scale', type=float, default=1.0, help="scale of mesh for patching")

    #pde params
    parser.add_argument('--pde_type', type=str, default='Poisson', choices=['Poisson', 'Burgers', 'NavierStokes'], help="PDE type")
    parser.add_argument('--boundary', type=str, default='dirichlet')
    parser.add_argument('--num_gauss', type=int, default=2, help='number of Gaussians in u')
    parser.add_argument('--anis_gauss', type=bool, default=False, help='whether Gaussians in u are anisotropic')
    parser.add_argument('--rand_gauss', type=bool, default=False, help='whether Gaussians in u random c/s')
    parser.add_argument('--scale', type=float, default=0.2, help="variance of Gaussian solution u")
    parser.add_argument('--amplitude_rescale', type=float, default=0.6)
    parser.add_argument('--center', type=float, default=0.5, help="center of Gaussian solution u")
    parser.add_argument('--U_mean', type=float, default=1.0, help="Mean flow in Navier-Stokes")
    parser.add_argument('--use_mpi_optimized', type=t_or_f, default=False, help="Use MPI-optimized solver parameters")

    # Burgers params
    parser.add_argument('--nu', type=float, default=0.001)
    parser.add_argument('--timestep', type=float, default=0.02)
    parser.add_argument('--num_time_steps', type=int, default=10)

    #fem params
    parser.add_argument('--eval_quad_points', type=int, default=101, help='number of quad points')
    parser.add_argument('--stiff_quad_points', type=int, default=3, help='number of quad points per interval')
    parser.add_argument('--eval_refinement_level', type=int, default=2, help='number of quad points')
    parser.add_argument('--load_quad_points', type=int, default=101, help='number of quad points per interval')
    parser.add_argument('--HO_degree', type=int, default=4, help='higher order degree of loss/evaluation')

    #model params
    parser.add_argument('--model', type=str, default='MeshAdaptor', choices=['fixed_mesh_1D','fixed_mesh_2D','backFEM_1D','backFEM_2D','MeshAdaptor'])
    parser.add_argument('--gnn_dont_train', type=str, default="False", help="gnn_dont_train.")

    #backFEM params
    parser.add_argument('--start_from_ma', type=str, default="False", help="Whether to start backFEM optimization from MA mesh.")

    #UM2N args
    parser.add_argument('--pretrained_weights', type=str, default=None, help='Path to pretrained model weights')
    parser.add_argument('--num_transformer_in', type=int, default=3, help='Input dimension for transformer')
    parser.add_argument('--num_transformer_out', type=int, default=16, help='Output dimension for transformer')
    parser.add_argument('--num_transformer_embed_dim', type=int, default=64, help='Embedding dimension for transformer')
    parser.add_argument('--num_transformer_heads', type=int, default=4, help='Number of transformer attention heads')
    parser.add_argument('--num_transformer_layers', type=int, default=1, help='Number of transformer layers')
    parser.add_argument('--transformer_training_mask', type=t_or_f, default=False, help='Use training mask in transformer')
    parser.add_argument('--transformer_key_padding_training_mask', type=t_or_f, default=False, help='Use key padding mask in transformer')
    parser.add_argument('--transformer_attention_training_mask', type=t_or_f, default=False, help='Use attention mask in transformer')
    parser.add_argument('--transformer_training_mask_ratio_lower_bound', type=float, default=0.5, help='Lower bound for training mask ratio')
    parser.add_argument('--transformer_training_mask_ratio_upper_bound', type=float, default=0.9, help='Upper bound for training mask ratio')
    parser.add_argument('--new_model_monitor_type', type=str, default='UM2N', help='Type of monitor information used in the GNN')  # 'Hessian_Frob_u_tensor' or 'UM2N'
    parser.add_argument('--grand_diffusion', type=bool, default=True, help='Use diffusion in GNN architecture')
    parser.add_argument('--grand_step_size', type=float, default=0.1, help='Step size for GNN diffusion')
    parser.add_argument('--grand_diffusion_steps', type=int, default=20, help='Number of diffusion steps')

    parser.add_argument('--use_inversion_loss', type=bool, default=False)
    parser.add_argument('--use_area_loss', type=bool, default=True)
    parser.add_argument('--weight_deform_loss', type=float, default=1.0)
    parser.add_argument('--weight_area_loss', type=float, default=1.0)
    parser.add_argument('--weight_chamfer_loss', type=float, default=1.0)

    #Burger's params
    parser.add_argument('--gauss_amplitude', type=float, default=0.25, help="amplitude of Gaussians")
    parser.add_argument('--burgers_limits', type=float, default=3.0, help="spatial domain limits")
    parser.add_argument('--plots_multistep_eval', type=str, default="False")
    parser.add_argument('--plots_mesh_movement', type=str, default="False")

    #training params
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--device', type=str, default='cpu')
    parser.add_argument('--batch_size', type=int, default=1) #todo batch size > 1 not supported for firedrake yet
    parser.add_argument('--train_idxs', nargs='+', default=None, help="list of data points to overfit to")
    parser.add_argument('--test_idxs', nargs='+', default=None, help="list of data points to overfit to")
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--dropout', type=float, default=0.)
    parser.add_argument('--decay', type=float, default=0.)
    parser.add_argument('--loss_type', type=str, default='pde_loss_regularised')
    parser.add_argument('--loss_regulariser_weight', type=float, default=1.0)
    parser.add_argument('--loss_fn', type=str, default='l1', choices=['mse', 'l1'])
    parser.add_argument('--eval_rollout_steps', type=int, default=10)

    #plot params
    parser.add_argument('--show_plots', type=str, default="True", help="flag to show plots")
    parser.add_argument('--show_dataset_plots', type=str, default="True", help="flag to show full test dataset plots")
    parser.add_argument('--show_train_evol_plots', type=str, default="True", help="flag to show evolution of training")
    parser.add_argument('--show_mesh_evol_plots', type=str, default="False", help="flag to show evolution of mesh plots")
    parser.add_argument('--show_mesh_plots', type=str, default="False", help="flag to show individual mesh plots")

    # wandb and raytune logging and tuning
    parser.add_argument('--wandb', action='store_true', help="flag if logging to wandb")
    parser.add_argument('--wandb_offline', action='store_true')
    parser.add_argument('--wandb_sweep', action='store_true', help="flag if sweeping")
    parser.add_argument('--wandb_entity', default=None, type=str, help="qls")
    parser.add_argument('--wandb_project', default="G-Adaptivity", type=str)
    parser.add_argument('--wandb_group', default="testing", type=str, help="testing,tuning,eval")
    parser.add_argument('--wandb_run_name', default=None, type=str)
    parser.add_argument('--wandb_log_plots', action='store_true', help="flag to log plots")
    parser.add_argument('--wandb_save_model', action='store_true', help="flag to save model")
    parser.add_argument('--wandb_load_model', type=str, default=None, choices=['wandb', 'local'])
    parser.add_argument('--wandb_model_path', type=str, default=None, help="path to load wandb artifacts f'{entity}/{project}/{id}'")
    parser.add_argument('--local_model_path', type=str, default=None, help="path to load  *.pt")

    parser.add_argument('--wandb_checkpoint_freq', type=int, default=None, help="frequency of saving checkpoints")
    parser.add_argument('--wandb_exp_idx', type=int, default=0, help="experiment index")

    args = parser.parse_args()
    opt = vars(args)
    
    opt = tf_sweep_args(opt)

    if not opt['wandb_sweep']:
        opt = run_params(opt)
    
    rand_seed = np.random.randint(3, 10000)
    opt['seed'] = rand_seed
    set_seed(opt['seed'])
    
    return opt