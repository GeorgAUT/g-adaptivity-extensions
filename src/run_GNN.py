import os
from tqdm.auto import tqdm
from copy import deepcopy
import numpy as np
import torch
import torch.nn.functional as F
from torch_geometric.loader import DataLoader
import wandb

from data import make_data_name, MeshInMemoryDataset
from data_mixed import MeshInMemoryDataset_Mixed
from data_mixed_loader import Mixed_DataLoader
from utils_main import plot_training_evol, inner_progress
from utils_train import equidistribution_loss
from models.mesh_adaptor_model import Mesh_Adaptor
from models.UM2N_aux.um2n_loss import UM2N_loss

from params import get_params, run_params, tf_sweep_args, set_seed, get_arg_list


def get_data(opt, train_test="test"):

    mesh_dims = get_arg_list(opt['mesh_dims'])
    num_data = opt['num_train'] if train_test == "train" else opt['num_test']
    opt = make_data_name(opt, train_test)
    print(f"Data name: {opt['data_name']}")

    if opt['data_type'] == 'randg_mix':
        # this branch services
        # Poisson_square_mixed
        # Poisson_cube_mixed
        # Poisson_patches_mixed
        dataset = MeshInMemoryDataset_Mixed(f"{opt['data_dir']}/{opt['data_name']}", num_data, mesh_dims, train_test, opt)
    else:
        # this branch services
        # Poisson_structured testing dataset
        # Poisson_square_fixed_resolution
        #TODO Poisson_cube_fixed_resolution - DONE
        #TODO Poisson_patches - DONE
        # Burgers_square_fixed_resolution
        # NavierStokes
        #TODO headland & L
        dataset = MeshInMemoryDataset(f"{opt['data_dir']}/{opt['data_name']}", train_test, num_data, mesh_dims, opt)

    return dataset



def main(opt):
    dataset = get_data(opt, train_test="train")
    if opt['data_type'] == 'randg_mix':
        exclude_keys = ['boundary_nodes_dict', 'mapping_dict', 'node_boundary_map', 'eval_errors', 'pde_params']
        follow_batch = []
        loader = Mixed_DataLoader(dataset, batch_size=opt['batch_size'], shuffle=not (opt['train_idxs']),
                                  exclude_keys=exclude_keys, follow_batch=follow_batch,
                                  generator=torch.Generator(device=opt['device']))
    else:
        new_generator = torch.Generator(device=opt['device'])
        loader = DataLoader(dataset, batch_size=opt['batch_size'], shuffle=not (opt['train_idxs']),
                            generator=new_generator)

    dim = dataset.dim
    model = Mesh_Adaptor(opt, gfe_in_c=dim + 1, lfe_in_c=dim + 1, deform_in_c=dim + 1).to(opt['device'])

    if opt['loss_fn'] == 'mse':
        loss_fn = F.mse_loss
    elif opt['loss_fn'] == 'l1':
        loss_fn = F.l1_loss

    optimizer = torch.optim.Adam(model.parameters(), lr=opt['lr'], weight_decay=opt['decay'])

    loss_list = []
    batch_loss_list = []
    best_loss = float('inf')
    best_dict = None
    
    model.train()
    
    epoch_bar = tqdm(range(opt['epochs']), desc='Epochs', position=0, leave=True,
                    bar_format='{l_bar}{bar:10}{r_bar}{bar:-10b}')
    
    for epoch in epoch_bar:
        model.epoch = epoch
        epoch_loss = 0
        batch_loss_list = []
        
        for i, data in enumerate(loader):
            data.idx = i
            
            if opt['train_idxs']:
                idx = dataset.data_idx_dict[i] if opt['data_type'] == "RBF" else i
                if idx not in opt['train_idxs']:
                    continue
            
            # Reset mesh coordinates before forward pass
            if opt['mesh_file_type'] == 'bin':
                dataset.reset_mesh_coordinates()
            
            optimizer.zero_grad()
            
            if opt['loss_type'] == 'mesh_loss':
                data = data.to(opt['device'])
                x_phys = model(data)
                loss = loss_fn(x_phys, data.x_ma)
                
            elif opt['loss_type'] == 'UM2N_loss':
                x_phys = model(data)
                loss = UM2N_loss(x_phys, loss_fn, data,
                                use_inversion_loss=opt['use_inversion_loss'],
                                use_area_loss=opt['use_area_loss'],
                                weight_deform_loss=opt['weight_deform_loss'],
                                weight_area_loss=opt['weight_area_loss'],
                                weight_chamfer_loss=opt['weight_chamfer_loss'])
                
            elif opt['loss_type'] in ['pde_loss_regularised']:
                if dataset.dim == 1:
                    n_meshpoints = data.num_nodes
                elif dataset.dim == 2:
                    n_meshpoints = int(np.sqrt(data.num_nodes))
                elif dataset.dim == 3:
                    n_meshpoints = int(np.cbrt(data.num_nodes))

                if opt['data_type'] == 'randg_mix':
                    coarse_solver = getattr(dataset, f"PDESolver_coarse_{n_meshpoints}")
                else:
                    coarse_solver = dataset.PDESolver_coarse
                    
                if opt['data_type'] == 'randg_mix':
                    pde_params = data.batch_dict[0]['pde_params']
                else:
                    pde_params = data.pde_params

                x_phys = model(data)
                coarse_solver.update_solver(pde_params)
                try:
                    if opt['pde_type'] in ['Poisson', 'Burgers']:
                        pseudo_loss, loss = coarse_solver.loss(opt, data.uu_ref[0], x_phys)
                    elif opt['pde_type'] == 'NavierStokes':
                        pseudo_loss, loss = coarse_solver.loss(opt, [data.uu_ref[0], data.pp_ref[0]], x_phys)
                    pseudo_loss = pseudo_loss + opt['loss_regulariser_weight'] * equidistribution_loss(x_phys, data)
                except:
                    Warning("PDE loss failed, using only regulariser for this datapoint")
                    pseudo_loss = opt['loss_regulariser_weight'] * equidistribution_loss(x_phys, data)
                    loss = pseudo_loss.detach()

            # Get the current loss for this batch
            current_loss = loss.item()
            epoch_loss += current_loss
            batch_loss_list.append(current_loss)

            # Update epoch bar with batch progress
            epoch_bar.set_postfix_str(f"loss={current_loss:.2e} | {inner_progress(i+1, len(loader))}")

            # Perform backpropagation and optimizer step
            if (opt['loss_type'] == 'modular' or 
                opt['loss_type'] == 'pde_loss_firedrake' or 
                opt['loss_type'] == 'mixed_UM2N_pde_loss_firedrake' or 
                opt['loss_type'] == 'pde_loss_regularised') and not opt['gnn_dont_train']:
                pseudo_loss.backward()
                optimizer.step()
            elif not opt['gnn_dont_train']:
                loss.backward()
                optimizer.step()

            # Reset mesh coordinates after backward pass
            if opt['mesh_file_type'] == 'bin':
                dataset.reset_mesh_coordinates()
        
        # After completing all batches for this epoch
        # Store total epoch loss and update epoch progress bar
        avg_loss = epoch_loss/len(loader)
        epoch_bar.set_postfix(loss=f"{avg_loss:.2e}")
        epoch_bar.set_description(f"Epoch {epoch+1}/{opt['epochs']}")

        # Save best model
        if epoch_loss < best_loss:
            best_loss = epoch_loss
            best_dict = deepcopy(model.state_dict())

        # Save model checkpoint if configured
        if opt['wandb_save_model'] and opt['wandb_checkpoint_freq'] is not None and (epoch + 1) % opt['wandb_checkpoint_freq'] == 0:
            torch.save(best_dict, os.path.join(wandb.run.dir, f"model_best_epoch_{epoch}.pt"))
            artifact = wandb.Artifact(f"model_best_epoch_{epoch}", type="model")
            artifact.add_file(os.path.join(wandb.run.dir, f"model_best_epoch_{epoch}.pt"))
            wandb.log_artifact(artifact)

    if opt['show_train_evol_plots']:
        loss_fig = plot_training_evol(loss_list, "loss", batch_loss_list=batch_loss_list,
                                      batches_per_epoch=len(dataset) // opt['batch_size'])

    if opt['wandb']:
        epoch_loss_pairs = [(epoch, loss) for epoch, loss in enumerate(loss_list)]
        loss_table = wandb.Table(data=epoch_loss_pairs, columns=["epoch", "loss"])
        wandb.log({"final_epochs": epoch,
                   "final_loss": epoch_loss,
                   "loss_table": loss_table})

        if opt['wandb_log_plots'] and opt['show_train_evol_plots']:
            wandb.log({"loss": wandb.Image(loss_fig)})

        if opt['wandb_save_model']:
            torch.save(best_dict, os.path.join(wandb.run.dir, "model_best.pt"))
            torch.save(model.state_dict(), os.path.join(wandb.run.dir, "model_last.pt"))
            artifact = wandb.Artifact("model", type="model")
            artifact.add_file(os.path.join(wandb.run.dir, "model_best.pt"))
            artifact.add_file(os.path.join(wandb.run.dir, "model_last.pt"))
            wandb.log_artifact(artifact)
    
    model.load_state_dict(best_dict)

    return model


if __name__ == "__main__":
    opt = get_params()
    main(opt)