# G-Adaptivity

This repository contains the official implementation of [G-Adaptivity](https://openreview.net/forum?id=pyIXyl4qFx): a GNN-based approach to adaptive mesh refinement for finite element methods (FEM).

## 📦 Installation

Our code depends on [Firedrake](https://www.firedrakeproject.org/), a Python-based finite element library used to solve the PDEs in our experiments.

We recommend installing Firedrake via the [official guide](https://www.firedrakeproject.org/install.html#installing-firedrake), which will also set up a dedicated virtual environment.

Once Firedrake is installed and its virtual environment activated, you can install G-Adaptivity and its dependencies from our `pyproject.toml`:

```bash
git clone https://github.com/JRowbottomGit/g-adaptivity.git
cd g-adaptivity
pip install -e .
```

Once installed in your virtual environment, this will allow you to import g-adaptivity to work with your own code and products, as well as run our training and evaluation scripts.

## 📁 Datasets

The code can generate training and test datasets directly, but this is computationally expensive due to the large number of FEM solves involved.

To save time, we provide precomputed datasets via Zotero:  
🔗 [https://www.zotero.org/groups/2722052/g-adaptivity](https://www.zotero.org/groups/2722052/g-adaptivity)

After downloading, place the datasets in the `data/` folder at the repository root:

```
g-adaptivity/
└── data/
    └── <your_downloaded_data_here>
```

⚠️ The `data/` folder may not exist until you create it manually or run a script that uses it.


## 🚀 Training and Evaluation

To train and evaluate models from the paper, run:

```bash
python src/run_pipeline.py --exp_config configs/XXX.yaml
```

where the folder `configs/` contains a number of configuration files that specify examples shown in the main paper. The simplest example is `configs/poisson_square_mixed.yaml`, which trains a model on the Poisson equation with a square mesh and mixed data types.

## Config file structure

The configuration files in `configs/` contain the `base_config.yaml` which contain a larger number of standardised parameter settings (e.g. training and model parameters) together with the experiment config files (e.g. `poisson_square_mixed.yaml`). Any parameter set in an experiment config file will overwrite the base_config setting of the corresponding parameter. Thus as a starting point we suggest users to work with variations on the experiment config files before diving deeper into the code and modifying any base configs.

The major components of the experiment config files are as follows:

```yaml
# Example configuration file for G-Adaptivity
run:
  pde_type: "Poisson"  # 'Poisson', 'Burgers', 'NavierStokes'
  data_type: "randg_mix"  # 'randg', 'randg_mix', 'RBF'
  model: "MeshAdaptor"  # 'MeshAdapter', 'backFEM2D'

data:
  mesh_geometry: "rectangle" # 'polygon_010', 'rectangle', 'cylinder100', 'cylinder015', 'cylinder010', 'H-shape', 'headland1', 'headland2', 'headland05' or 'L-shape'
  mesh_dims_train: [[15, 15], [20, 20]]
  mesh_dims_test: [[12, 12], [14, 14], [16, 16], [18, 18], [20, 20], [22, 22]]
```

Note that you can also work with your own mesh. For this you need to place your custom `.mesh` file in the `/meshes/` folder and specify the filename (no `.mesh` ending) as the `mesh_geometry` parameter.

## ⚠️ Known issues

- Anaconda is known to cause issues when installing firedrake on MacOS, we recommend using homebrew where possible and consulting the [Firedrake installation guide](https://www.firedrakeproject.org/install.html) as well as their [GitHub page](https://github.com/firedrakeproject/firedrake/issues) for more information and installation support.


## 📄 License and citation

This open-source version of our code is licensed under Apache 2.0 - if you are interested in a commercial license with support/customization, please do contact us. If you use this work, please cite:

```
@inproceedings{Rowbottom_G-Adaptivity_optimised_graph-based_2025,
    author = {Rowbottom, James and Maierhofer, Georg and Deveney, Teo and Müller, Eike Hermann and Paganini, Alberto and Schratz, Katharina and Lio, Pietro and Schönlieb, Carola-Bibiane and Budd, Chris},
    booktitle = {Proceedings of the Forty-second International Conference on Machine Learning},
    title = {{G-Adaptivity: optimised graph-based mesh relocation for finite element methods}},
    year = {2025}
}
```
