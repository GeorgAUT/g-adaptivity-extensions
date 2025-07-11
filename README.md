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
python src/run_pipeline.py --config configs/XXX.yaml
```

where the folder `configs/` contains a number of configuration files that specify examples shown in the main paper including:

- `configs/poisson.yaml`: Training and evaluation of the Poisson equation example.
- ...


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
