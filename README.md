## WARNING!

This code us currently out of date and we are working hard to realise a revised version in the coming weeks. If your project timeline allows, please do wait for the new version as it significantly improves both on speed and performance. If you have any questions in the maintime please do feel free to contact us via email (see preprint for contact details).

## g-adaptivity
This repository is the official implementation of [g-adaptivity](https://arxiv.org/abs/2407.04516). A GNN based approach to adaptive mesh refinement for FEM.


## Requirements

First note the firedrake anti-requirement regarding Anaconda, we found this can be managed by having a brew installed version of python (ours 3.11.9) and rearranging the PATH in the .bash_profile to have the brew python before the anaconda python. 
```bash
% echo "$PATH"
/Users//workspace/firedrake/bin:/opt/homebrew/bin:/Users//miniforge3/bin:
```

To install firedrake follow:
https://www.firedrakeproject.org/download.html
in workspace directory run:
```firedrake
curl -O https://raw.githubusercontent.com/firedrakeproject/firedrake/master/scripts/firedrake-install
python firedrake-install
source ~/workspace/firedrake/bin/activate
source ./firedrake_new/bin/activate
```

To install requirements:
```setup
pip install torch torchvision torchaudio
pip install torch-geometric==2.4.0
pip install torchdiffeq
pip install torchquad
pip install pandas
pip install plotly
pip install wandb
pip install git+https://github.com/pyroteus/movement.git
pip install git+https://github.com/rusty1s/pytorch_scatter.git
pip install adjustText
pip install imageio\[ffmpeg\]
pip install tensorboard
pip install einops
pip install "git+https://github.com/facebookresearch/pytorch3d.git@stable"
pip install pyvista

#python -m pip install siphash24
#cd /home/firedrake/firedrake/src/
#git clone https://github.com/mesh-adaptation/animate.git
#cd animate
#pip install -e .

git clone https://github.com/mesh-adaptation/movement.git
cd movement
make install
cd ../src
```

## Training and Evaluation

To train the model(s) in the paper, run this command:

```train
python run_pipeline.py
```

Configs are stored in `params.py`.

## Cite us
If you found this work useful, please consider citing our paper:

```
@misc{g-adaptivity,
      title={G-Adaptive mesh refinement - leveraging graph neural networks and differentiable finite element solvers}, 
      author={James Rowbottom, Georg Maierhofer, Teo Deveney, Katharina Schratz, Pietro Liò, Carola-Bibiane Schönlieb and Chris Budd},
      year={2024},
      eprint={},
      archivePrefix={arXiv},
      primaryClass={cs.LG}
}
```
