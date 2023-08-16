# Cluttered Pushing
This repository contains the accompanying code for the paper Learning Goal-Oriented Non-Prehensile Pushing in Cluttered Scenes by N. Dengler, D. Großklaus and M. Bennewitz submitted for IROS, 2022. you can find the paper at 
http://arxiv.org/abs/2203.02389
 
## Setup

Step 1: Clone the repository

```bash
cd 
git clone https://github.com/NilsDengler/cluttered-pushing.git
```

Step 2: Create a virtual environment

```bash
cd cluttered-pushing
conda env create -f environment.yml -n <env_name>

conda activate <env_name>
```

Step 3: Install the package

Dependencies:
```bash
cd cluttered-pushing/push_gym/push_gym/utils/Lazy_Theta_with_optimization_any_angle_pathfinding

mkdir build && cd build

cmake ..

make
```

Package:
```bash
cd cluttered-pushing/push_gym

pip install -e .
```

## Usage

Change directory to ```cluttered-pushing/Networks/RL/scripts```
```bash
cd cluttered-pushing/RL
```

### Training

- To train an RL-agent, customize the parameters given in ```scripts/parametes.yaml```.
- Set the ```train: True``` in ```scripts/parametes.yaml```.
- Check or change the network's 
hyperparameter in ```scripts/train_agent_script```.

- A VAE model is required for training, please refere to [VAE Readme](https://github.com/NilsDengler/cluttered-pushing/tree/main/Networks/VAE) to train a VAE model or download already trained models.

- To start training:
  ```bash
  python run_agent.py
  ``` 

### Testing
- To evaluate a trained agent, set ```train: False``` in ```scripts/parametes.yaml```.
- Testing:
  ```bash
  python scripts/run_agent.py
  ```

### Baseline
- To run the Baseline by Krivic and Piater, set ```train: False``` and ```test_baseline: True``` in ```scripts/parametes.yaml```.
- Testing:
  ```bash
  python scripts/run_agent.py
  ```

For more information, please refer the [README](https://github.com/NilsDengler/cluttered-pushing/tree/main/Networks/RL) in ```cluttered-pushing/RL```.

