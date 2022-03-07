This code trains an RL-agent with help of the ```stable baseline3``` framework. The main code of the training 
environment is located in the ```push_gym``` folder, here are only the training scripts.

- To train an RL-agent, customize the parameters given in ```scripts/parametes.yaml```, check or change the network's 
hyperparameter in ```scripts/train_agent_script``` and run ```python scripts/run_agent.py```. 

- To evaluate a trained agent, set ```train: False``` in ```scripts/parametes.yaml``` and run ```python scripts/run_agent.py```.

- To evaluate the baseline approach by krivic et al [1], set ```train: False``` and ```test_baseline: True``` in ```scripts/parametes.yaml``` and run ```python scripts/run_agent.py```.

For training with curiculum learning check or change the curiculum step range in ```python scripts/custom_callbacks.py```