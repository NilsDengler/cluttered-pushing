 This code trains an VAE to use the latent space as observation for an RL-agent.
 To train the VAE, customize the parameters given in ```scripts/parametes.yaml```, check or change the network's 
 layers in ```models/tf_models``` and run ```python scripts/run_vae.py```. The code is written in python 3.8 and uses 
 tensorflow 2.4, as well as the tensorflow-probability package.

 ### Example:
- To train an example VAE, download the .h5 file from [1] and place it in ```train_data```.
- check ```scripts/parametes.yaml```.
- run ```python scripts/run_vae.py```.

[1] You can find trained example VAEs and and example dataset here: https://cloud.vi.cs.uni-bonn.de/index.php/s/xjaZ39Gjd3RaPpP

