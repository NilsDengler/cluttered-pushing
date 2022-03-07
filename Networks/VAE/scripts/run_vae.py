import sys, os
sys.path.append('/home/user/dengler/Documents/robot_arm_RL/pushing_net/Networks/VAE/')
sys.path.append('/home/user/dengler/Documents/robot_arm_RL/pushing_net/Networks/VAE/scripts/')
sys.path.append('/home/user/dengler/Documents/robot_arm_RL/pushing_net/Networks/VAE/models/')
import yaml
from train_vae import train_model

if __name__ == "__main__":
    params = []
    file = open("parameters.yaml", 'r')
    dict = yaml.load(file, Loader=yaml.FullLoader)
    for key, value in dict.items():
        print("Key: ", key, ", Value: ", value)
        params.append(value)
    if params[0]:
        print("plain")
        train_model(params[1], params[2], params[3], params[4], params[5], params[6], params[7], params[8], params[9],
                    params[10], params[11], params[12], params[13], params[14])
    else:
        print("TESTING NOT IMPLEMENTED!")

