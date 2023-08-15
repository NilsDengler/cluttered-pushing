from test_agent_script import test_agent
from train_agent_script import train_model
import yaml
import inspect


if __name__ == "__main__":
    # load parameters from yaml file
    file = open("parameters.yaml", 'r')
    param_dict = yaml.load(file, Loader=yaml.FullLoader)

    # get train_model and test_agent parameters from function call
    train_args = inspect.getfullargspec(train_model).args
    test_args = inspect.getfullargspec(test_agent).args

    # check if any args are missing
    if not all(arg in param_dict for arg in train_args):
        raise ValueError("Missing arguments: ", [arg for arg in train_args if arg not in param_dict])

    if not all(arg in param_dict for arg in test_args):
        raise ValueError("Missing arguments: ", [arg for arg in test_args if arg not in param_dict])

    if param_dict['train']:
        # remove unnecessary parameters
        train_param_dict = {k: v for k, v in param_dict.items() if k in train_args}

        train_model(**train_param_dict)
    else:
        test_param_dict = {k: v for k, v in param_dict.items() if k in test_args}
        
        test_agent(**test_param_dict)
