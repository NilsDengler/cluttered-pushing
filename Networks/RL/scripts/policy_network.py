import gym
import torch as th
import torch.nn as nn
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor

class CNN(BaseFeaturesExtractor):
    """
    :param observation_space: (gym.Space)
    :param features_dim: (int) Number of features extracted.
        This corresponds to the number of unit for the last layer.
    """
    def __init__(self, observation_space: gym.spaces.Dict, features_dim: int = 256, local_window_size: int = 64, num_scalar: int = 29):
        super(CNN, self).__init__(observation_space, features_dim)
        # We assume CxHxW images (channels first)
        # Re-ordering will be done by pre-preprocessing or wrapper

        self.local_window_size = local_window_size

        self.local_window_cnn = nn.Sequential(
            nn.Conv2d(1, 64, kernel_size=3, stride=1, padding=0),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=0),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 32, kernel_size=3, stride=1, padding=0),
            nn.ReLU(),
            nn.Conv2d(32, 16, kernel_size=3, stride=1, padding=0),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Flatten(),
            nn.Linear(1*16*13*13, features_dim)

        )

        self.scalar_dense = nn.Sequential(nn.Linear(num_scalar, features_dim))

        self.linear = nn.Sequential(nn.Linear(features_dim*2, features_dim), nn.ReLU())


    def forward(self, observations: th.Tensor) -> th.Tensor:
        observation = observations
        local_window = observation[:,:self.local_window_size * self.local_window_size]
        local_window = th.reshape(local_window, (observation.size()[0], 1, self.local_window_size, self.local_window_size))
        window_features = self.local_window_cnn(local_window)
        scalar_features = self.scalar_dense(observation[:,self.local_window_size * self.local_window_size:])
        concat_features = th.cat((window_features, scalar_features), 1)
        return self.linear(concat_features)


class CNNHER(BaseFeaturesExtractor):
    """
    :param observation_space: (gym.Space)
    :param features_dim: (int) Number of features extracted.
        This corresponds to the number of unit for the last layer.
    """

    def __init__(self, observation_space: gym.spaces.Dict, features_dim: int = 256, local_window_size: int = 64, num_scalar: int = 29):
        super(CNNHER, self).__init__(observation_space, features_dim)
        # We assume CxHxW images (channels first)
        # Re-ordering will be done by pre-preprocessing or wrapper

        self.local_window_size = local_window_size

        self.local_window_cnn = nn.Sequential(
            nn.Conv2d(1, 64, kernel_size=3, stride=1, padding=0),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=0),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 32, kernel_size=3, stride=1, padding=0),
            nn.ReLU(),
            nn.Conv2d(32, 16, kernel_size=3, stride=1, padding=0),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Flatten(),
            nn.Linear(1*16*13*13, features_dim)

        )

        self.scalar_dense = nn.Sequential(nn.Linear(num_scalar, features_dim))

        self.linear = nn.Sequential(nn.Linear(features_dim*2, features_dim), nn.ReLU())


    def forward(self, observations: th.Tensor) -> th.Tensor:
        observation = observations["observation"]
        local_window = observation[:,:self.local_window_size * self.local_window_size]
        local_window = th.reshape(local_window, (observation.size()[0], 1, self.local_window_size, self.local_window_size))
        window_features = self.local_window_cnn(local_window)
        scalar_features = self.scalar_dense(observation[:,self.local_window_size * self.local_window_size:])
        concat_features = th.cat((window_features, scalar_features), 1)
        return self.linear(concat_features)