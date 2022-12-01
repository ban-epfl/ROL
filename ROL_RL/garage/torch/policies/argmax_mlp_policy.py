"""ArgMaxMLPPolicy."""

import click

import torch
from torch import nn

from garage.torch.modules import MLPModule
from garage.torch.policies.stochastic_policy import StochasticPolicy
from pdb import set_trace as bp
from garage.torch.policies.policy import Policy
import numpy as np
import akro
from garage.torch._functions import list_to_tensor, np_to_torch


class ArgMaxMLPPolicy(Policy):
    """MLP whose outputs are fed into a Normal distribution..

    A policy that contains a MLP to make prediction based on a gaussian
    distribution.

    Args:
        env_spec (EnvSpec): Environment specification.
        hidden_sizes (list[int]): Output dimension of dense layer(s) for
            the MLP for mean. For example, (32, 32) means the MLP consists
            of two hidden layers, each with 32 hidden units.
        hidden_nonlinearity (callable): Activation function for intermediate
            dense layer(s). It should return a torch.Tensor. Set it to
            None to maintain a linear activation.
        hidden_w_init (callable): Initializer function for the weight
            of intermediate dense layer(s). The function should return a
            torch.Tensor.
        hidden_b_init (callable): Initializer function for the bias
            of intermediate dense layer(s). The function should return a
            torch.Tensor.
        output_nonlinearity (callable): Activation function for output dense
            layer. It should return a torch.Tensor. Set it to None to
            maintain a linear activation.
        output_w_init (callable): Initializer function for the weight
            of output dense layer(s). The function should return a
            torch.Tensor.
        output_b_init (callable): Initializer function for the bias
            of output dense layer(s). The function should return a
            torch.Tensor.
        layer_normalization (bool): Bool for using layer normalization or not.
        name (str): Name of policy.

    """

    def __init__(self,
                 env_spec,
                 hidden_sizes=(32, 32),
                 hidden_nonlinearity=torch.tanh,
                 hidden_w_init=nn.init.xavier_uniform_,
                 hidden_b_init=nn.init.zeros_,
                 output_nonlinearity=torch.softmax,
                 output_w_init=nn.init.xavier_uniform_,
                 output_b_init=nn.init.zeros_,
                 layer_normalization=False,
                 name='ArgMaxMLPPolicy'):
        super().__init__(env_spec, name)
        self._obs_dim = env_spec.observation_space.flat_dim
        self._action_dim = env_spec.action_space.flat_dim
        self._module = MLPModule(
            input_dim=self._obs_dim,
            output_dim=self._action_dim,
            hidden_sizes=hidden_sizes,
            hidden_nonlinearity=hidden_nonlinearity,
            hidden_w_init=hidden_w_init,
            hidden_b_init=hidden_b_init,
            output_nonlinearity=output_nonlinearity,
            output_w_init=output_w_init,
            output_b_init=output_b_init,
            layer_normalization=layer_normalization)

    def get_action(self, observation):
        r"""Get a single action given an observation.

        Args:
            observation (np.ndarray): Observation from the environment.
                Shape is :math:`env_spec.observation_space`.

        Returns:
            tuple:
                * np.ndarray: Predicted action. Shape is
                    :math:`env_spec.action_space`.
                * dict:
                    * np.ndarray[float]: Mean of the distribution
                    * np.ndarray[float]: Standard deviation of logarithmic
                        values of the distribution.
        """
        if not isinstance(observation, np.ndarray) and not isinstance(
                observation, torch.Tensor):
            observation = self._env_spec.observation_space.flatten(observation)
        elif isinstance(observation,
                        np.ndarray) and len(observation.shape) > 1:
            observation = self._env_spec.observation_space.flatten(observation)
        elif isinstance(observation,
                        torch.Tensor) and len(observation.shape) > 1:
            observation = torch.flatten(observation)
        with torch.no_grad():
            if isinstance(observation, np.ndarray):
                observation = np_to_torch(observation)
            if not isinstance(observation, torch.Tensor):
                observation = list_to_tensor(observation)
            observation = observation.unsqueeze(0)
            action, agent_infos = self.get_actions(observation)
            return action[0], {k: v[0] for k, v in agent_infos.items()}

    def get_actions(self, observations):
        r"""Get actions given observations.

        Args:
            observations (np.ndarray): Observations from the environment.
                Shape is :math:`batch_dim \bullet env_spec.observation_space`.

        Returns:
            tuple:
                * np.ndarray: Predicted actions.
                    :math:`batch_dim \bullet env_spec.action_space`.
                * dict:
                    * np.ndarray[float]: Mean of the distribution.
                    * np.ndarray[float]: Standard deviation of logarithmic
                        values of the distribution.
        """
        if not isinstance(observations[0], np.ndarray) and not isinstance(
                observations[0], torch.Tensor):
            observations = self._env_spec.observation_space.flatten_n(
                observations)

        # frequently users like to pass lists of torch tensors or lists of
        # numpy arrays. This handles those conversions.
        if isinstance(observations, list):
            if isinstance(observations[0], np.ndarray):
                observations = np.stack(observations)
            elif isinstance(observations[0], torch.Tensor):
                observations = torch.stack(observations)

        if isinstance(observations[0],
                      np.ndarray) and len(observations[0].shape) > 1:
            observations = self._env_spec.observation_space.flatten_n(
                observations)
        elif isinstance(observations[0],
                        torch.Tensor) and len(observations[0].shape) > 1:
            observations = torch.flatten(observations, start_dim=1)
        with torch.no_grad():
            if isinstance(observations, np.ndarray):
                observations = np_to_torch(observations)
            if not isinstance(observations, torch.Tensor):
                observations = list_to_tensor(observations)

            if isinstance(self._env_spec.observation_space, akro.Image):
                observations /= 255.0  # scale image
            dist, info = self.forward(observations)
            return dist.sample().cpu().numpy(), {
                k: v.detach().cpu().numpy()
                for (k, v) in info.items()
            }

    def forward(self, observations):
        """Compute the action distributions from the observations.

        Args:
            observations (torch.Tensor): Batch of observations on default
                torch device.

        Returns:
            torch.distributions.Distribution: Batch distribution of actions.
            dict[str, torch.Tensor]: Additional agent_info, as torch Tensors

        """
        # We're given flattened observations.
        # observations = observations.reshape(
        #     -1, self._env_spec.observation_space.flat_dim)
        # bp()
        # from dowel import logger
        # logger.log('in optimizer...********************')
        # click.echo('Hello World!****************',)
        mlp_output = self._module(observations)
        mlp_output[observations == 0] = -1e10
        # mlp_output = mlp_output.reshape(
        #     -1, self._env_spec.observation_space.flat_dim)
        # print("******", observations.shape)
        # print("******", mlp_output.shape)
        # print("******", mlp_output)
        probs = torch.softmax(mlp_output, dim=1)
        # print("******", probs.shape)
        # print("******", probs)
        # print("******",torch.argmax(logits, dim=1).shape)
        # print("******",torch.argmax(logits, dim=1))
        dist = torch.distributions.OneHotCategorical(probs=probs)
        # samples = dist.sample().cpu()
        # print("draw: ", samples)
        # print("prob: ", dist.log_prob(samples))
        return dist, {}
