"""Simple Graph environment"""
import akro
import numpy as np
from garage import Environment, EnvSpec, EnvStep, StepType
from scipy.stats import norm
import networkx as nx
import math
from scipy import stats

class GraphEnv(Environment):
    """
    The proper Environment class for casual discovery

    """

    def __init__(self,
                 done_bonus=0.,
                 never_done=False,
                 alpha=0.001,
                 data=None,
                 A=None,
                 oracle=False):

        self.oracle = oracle
        self.alpha = alpha
        self.data = data
        self.adjacency_matrix = A
        n_vertices = data.shape[1]
        self.remaining_ids = np.arange(n_vertices)
        self.markov_boundary = self.reconstruct_mb()
        self.first_markov_boundary = self.markov_boundary.copy()
        self.generated_graph = [[0] * n_vertices for i in range(n_vertices)]
        self._done_bonus = done_bonus
        self._never_done = never_done
        self._step_cnt = None
        self._max_episode_length = n_vertices
        # self._observation_space = akro.Discrete(n_vertices)
        self._observation_space = akro.Box(low=-np.inf,
                                           high=np.inf,
                                           shape=(n_vertices,),
                                           dtype=np.float32)
        self._action_space = akro.Discrete(n_vertices)
        self._spec = EnvSpec(action_space=self.action_space,
                             observation_space=self.observation_space,
                             max_episode_length=n_vertices)

    @property
    def action_space(self):
        """akro.Space: The action space specification."""
        return self._action_space

    @property
    def observation_space(self):
        """akro.Space: The observation space specification."""
        return self._observation_space

    @property
    def spec(self):
        """EnvSpec: The environment specification."""
        return self._spec

    def step(self, action, test_mode=False):
        """Step the environment.

        Args:
            action (np.ndarray): An action provided by the agent.
            test_mode
        Returns:
            EnvStep: The environment step resulting from the action.

        Raises:
            RuntimeError: if `step()` is called after the environment
            has been
                constructed and `reset()` has not been called.

        """
        if self._step_cnt is None:
            raise RuntimeError('reset() must be called before step()!')
        # enforce action space
        # get the index of target variable

        if np.isscalar(action):
            target_var = np.where(self.remaining_ids == action)[0][0]
        else:
            target_var = np.argmax(action[self.remaining_ids])

        if self.oracle:
            reward = self.oracle_find_neighbor(target_var, test_mode=test_mode)
        else:
            reward = self.find_neighbor(target_var, test_mode=test_mode)
        # Type conversion
        if not isinstance(reward, float):
            reward = float(reward)
        self._step_cnt += 1
        step_type = StepType.get_step_type(
            step_cnt=self._step_cnt,
            max_episode_length=self._max_episode_length,
            done=False)

        if step_type in (StepType.TERMINAL, StepType.TIMEOUT):
            self._step_cnt = None
        self.remove_target(target_var)
        self.markov_boundary = self.reconstruct_mb()
        return EnvStep(env_spec=self.spec,
                       action=action,
                       reward=reward,
                       observation=self._state,
                       env_info={
                           'task': "graph",
                           'success': False,
                           "graph": self.generated_graph
                       },
                       step_type=step_type)

    def render(self, mode):
        print('current state:', self._state)

    def reset(self):
        self._state = np.ones(self._observation_space.flat_dim)
        observation = np.copy(self._state)
        self._step_cnt = 0
        self.remaining_ids = np.arange(self._observation_space.flat_dim)
        self.markov_boundary = self.reconstruct_mb()
        self.generated_graph = [[0] * self._observation_space.flat_dim for i in range(self._observation_space.flat_dim)]
        return observation, dict()

    def set_state(self, state):
        self._state = np.copy(state)
        observation = np.copy(self._state)
        self._step_cnt = 0
        self.remaining_ids = np.where(self._state == 1)[0]
        self.markov_boundary = self.reconstruct_mb()
        self.generated_graph = [[0] * self._observation_space.flat_dim for i in range(self._observation_space.flat_dim)]
        return observation, dict()

    def find_neighbor(self, target_var, test_mode):
        markov_boundary = self.markov_boundary[target_var]
        # get first markov list
        first_markov_boundary = self.first_markov_boundary[self.remaining_ids[target_var]][self.remaining_ids]
        # if in test, take the mask.
        if test_mode:
            # get a list of markov boundary indexes.
            markov_boundary_ids = np.where(markov_boundary * first_markov_boundary == 1)[0].tolist()
        else:
            # get a list of markov boundary indexes.
            markov_boundary_ids = np.where(markov_boundary == 1)[0].tolist()

        # assume all vertices are neighbors, then remove co parents
        neighbors = len(markov_boundary_ids)
        for y in markov_boundary_ids:
            self.generated_graph[self.remaining_ids[target_var]][self.remaining_ids[y]] = 1

        for y in markov_boundary_ids:
            for z in markov_boundary_ids:
                if z != y and self.gcastle_test(self.remaining_ids[target_var], self.remaining_ids[y],
                                           [self.remaining_ids[e] for e in markov_boundary_ids if e != y and e != z],
                                           ):
                    neighbors -= 1
                    # remove Y from the X in self.generated_graph which were all one
                    self.generated_graph[self.remaining_ids[target_var]][self.remaining_ids[y]] = 0
                    break

        # return an array with neighbors and the reward of this step.
        return -neighbors
    def oracle_find_neighbor(self, target_var, test_mode):
            markov_boundary = self.markov_boundary[target_var]
            # get first markov list
            first_markov_boundary = self.first_markov_boundary[self.remaining_ids[target_var]][self.remaining_ids]
            # if in test, take the mask.
            if test_mode:
                # get a list of markov boundary indexes.
                markov_boundary_ids = np.where(markov_boundary * first_markov_boundary == 1)[0].tolist()
            else:
                # get a list of markov boundary indexes.
                markov_boundary_ids = np.where(markov_boundary == 1)[0].tolist()

            # assume all vertices are neighbors, then remove co parents
            neighbors = len(markov_boundary_ids)
            for y in markov_boundary_ids:
                self.generated_graph[self.remaining_ids[target_var]][self.remaining_ids[y]] = 1

            for y in markov_boundary_ids:
                for z in markov_boundary_ids:
                    if z != y and self.oracle_CI(target_var, y,
                                               [e for e in markov_boundary_ids if e != y and e != z],
                                               ):
                        neighbors -= 1
                        # remove Y from the X in self.generated_graph which were all one
                        self.generated_graph[self.remaining_ids[target_var]][self.remaining_ids[y]] = 0
                        break

            # return an array with neighbors and the reward of this step.
            return -neighbors

    def empirical_CI(self, x, y, s, ):
        # inverse of a cdf of normal distribution
        c = norm.ppf(1 - self.alpha / 2)
        # get the proposed columns of data
        truncated_data = self.data[:, np.array([x, y] + s)]
        corrcoef_matrix = np.corrcoef(truncated_data, rowvar=False)
        if truncated_data.shape[1] == 1:
            corrcoef_matrix = np.array([[corrcoef_matrix]])
        perecion_matrix = np.linalg.inv(corrcoef_matrix)
        partial_corrolation = perecion_matrix[0, 1] / (perecion_matrix[0, 0] * perecion_matrix[1, 1]) ** 0.5
        threshold = c / (truncated_data.shape[0] - len(s) - 3) ** 0.5
        return abs(partial_corrolation) <= threshold

    def gcastle_test(self, x, y, s, ):
        n=self.data.shape[0]
        k = len(s)
        if k == 0:
            r = np.corrcoef(self.data[:, [x, y]].T)[0][1]
        else:
            sub_index = [x, y]
            sub_index.extend(s)
            sub_corr = np.corrcoef(self.data[:, sub_index].T)
            # inverse matrix
            try:
                PM = np.linalg.inv(sub_corr)
            except np.linalg.LinAlgError:
                PM = np.linalg.pinv(sub_corr)
            r = -1 * PM[0, 1] / math.sqrt(abs(PM[0, 0] * PM[1, 1]))
        cut_at = 0.99999
        r = min(cut_at, max(-1 * cut_at, r))  # make r between -1 and 1

        # Fisher’s z-transform
        res = math.sqrt(n - k - 3) * .5 * math.log1p((2 * r) / (1 - r))
        p_value = 2 * (1 - stats.norm.cdf(abs(res)))

        return p_value >= self.alpha

    def visualize(self):
        """Creates a visualization of the environment."""
        self._visualize = True
        print(self.render('ascii'))

    def close(self):
        """Close the env."""

    @property
    def render_modes(self):
        """list: A list of string representing the supported render modes."""
        return [
            'ascii',
        ]

    def remove_target(self, target_var):
        self._state[self.remaining_ids[target_var]] = 0
        self.remaining_ids = np.delete(self.remaining_ids, target_var)

    def reconstruct_mb(self, ):
        if self.oracle:
            return self.oracle_mb()
        else:
            return self.empirical_mb()

    def empirical_mb(self, ):
        beta = 2 / (len(self.remaining_ids) ** 2+0.00001)
        # inverse of a cdf of normal distribution
        c = norm.ppf(1 - beta / 2)
        # get the proposed columns of data
        truncated_data = self.data[:, self.remaining_ids]
        corrcoef_matrix = np.corrcoef(truncated_data, rowvar=False)
        if truncated_data.shape[1] == 1:
            corrcoef_matrix = np.array([[corrcoef_matrix]])
        perecion_matrix = np.linalg.inv(corrcoef_matrix)
        threshold = c / (truncated_data.shape[0] - truncated_data.shape[1] - 1) ** 0.5
        Mb = np.zeros_like(perecion_matrix)
        for i in range(len(perecion_matrix)):
            for j in range(len(perecion_matrix)):
                if i != j:
                    ro = perecion_matrix[i, j] / (perecion_matrix[i, i] * perecion_matrix[j, j]) ** 0.5
                    Mb[i, j] = 1 if abs(ro) > threshold else 0
        return Mb

    def oracle_CI(self, x, y, s, ):
        # construct the graphX
        rows, cols = np.where(self.adjacency_matrix[self.remaining_ids][:, self.remaining_ids] == 1)
        edges = zip(rows.tolist(), cols.tolist())
        gr = nx.DiGraph()
        gr.add_nodes_from([i for i in range(len(self.remaining_ids))])
        gr.add_edges_from(edges)
        conditioned_vertices = set(s)
        result = nx.d_separated(gr, {x}, {y}, conditioned_vertices)
        return result

    def oracle_mb(self):
        gr = self.adjacency_matrix[self.remaining_ids][:, self.remaining_ids]
        return self.real_markov_boundary(gr)

    def real_markov_boundary(self, graph):
        mb = graph.copy()
        for i in range(mb.shape[0]):
            child_ids = np.where(graph[i] == 1)[0]
            for j in child_ids:
                parents_of_the_child = np.where(graph[:, j] == 1)[0]
                mb[i][parents_of_the_child] = 1
                mb[i][i] = 0
        mb = mb + graph.T
        mb[mb > 0] = 1
        mb[mb <= 0] = 0
        return mb
