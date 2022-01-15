from abc import ABC, abstractmethod
from pathlib import Path
from enum import IntEnum

import numpy as np
import seaborn as sns

from tools.math_utils import random_noisy_states
from tools.utils import dump_pickle, load_pickle
sns.set()

DATA_PATH = 'data'
MODEL_PATH = 'models'


class Index(IntEnum):
    READY = 1,
    READY_SET = 2,
    SET = 3,
    SET_GO = 4,
    GO = 5,
    END = 6


# TODO: Edit
class Analyzer(ABC):
    results_path = 'analysis_results'

    def __init__(self, model, data_params, model_name, instance, outputs, states, checkpoint=None):
        self.model = model
        self.X, self.Y = data_params.get_data()
        self.data_params = data_params
        self.model_name = model_name
        self.instance = instance
        self.checkpoint = checkpoint
        self.data_params.get_data
        # self.X = self.data_params.x
        # self.Y = self.data_params.y
        self.outputs, self.states = outputs, states

    def save_plot(self, plt, desc=''):
        # save analysis-major
        path_analysis = f'analysis_results/{self.name}/'
        analysis_model_name = f'{desc}_{self.model_name}_i{self.instance}'
        if self.checkpoint is not None:
            path_analysis += 'valid_checkpoints/'
            analysis_model_name += f'_chkpt{self.checkpoint}'

        Path(path_analysis).mkdir(parents=True, exist_ok=True)
        plt.savefig(path_analysis + analysis_model_name + '.pdf', bbox_inches='tight')
        plt.close()
        return
        # save model major
        path_model = f'analysis_results/models/{self.model_name}/'
        if self.checkpoint is None:
            analysis_full_name = f'i{self.instance}_{self.name}_{desc}.png'
        else:
            path_model += 'checkpoints/'
            analysis_full_name = f'i{self.instance}_chkpt{self.checkpoint}_{self.name}_{desc}.png'

        Path(path_model).mkdir(parents=True, exist_ok=True)
        plt.savefig(path_model + analysis_full_name, bbox_inches='tight')
        plt.close()

    def save_file(self, file, desc):
        if self.checkpoint is None:
            path = f'models/{self.model_name}/i{self.instance}/{desc}.pkl'
        else:
            path = f'models/{self.model_name}/i{self.instance}/valid_checkpoints/chkpt{self.checkpoint}_{desc}.pkl'
            Path(f'models/{self.model_name}/i{self.instance}/valid_checkpoints/').mkdir(parents=True, exist_ok=True)

        dump_pickle(path, file)

    def load_file(self, desc):
        path = f'models/{self.model_name}/i{self.instance}/{desc}.pkl'
        return load_pickle(path)

    @property
    @abstractmethod
    def name(self):
        pass

    @abstractmethod
    def run(self):
        pass


class RSGTrialsAnalyzer(Analyzer):
    def __init__(self, model, data_params, model_name, inst, outputs, states, checkpoints=None):
        super().__init__(model, data_params, model_name, inst, outputs, states, checkpoints)
        #self.X, self.Y, self.outputs, self.states = self.preprocess_data()
        self.low_ts = self.data_params.low_ts
        self.high_ts = self.data_params.high_ts
        self.pulse = data_params.pulse
        self.ts_dict = self.data_params.transition_indices
        #self.ts_dict = self._get_transition_indices()


    def preprocess_data(self):
        x, y = self.data_params.get_validation_trials()
        pred = self.model.run_system_from_input(x)
        x = self.data_params.postprocess_data(x)
        y = self.data_params.postprocess_data(y)
        outputs = self.data_params.postprocess_data(pred['output'])
        states = self.data_params.postprocess_data(pred['state'])
        return x, y, outputs, states

    def adjust_input_and_run_system(self, x):
        pred = self.model.run_system_from_input(self.data_params.preprocess(x))
        pred['output'] = self.data_params.postprocess_data(pred['output'])
        pred['state'] = self.data_params.postprocess_data(pred['state'])
        return pred

    def run_system_from_noisy_states(self, initial_state, sigma, steps, reps):
        init_noisy = random_noisy_states(initial_state,
                                         reps=reps,
                                         sigma=sigma)
        init_noisy[:,:-self.model.units] = initial_state[:-self.model.units]
        pred = self.model.run_system_from_inits(init_states=init_noisy, steps=steps)
        pred['state'] = pred['state'][:,:,-self.model.units:]
        return pred


    def get_clock_init_idx(self, tt):
        return self.ts_dict[tt][4]

    def get_clock_init(self, tt, start_offset=0):
        return self.states[tt - self.low_ts][self.ts_dict[tt][4] + start_offset]

    def get_clock_end(self, tt, end_offset=0):
        return self.states[tt - self.low_ts][self.ts_dict[tt][5] - end_offset]

    def get_clock_inits(self, start_offset=0):
        dims = self.states.shape[-1]
        inits = np.zeros((self.high_ts - self.low_ts + 1, dims))
        for tt in range(self.low_ts, self.high_ts + 1):
            inits[tt - self.low_ts] = self.get_clock_init(tt, start_offset)

        return inits

    def get_clock_ends(self, end_offset=0):
        dims = self.states.shape[-1]
        inits = np.zeros((self.high_ts - self.low_ts + 1, dims))
        for tt in range(self.low_ts, self.high_ts + 1):
            inits[tt - self.low_ts] = self.get_clock_end(tt, end_offset)

        return inits

    def get_all_clock(self, ts, start_offset=0, end_offset=0):
        #if tt - 10 - start_offset - end_offset >= 0
        clock = self.states[ts - self.low_ts][self.ts_dict[ts][Index.SET_GO] + start_offset: self.ts_dict[ts][Index.GO] - end_offset]
        return clock[:]

    def get_go_start_idx(self, tt):
        return self.ts_dict[tt][Index.GO]

    def get_set_init_idx(self, ts):
        return self.ts_dict[ts][Index.SET]

    def get_set_init(self, ts, start_offset=0):
        return self.states[ts - self.low_ts][self.ts_dict[ts][Index.SET] + start_offset]

    def get_ready_set(self, ts, start_offset=0, end_offset=0):
        assert ts - 10 - start_offset - end_offset >= 0
        return self.states[ts - self.low_ts][self.ts_dict[ts][Index.READY_SET] + start_offset: self.ts_dict[ts][Index.SET]-end_offset]

    def get_r_theta(self, ts, start_offset=0, normalize=False):
        assert ts > self.low_ts
        init_clock_prev = self.get_clock_init(ts - 1, start_offset)
        init_clock = self.get_clock_init(ts, start_offset)
        init_clock_p1 = self.get_clock_init(ts, start_offset + 1)
        r = init_clock - init_clock_prev
        theta = init_clock_p1 - init_clock

        if normalize:
            r /= np.linalg.norm(r)
            theta /= np.linalg.norm(theta)

        return r, theta

    def get_manifold(self, start_offset=0, end_offset=0, low_ts=None, high_ts=None):
        low_ts = self.low_ts if low_ts is None else low_ts
        high_ts = self.high_ts if high_ts is None else high_ts
        manifold_lst = []
        for ts in range(low_ts, high_ts + 1):
            if ts - 10 - start_offset - end_offset >= 0:
                ts_clock = self.get_all_clock(ts, start_offset, end_offset)[:]
                manifold_lst.append(ts_clock)

        manifold = np.concatenate(manifold_lst)
        return manifold

    def get_ts_vs_tp(self, outputs=None):
        if outputs is None:
            outputs = self.outputs.squeeze()
        else:
            assert outputs.shape[0] == self.outputs.shape[0]
        actual_tt = {}
        for tt in range(self.low_ts, self.high_ts + 1):
            indices = np.argwhere(outputs[tt - self.low_ts] > 0.8).squeeze()
            if indices.size == 0:
                continue
            elif indices.size == 1:
                indices = [indices.item()]
            else:
                indices = [indices[0]] + [indices[i] for i in range(1, indices.shape[0]) if
                                          indices[i] - indices[i - 1] > 1]
            idx = indices[0]
            if idx > self.ts_dict[tt][3]:
                actual_tt[tt] = idx - self.ts_dict[tt][3]

        return actual_tt

    def get_ts_vs_tps(self, outputs=None):
        if outputs is None:
            outputs = self.outputs.squeeze()
        else:
            assert outputs.shape[0] == self.outputs.shape[0]
        actual_tt = {}
        for tt in range(self.low_ts, self.high_ts + 1):
            indices = np.argwhere(outputs[tt - self.low_ts] > 0.5).squeeze()
            if indices.size == 0:
                actual_tt[tt] = []
            elif indices.size == 1:
                actual_tt[tt] = [indices.item() - self.ts_dict[tt][3]]
            else:
                indices = [indices[0]] + [indices[i] for i in range(1, indices.shape[0]) if
                                          indices[i] - indices[i - 1] > 5]
                actual_tt[tt] = [idx - self.ts_dict[tt][3] for idx in indices]

        return actual_tt

    def get_set_manifold(self):
        manifold_lst = []
        manifold_dict = {}
        for ts in range(self.low_ts, self.high_ts + 1):
            set_pulse = self.states[ts-self.low_ts, self.ts_dict[ts][3]-1:self.ts_dict[ts][4]+1]
            manifold_dict[ts] = set_pulse
            manifold_lst.append(set_pulse)

        manifold = np.concatenate(list(manifold_dict.values()))
        return manifold, manifold_dict


    def add_noise_set_go(self, sigma, rep):
        duration = len(NOISE[rep])
        noisy_x = np.copy(self.X)
        for ts in [60, 90]:#range(self.low_ts + 30, self.high_ts + 1):
            start_idx = self.ts_dict[ts][4] + 20
            end_idx = min(self.ts_dict[ts][5], start_idx + duration)
            noisy_x[ts - self.low_ts, start_idx:end_idx] += sigma*NOISE[rep]#np.random.normal(loc=0, scale=sigma, size=(end_idx-start_idx, 1))

        return noisy_x[np.array([30, 60])]


    def get_output_matrix(self, low_ts, high_ts, pulse, steps=None):
        if steps is None:
            steps = 2*high_ts + pulse*10
        matrix = np.zeros((high_ts - low_ts + 1, steps, 2))
        matrix[:,pulse:2*pulse,0] = 1
        for ts in range(low_ts, high_ts+1):
            matrix[ts - low_ts, pulse+ts:2*pulse+ts,1] = 1

        output_matrix = self.model.predict(matrix)['output'][:,pulse:]
        return output_matrix


