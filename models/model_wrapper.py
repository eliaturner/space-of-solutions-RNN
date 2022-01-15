import itertools
from pathlib import Path
from abc import abstractmethod, ABC

import numpy as np
import sklearn
from dataclasses import dataclass
from sklearn.metrics import mean_squared_error

from tools.utils import dump_pickle, load_pickle

PULSE = 10

def save_weights(weights, path, name):
    import torch
    torch.save(weights, f'{path}/{name}.pt')


def load_weights(path, name, map_location='cpu'):
    import torch
    return torch.load(f'{path}/{name}.pt', map_location=map_location)

@dataclass
class OptimizationParameters:
    batch_size: int
    losses: list
    epochs: int
    minimal_loss: float


def unison_shuffled_copies(a, b):
    assert len(a) == len(b)
    p = np.random.permutation(len(a))
    return a[p], b[p]


def check_if_valid_name_later(out, y):
    T = []
    P = []
    for trial in range(out.shape[0]):
        a = np.argwhere(out[trial] > 0.4).flatten()
        if a.size == 0:
            return False
        a = a[0]
        b = np.argwhere(y[trial] > 0.4).flatten()[0]
        T.append(b)
        P.append(a)

    score = mean_squared_error(T, P)
    return score < 2


class InnerModelWrapper(ABC):
    def __init__(self, architecture, name, instance):
        self.architecture = architecture
        self.instance = instance
        self.name = name
        Path(self.model_path).mkdir(parents=True, exist_ok=True)
        self.architecture.set_model_dir(self.model_path)

    @property
    def model_path(self):
        return f'models/{self.name}/i{self.instance}'

    @property
    def model_name(self):
        return f'{self.name}_i{self.instance}'

    def retrain_model_from_initial_weights(self):
        self.architecture.load_weights()
        self.train_model(weights=self.architecture.initial_weights())

    def train_model(self, x_train, y_train, x_val, y_val, optimization_params, weights=None, shuffle=True):
        print('Training {}, instance {}'.format(self.name, self.instance))
        return self.architecture.train(x_train, y_train, x_val, y_val, optimization_params, weights,
                                           shuffle=shuffle)

    def get_initial_weights(self):
        return self.architecture.initial_weights()

    def get_weights(self):
        return self.architecture.get_weights()

    def analyze(self, analyses, data_params):
        print(self.model_name, self.instance)
        self.architecture.load_weights()
        x, _ = data_params.get_data()
        pred = self.architecture.predict(x)
        for analysis_class in analyses:
            analysis_class(self.architecture, data_params, self.name, self.instance, pred['output'], pred['state']).run()

    def get_analysis(self, analyses, data_params):
        print(self.model_name, self.instance)
        results = {}
        self.architecture.load_weights()
        x, _ = data_params.get_data()
        pred = self.architecture.predict(x)
        for analysis_class in analyses:
            results[analysis_class] = analysis_class(self.architecture, data_params, self.name, self.instance, pred['output'], pred['state']).run()

        return results

    def get_analysis_checkpoints(self, analyses, data_params, checkpoints=None):
        results = {analysis_class: {} for analysis_class in analyses}
        x, y = data_params.get_data()
        print('analyzing inst {}'.format(self.instance))
        checkpoints_dict = self.get_checkpoints_weights(checkpoints, valid=True)
        print(checkpoints_dict.keys())
        for chkpt in checkpoints_dict.keys():
            self.architecture.load_weights(checkpoints_dict[chkpt])
            pred = self.architecture.predict(x)
            outputs, states = pred['output'], pred['state']
            for analysis_class in analyses:
                results[analysis_class][chkpt] = analysis_class(self.architecture,
                                                               data_params, self.name, self.instance,
                                                               outputs, states, chkpt).run()

        return results

    def get_checkpoints_weights(self, checkpoints=None, valid=False):
        checkpoints_dict = {}
        checkpoints_path = self.model_path
        if valid:
            checkpoints_path += '/valid_checkpoints'

        if checkpoints is None:
            checkpoints_dir = Path(checkpoints_path)
            checkpoints_files = [x for x in checkpoints_dir.iterdir() if 'weights' in x.name and 'weights.pt' not in x.name]
            for file in checkpoints_files:
                chkpt = int(file.name.rstrip('.pt').lstrip('weights'))
                checkpoints_dict[chkpt] = load_weights(checkpoints_path, f'weights{chkpt}')

        else:
            for chkpt in checkpoints:
                checkpoints_dict[chkpt] = load_weights(checkpoints_path, f'weights{chkpt}')

        return checkpoints_dict

    def get_file_checkpoints(self, filename):
        return load_pickle(f'{self.model_path}/{filename}_checkpoints.pkl')

    def get_file(self, filename):
        return load_pickle(f'{self.model_path}/{filename}.pkl')

    def check_if_valid(self, x, y, weights):
        self.architecture.load_weights(weights)
        pred = self.architecture.predict(x)
        return check_if_valid_name_later(pred['output'], y)

    def validation_scores(self, x, y):
        self.architecture.load_weights()
        input = np.zeros((3, 2000, 2))
        # input[1:,10:20,0] = 1
        input[1:,70:80,0] = 1
        input[2,170:180,1] = 1
        pred = self.architecture.predict(x)
        import matplotlib.pyplot as plt
        from sklearn.decomposition import PCA
        from analysis.dynamics_to_diagram2 import plot_n_states
        # states = [pred['state'][i, 10+60+20+10*i+20:10+60+20+10*i+40] for i in [3,4,5]]
        # plt.clf()
        # plt.plot(pred['output'][4])
        # plt.title(self.model_name)
        # plt.show()
        plt.clf(); plt.plot(pred['output'][18:23,50:150].transpose());plt.show()
        idx = np.array([19, 21])
        states = [pred['state'][18, 75:95], pred['state'][19, 85:105], pred['state'][20, 95:115], pred['state'][21, 105:125], pred['state'][22, 115:135]]
        plot_n_states(states)
        print(self.model_name)
        return
        print()

    def validation_scores1(self, x, y):
        new_path = self.model_path + '/valid_checkpoints'
        Path(new_path).mkdir(parents=True, exist_ok=True)
        print('analyzing inst {}'.format(self.instance))
        self.architecture.set_model_dir(self.model_path)
        checkpoints_dict = self.get_checkpoints_weights(valid=False)
        valid_checkpoints = []
        valids = 0
        for chkpt in checkpoints_dict.keys():
            is_valid = self.check_if_valid(x, y, checkpoints_dict[chkpt])
            if is_valid:
                save_weights(checkpoints_dict[chkpt], new_path, f'weights{valids}')
                valids += 1
                valid_checkpoints.append(chkpt)

        save_weights(self.get_weights(), new_path, f'weights{valids}')

    def replace_file_checkpoints(self, filename_old, filename_new):
        checkpoints_dict = self.get_checkpoints_weights(None, valid=True)
        checkpoints = checkpoints_dict.keys()
        model_path = self.model_path
        new_path = model_path + '/valid_checkpoints'
        for chkpt in checkpoints:
            dump_pickle(f'{new_path}/chkpt{chkpt}_{filename_new}.pkl', load_pickle(f'{new_path}/chkpt{chkpt}_{filename_old}.pkl'))

    def group_file_checkpoints(self, filename):
        checkpoints_dict = self.get_checkpoints_weights(None, valid=True)
        checkpoints = checkpoints_dict.keys()
        model_path = self.model_path
        new_path = model_path + '/valid_checkpoints'
        matrix = np.stack([load_pickle(f'{new_path}/chkpt{chkpt}_{filename}.pkl') for chkpt in checkpoints])
        dump_pickle(f'{model_path}/{filename}_checkpoints.pkl', matrix)

    def loss_through_time(self, x, y):
        loss_over_time = []
        print('analyzing inst {}'.format(self.instance))
        checkpoints_dict = self.get_checkpoints_weights(valid=False)
        for chkpt in checkpoints_dict.keys():
            self.architecture.load_weights(checkpoints_dict[chkpt])
            pred = self.architecture.predict(x)['output']
            loss = sklearn.metrics.mean_squared_error(y.squeeze(), pred)
            loss_over_time.append(loss)

        return loss_over_time


class ModelWrapper(ABC):
    def __init__(self, architecture, train_data, test_data, instance_range, color='black', optimization_params=None):
        self.name = train_data.name + '_' + architecture.name
        self.architecture = architecture
        if optimization_params is None:
            optimization_params = OptimizationParameters(64, ['mse'], 10000, 1e-4)
        self.optimization_params = optimization_params
        self.train_data = train_data
        self.test_data = test_data
        self.instance_range = instance_range
        self.color = color

    def train_model(self, weights=None, shuffle=True):
        generator = self.train_data
        for inst in self.instance_range:
            inner_wrapper = InnerModelWrapper(self.architecture, self.name, inst)
            while True:
                x_train, y_train = generator.generate_train_data()
                x_val, y_val = generator.generate_validation_data()
                train_on = inner_wrapper.train_model(x_train, y_train, x_val, y_val, self.optimization_params, weights, shuffle)
                if not train_on:
                    break

    def train_model_with_data(self, x, y, weights=None, shuffle=True):
        for inst in self.instance_range:
            inner_wrapper = InnerModelWrapper(self.architecture, self.name, inst)
            while True:
                train_on = inner_wrapper.train_model(x, y, x, y, self.optimization_params, weights, shuffle)
                if not train_on:
                    break

    def grid_search(self, weights=None):
        generator = self.train_data
        print('Training {}'.format(self.name))
        x_val, y_val = generator.generate_validation_data()
        x_train, y_train = x_val, y_val
        self.architecture.grid_search(x_train, y_train, weights)

    def train_identical_models(self, weights=None, shuffle=False, shuffle_once=False):
        assert not (shuffle and shuffle_once)
        generator = self.train_data
        x_train, y_train = generator.generate_train_data()
        x_val, y_val = x_train, y_train

        if type(weights) == np.ndarray or weights is None:
            weights = {inst: weights for inst in self.instance_range}

        for inst in self.instance_range:
            inner_wrapper = InnerModelWrapper(self.architecture, self.name, inst)
            if shuffle_once:
                x_train, y_train = unison_shuffled_copies(x_train, y_train)
            inner_wrapper.train_model(x_train, y_train, x_val, y_val, self.optimization_params, weights[inst], shuffle)

    def analyze(self, analyses):
        for inst in self.instance_range:
            inner_wrapper = InnerModelWrapper(self.architecture, self.name, inst)
            inner_wrapper.analyze(analyses, self.test_data)

    def get_analysis(self, analyses):
        results = {analysis_class : {} for analysis_class in analyses}
        for inst in self.instance_range:
            inner_wrapper = InnerModelWrapper(self.architecture, self.name, inst)
            res_inst = inner_wrapper.get_analysis(analyses, self.test_data)
            for analysis_class in analyses:
                results[analysis_class][inst] = res_inst[analysis_class]

        if len(analyses) == 1:
            results = list(results[analyses[0]].values())

        return results

    def loss_through_time(self, x, y):
        loss_over_time = {}
        for inst in self.instance_range:
            inner_wrapper = InnerModelWrapper(self.architecture, self.name, inst)
            loss_over_time[inst] = inner_wrapper.loss_through_time(x, y)

        return loss_over_time

    def validation_score(self):
        self.test_data.high_ts = 120
        x, y = self.test_data.get_data()
        for inst in self.instance_range:
            inner_wrapper = InnerModelWrapper(self.architecture, self.name, inst)
            inner_wrapper.validation_scores(x, y)

    def group_file_checkpoints(self, filename):
        for inst in self.instance_range:
            inner_wrapper = InnerModelWrapper(self.architecture, self.name, inst)
            inner_wrapper.group_file_checkpoints(filename)

    def replace_file_checkpoints(self, filename_old, filename_new):
        for inst in self.instance_range:
            inner_wrapper = InnerModelWrapper(self.architecture, self.name, inst)
            inner_wrapper.replace_file_checkpoints(filename_old, filename_new)

    def get_file(self, filename):
        res = {}
        for inst in self.instance_range:
            inner_wrapper = InnerModelWrapper(self.architecture, self.name, inst)
            res[inst] = inner_wrapper.get_file(filename)
        return res

    def get_file_checkpoints(self, filename):
        self.group_file_checkpoints(filename)
        res = {}
        for inst in self.instance_range:
            inner_wrapper = InnerModelWrapper(self.architecture, self.name, inst)
            res[inst] = inner_wrapper.get_file_checkpoints(filename)

        return res

    def get_analysis_checkpoints(self, analyses, checkpoints=None):
        results = {analysis_class: {inst:{} for inst in self.instance_range} for analysis_class in analyses}
        for inst in self.instance_range:
            inner_wrapper = InnerModelWrapper(self.architecture, self.name, inst)
            res_inst = inner_wrapper.get_analysis_checkpoints(analyses, self.test_data, checkpoints)
            for analysis_class in analyses:
                results[analysis_class][inst] = res_inst[analysis_class]

        return results
