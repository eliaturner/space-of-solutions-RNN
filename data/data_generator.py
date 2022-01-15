from abc import abstractmethod, ABC
from scipy.stats import logistic
from pathlib import Path
from tools.utils import load_pickle, dump_pickle
import numpy as np
from cached_property import cached_property
from tools.rsg_utils import transition_indices
from collections import defaultdict

class RNNDataGenerator(ABC):
    @property
    @abstractmethod
    def n_inputs(self):
        pass

    @property
    @abstractmethod
    def n_outputs(self):
        pass

    @property
    @abstractmethod
    def steps(self):
        pass

    @property
    @abstractmethod
    def name(self):
        pass

    @abstractmethod
    def task_name(self):
        pass

    @abstractmethod
    def generate_train_data(self):
        pass

    @abstractmethod
    def generate_validation_data(self):
        pass

    def preprocess_data(self, x):
        return x

    def postprocess_data(self, x):
        return x

    def predict_on_data(self, model):
        x, y = self.get_data()
        pred = model.run_system_from_input(x)
        x = self.postprocess_data(x)
        y = self.postprocess_data(y)
        outputs = self.postprocess_data(pred['output'])
        states = self.postprocess_data(pred['state'])
        return outputs, states

    def get_data(self):
        data_path = 'data/' + self.name
        if False and Path(data_path + '/x.pkl').is_file():
            x = load_pickle(data_path + '/x.pkl')
            y = load_pickle(data_path + '/y.pkl')
        else:
            x, y = self.generate_validation_data()
            Path(data_path).mkdir(parents=True, exist_ok=True)
            dump_pickle(data_path + '/x.pkl', x)
            dump_pickle(data_path + '/y.pkl', y)

        return x, y
    
    def initialize_data_placeholder(self, n_samples):
        x = np.zeros([n_samples, self.steps, self.n_inputs])
        y = np.zeros([n_samples, self.steps, self.n_outputs])
        
        return x, y


class RSGDataGenerator(RNNDataGenerator):
    n_inputs = 2
    n_outputs = 1

    def __init__(self, low_ts=30, high_ts=120, pulse=10):
        self.low_ts = low_ts
        self.high_ts = high_ts
        self.pulse = pulse
        self.n_trials = 128
        self.x = None
        self.y = None
        self.transition_indices = None

    def task_name(self):
        return 'rsg_single'

    @property
    def name(self):
        if self.high_ts == 120:
            return self.task_name()

        return self.task_name() + '_' + str(self.high_ts)

    @property
    def steps(self):
        return 2*self.high_ts + 50

    def generate_train_data(self):
        x_train, y_train = self.generate_rsg_train_trials()
        return x_train, y_train

    def generate_validation_data(self):
        x_val, y_val = self.generate_rsg_validation_trials()
        return x_val, y_val

    @cached_property
    def data(self):
        self.x, self.y = self.generate_validation_data()
        self.transition_indices = self._get_transition_indices()
        return self.x, self.y

    def get_data(self):
        return self.data

    def generate_rsg_trial(self, ts, x_trial, y_trial, start=0):
        ready = np.random.randint(low=self.pulse, high=self.pulse+1) + start
        set, go = ready + ts, ready + 2*ts
        if go + self.pulse >= self.steps:
            return self.steps
        x_trial[ready:ready + self.pulse, 0] = 1
        x_trial[set:set + self.pulse, -1] = 1
        y_trial[go:go + self.pulse, 0] = 1
        return go + self.pulse

    def generate_rsg_validation_trials(self):
        n_trials = (self.high_ts - self.low_ts + 1)
        x, y = self.initialize_data_placeholder(n_samples=n_trials)
        for trial in range(0, n_trials):
            ts = trial + self.low_ts
            self.generate_rsg_trial(ts, x[trial], y[trial])

        return x, y

    def generate_rsg_train_trials(self):
        factor = 1
        x, y = self.initialize_data_placeholder(n_samples=self.n_trials)
        for trial in range(self.n_trials):
            ts = np.random.randint(low=self.low_ts // factor, high=self.high_ts // factor + 1) * factor
            self.generate_rsg_trial(ts, x[trial], y[trial])

        return x, y

    def _get_transition_indices(self, inputs=None, outputs=None):
        ts_dict = {}
        if inputs is None:
            inputs = self.x
            outputs = self.y

        for ts in range(self.low_ts, self.high_ts + 1):
            input = inputs[ts - self.low_ts]
            output = outputs[ts - self.low_ts]
            ts_dict[ts] = transition_indices(input, output)

        return ts_dict

    def partition_to_epochs(self, outputs, states):
        trial_epoch_state_dict = defaultdict(dict)
        trial_epoch_output_dict = defaultdict(dict)
        epochs = ['START', 'READY', 'READY_SET', 'SET', 'SET_GO']
        i = 0
        for ts in range(self.low_ts, self.high_ts + 1):
            indices = self.transition_indices[ts]
            for j, epoch in enumerate(epochs):
                trial_epoch_state_dict[ts][epoch] = states[i, indices[j]:indices[j+1]]
                trial_epoch_output_dict[ts][epoch] = outputs[i, indices[j]:indices[j+1]]
            i += 1

        return trial_epoch_output_dict, trial_epoch_state_dict

    def epoch1_epoch2(self, outputs, states):
        o_dict, s_dict = self.partition_to_epochs(outputs, states)
        epoch1 = s_dict[50]['READY_SET'][10:]
        epoch2 = [s_dict[t]['SET_GO'][5:t-5] for t in range(self.low_ts + 2, self.high_ts, 2)]
        return epoch1, epoch2


class IntervalReproduction(RSGDataGenerator):
    n_inputs = 2
    n_outputs = 1

    def task_name(self):
        return 'interval_reproduction'


class IntervalDiscrimination(RNNDataGenerator):
    n_inputs = 2
    n_outputs = 1
    task_name = 'interval_discrimination'

    def __init__(self):
        self.x, self.y = None, None
        self.low_t = 10
        self.high_t = 30
        self.transition_indices = None

    @property
    def steps(self):
        return 130


    @property
    def pulse(self):
        return 2

    @property
    def name(self):
        name = f'{self.task_name}_{self.steps}'
        return name

    def generate_trial(self, t1, t2, x_trial, y_trial):
        CONSTANT = 0
        # x_trial[5:10, 0] = 1
        x_trial[CONSTANT +t1:CONSTANT + t1 + self.pulse, 0] = 1
        x_trial[CONSTANT + t1 + t2:CONSTANT + t1 + t2 + self.pulse, 1] = 1
        y_trial[CONSTANT + 15 + t1 + t2:CONSTANT + 15 + t1 + t2 + self.pulse, 0] = np.sign(t1 - t2)
        # y_trial[CONSTANT + 15 + t1 + t2:, 0] = np.sign(t1 - t2)

    def generate_train_data(self):
        return self.generate_validation_data()
        N = 150
        x = np.zeros((N,self.steps, 2))
        y = np.zeros((N,self.steps, 1))
        for i in range(N):
            # x[i,10:20,0] = 1
            t1, t2 = np.random.choice(np.arange(self.low_t, self.high_t + 1, 2), 2, replace=False)
            self.generate_trial(t1, t2, x[i], y[i])

        return x, y

    def generate_validation_data(self):
        intervals = np.arange(self.low_t, self.high_t + 1, 1)
        N = len(intervals)**2 - len(intervals)
        steps = 2* max(intervals) + 30
        x = np.zeros((N,steps, 2))
        y = np.zeros((N,steps, 1))
        i = 0
        for t1 in intervals:
            for t2 in intervals:
                if t1 != t2:
                    self.generate_trial(t1, t2, x[i], y[i])
                    i += 1

        return x, y


    @cached_property
    def data(self):
        self.x, self.y = self.generate_validation_data()
        self.transition_indices = self._get_transition_indices()
        return self.x, self.y

    def get_data(self):
        return self.data

    def _get_transition_indices(self, inputs=None, outputs=None):
        ts_dict = defaultdict(dict)
        if inputs is None:
            inputs = self.x
            outputs = self.y

        i = 0
        for t1 in range(self.low_t, self.high_t + 1, 1):
            for t2 in range(self.low_t, self.high_t + 1, 1):
                if t1 == t2:
                    continue

                input = inputs[i]
                output = outputs[i]
                ts_dict[t1][t2] = transition_indices(input, None)
                i += 1

        return ts_dict

    def partition_to_epochs(self, outputs, states):
        trial_epoch_state_dict = defaultdict(dict)
        trial_epoch_output_dict = defaultdict(dict)
        epochs = ['T1', 'PULSE1', 'T2', 'PULSE2', 'OUTPUT']
        i = 0
        for t1 in range(self.low_t, self.high_t + 1, 1):
            trial_epoch_state_dict[t1] = defaultdict(dict)
            trial_epoch_output_dict[t1] = defaultdict(dict)
            for t2 in range(self.low_t, self.high_t + 1, 1):
                if t1 == t2:
                    continue
                indices = self.transition_indices[t1][t2]
                for j, epoch in enumerate(epochs):
                    trial_epoch_state_dict[t1][t2][epoch] = states[i, max(indices[j]-1,0):indices[j+1]]
                    trial_epoch_output_dict[t1][t2][epoch] = outputs[i, max(indices[j]-1,0):indices[j+1]]
                i += 1

        return trial_epoch_output_dict, trial_epoch_state_dict

    def epoch1_epoch2(self, outputs, states):
        o_dict, s_dict = self.partition_to_epochs(outputs, states)
        epoch1 = s_dict[30][10]['T1'][8:]
        epoch2 = [s_dict[20][t]['T2'][8:] for t in [12, 14, 16, 18, 22, 24, 26, 28]]
        return epoch1, epoch2


class DelayedDiscrimination(RNNDataGenerator):
    n_inputs = 2
    n_outputs = 1
    task_name = 'delayed_discrimination'

    def __init__(self):
        self.x, self.y = None, None
        self.low_t = 2
        self.high_t = 10
        self.transition_indices = None
        self.max_delay = 20


    def generate_trial(self, t1, t2, delay=0, steps=None):
        if steps is None:
            steps = self.steps
        x = np.zeros((steps, 2))
        y = np.zeros((steps, 1))
        x[5:5 + self.pulse, 0] = t1
        if delay is None:
            delay = np.random.randint(0, self.max_delay)
        # x[10:15, 0] = np.random.normal(0,0.01, 5)
        x[15+delay:15 +delay+ self.pulse, 1] = t2
        y[30+delay:30+delay + self.pulse, 0] = np.sign(t1 - t2)
        return x, y

    def generate_train_data(self):
        low_t, high_t = self.low_t, self.high_t
        N = (high_t - low_t + 1) ** 2 - (high_t - low_t + 1)
        N = N*self.max_delay
        x = np.zeros((N,self.steps, 2))
        y = np.zeros((N,self.steps, 1))
        i = 0
        for t1 in range(self.low_t, self.high_t + 1):
            for t2 in range(self.low_t, self.high_t + 1):
                if t1 == t2:
                    continue
                for delay in range(0, self.max_delay):
                    x[i], y[i] = self.generate_trial(t1, t2, delay=delay)
                    i += 1

        # for i in range(N):
        #     t1, t2 = np.random.choice(np.arange(2, 11), 2, replace=False)
        #     x[i], y[i] = self.generate_trial(t1, t2, random=True)

        return x, y

    def generate_validation_data(self):
        intervals = np.arange(self.low_t, self.high_t + 1)
        N = len(intervals)**2 - len(intervals)
        steps = self.steps + 0
        x = np.zeros((N,steps, 2))
        y = np.zeros((N,steps, 1))
        i = 0
        for t1 in intervals:
            for t2 in intervals:
                if t1 == t2:
                    continue
                x[i], y[i] = self.generate_trial(t1, t2, self.max_delay-1, steps)
                i += 1

        return x, y

    @property
    def steps(self):
        return 40 + self.max_delay

    @property
    def data(self):
        self.x, self.y = self.generate_validation_data()
        self.transition_indices = self._get_transition_indices()
        return self.x, self.y

    def get_data(self):
        return self.data

    def _get_transition_indices(self, inputs=None, outputs=None):
        ts_dict = defaultdict(dict)
        if inputs is None:
            inputs = self.x
            outputs = self.y

        i = 0
        for t1 in range(self.low_t, self.high_t + 1):
            for t2 in range(self.low_t, self.high_t + 1):
                if t1 == t2:
                    continue

                input = inputs[i]
                output = outputs[i]
                ts_dict[t1][t2] = transition_indices(input, None)
                i += 1

        return ts_dict

    def partition_to_epochs(self, outputs, states):
        trial_epoch_state_dict = defaultdict(dict)
        trial_epoch_output_dict = defaultdict(dict)
        epochs = ['START', 'PULSE1', 'WAIT', 'PULSE2', 'OUTPUT']
        i = 0
        for t1 in range(self.low_t, self.high_t + 1):
            trial_epoch_state_dict[t1] = defaultdict(dict)
            trial_epoch_output_dict[t1] = defaultdict(dict)
            for t2 in range(self.low_t, self.high_t + 1):
                if t1 == t2:
                    trial_epoch_state_dict[t1][t2] = None
                    trial_epoch_output_dict[t1][t2] = None
                    continue
                indices = self.transition_indices[t1][t2]
                for j, epoch in enumerate(epochs):
                    trial_epoch_state_dict[t1][t2][epoch] = states[i, indices[j]:indices[j+1]]
                    trial_epoch_output_dict[t1][t2][epoch] = outputs[i, indices[j]:indices[j+1]]
                i += 1

        return trial_epoch_output_dict, trial_epoch_state_dict

    def epoch1_epoch2(self, outputs, states):
        o_dict, s_dict = self.partition_to_epochs(outputs, states)
        epoch3 = [s_dict[2][10]['OUTPUT'][-10:], s_dict[10][2]['OUTPUT'][-10:]]
        epoch2 = [s_dict[t][10]['WAIT'][5:] for t in range(3, 10)]
        return epoch2, epoch3

    @property
    def pulse(self):
        return 5

    # @property
    # def max_delay(self):
    #     return 20

    @property
    def name(self):
        name = f'{self.task_name}_{self.steps}'
        return name


