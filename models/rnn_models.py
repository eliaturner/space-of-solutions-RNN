from abc import abstractmethod, ABC

PULSE = 10


class RNNModel(ABC):
    def __init__(self, units=70, inputs=1, outputs=1, activation='tanh', bias=True, initial_state=None, num_layers=1):
        self.activation = activation
        self.inputs = inputs
        self.outputs = outputs
        self.units = units
        self.model_dir = None
        self.weights = None
        self.bias = bias
        self.initial_state = initial_state
        self.num_layers = num_layers

    @abstractmethod
    def create_model(
            self,
            steps,
            initial_state=None,
            return_states=False,
            return_output=True,
            return_sequences=True,
            kill_neuron=None):
        pass

    def set_model_dir(self, model_dir):
        self.model_dir = model_dir

    @abstractmethod
    def load_weights(self):
        pass

    @abstractmethod
    def get_weights(self):
        pass

    @abstractmethod
    def assign_weights(self, model, weights=None):
        pass

    @abstractmethod
    def train(self, x_train, y_train, x_val, y_val, params, weights=None, shuffle=True):
        pass

    @property
    def name(self):
        name = self.rnn_type + '_' + str(self.units)
        if self.num_layers > 1:
            name += f'_{self.num_layers}'

        return name

    @property
    @abstractmethod
    def rnn_type(self):
        pass

