from abc import abstractmethod

import torch
import torch.nn as nn


class RNN(nn.Module):
    def __init__(self, input_size, output_size, units, return_states, initial_state, kill_neuron, bias, num_layers):
        super(RNN, self).__init__()

        self.units = units
        self.return_states = return_states
        self.initial_state = initial_state
        self.bias = bias
        if self.initial_state is not None:
            self.initial_state = self.initial_state.unsqueeze(0)
        self.rnn = self.rnn_class(input_size, units, batch_first=True, bias=self.bias, num_layers=num_layers)
        self.fc = nn.Linear(units, output_size, bias=self.bias)
        self.kill_neuron = kill_neuron

    @property
    @abstractmethod
    def rnn_class(self):
        pass

    def forward(self, x):
        if self.return_states:
            return self.forward_states(x)

        rnn_out, _ = self.rnn(x.float(), self.initial_state)
        readout = self.fc(rnn_out)
        return readout

    @abstractmethod
    def forward_states(self, x):
        pass


class Vanilla(RNN):
    @property
    def rnn_class(self):
        return nn.RNN

    def forward_states1(self, x):
        rnn_out, _ = self.rnn(x.float(), self.initial_state)
        readout = self.fc(rnn_out)
        return readout, rnn_out

    def forward_states(self, x):
        hidden = self.initial_state
        hs = []
        readouts = []
        steps = x.shape[1]

        for step in range(steps):
            temp, hidden = self.rnn(x[:, step:step + 1], hidden)
            if self.kill_neuron is not None:
                if type(self.kill_neuron) == int:
                    hidden[:, :, self.kill_neuron] = 0
                else:
                    for neuron in self.kill_neuron:
                        hidden[:, :, neuron] = 0

            hs.append(hidden.detach())
            readouts.append(self.fc(temp).detach())

        hs = torch.cat(hs).transpose(0, 1)
        readouts = torch.cat(readouts, 1)
        return readouts, hs


class GRU(RNN):
    @property
    def rnn_class(self):
        return nn.GRU

    def forward_states1(self, x):
        rnn_out, _ = self.rnn(x.float(), self.initial_state)
        readout = self.fc(rnn_out)
        return readout, rnn_out

    def forward_states(self, x):
        hidden = self.initial_state
        hs = []
        readouts = []
        steps = x.shape[1]
        hidden0 = []
        hidden1 = []
        for step in range(steps):
            temp, hidden = self.rnn(x[:, step:step + 1], hidden)
            if self.kill_neuron is not None:
                if type(self.kill_neuron) == int:
                    hidden[:, :, self.kill_neuron] = 0
                else:
                    for neuron in self.kill_neuron:
                        hidden[:, :, neuron] = 0

            hs.append(hidden.detach())
            hidden0.append(hidden[0:1].detach())
            hidden1.append(hidden[1:].detach())
            readouts.append(self.fc(temp).detach())

        hs = torch.cat(hs).transpose(0, 1)
        readouts = torch.cat(readouts, 1)
        return readouts, hs


class LSTM(RNN):
    def __init__(self, *args, **kwargs):
        super(LSTM, self).__init__(*args, **kwargs)
        if self.initial_state is not None:
            self.initial_state = torch.split(self.initial_state, self.units, dim=-1)

    @property
    def rnn_class(self):
        return nn.LSTM

    def forward_states(self, x):
        # rnn_out, hidden = self.rnn(x.float(), self.initial_state)
        # readout_origin = self.fc(rnn_out)

        hidden = self.initial_state
        hs = []
        cs = []
        readouts = []
        steps = x.shape[1]

        for step in range(steps):
            temp, hidden = self.rnn(x[:, step:step + 1], hidden)
            if self.kill_neuron is not None:
                if type(self.kill_neuron) == int:
                    hidden[1][:, :, self.kill_neuron] = 0
                else:
                    for neuron in self.kill_neuron:
                        hidden[1][:, :, neuron] = 0

            cs.append(hidden[1].detach())
            hs.append(hidden[0].detach())
            readouts.append(self.fc(temp).detach())

        #cs = np.concatenate(cs)
        cs = torch.cat(cs).transpose(0, 1)
        hs = torch.cat(hs).transpose(0, 1)
        readouts = torch.cat(readouts, 1)
        #hidden = np.concatenate((rnn_out.detach().numpy(), cs))
        hidden = torch.cat((hs, cs), dim=-1)

        return readouts, hidden








