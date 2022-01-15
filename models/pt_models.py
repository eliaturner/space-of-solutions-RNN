import os
from abc import abstractmethod

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.utils.data import TensorDataset
from torch.optim import Adam, lr_scheduler, SGD

from models import pt_modules
from models.rnn_models import RNNModel

from tools import training_utils, pytorchtools
from tools.utils import load_pickle, dump_pickle
import matplotlib.pyplot as plt
import shutil
import os
import glob
#from ray import tune
import pandas as pd


def maxnorm_loss(input, target):
    return torch.nn.MSELoss()(input, target)
    return 0.01*(torch.max(torch.abs(input-target)))**2 + torch.nn.MSELoss()(input, target)

def train_epoch(train_loader, train_step):
    losses = []
    for x_batch, y_batch in train_loader:
        x_batch = x_batch.to(pytorchtools.device).float()
        y_batch = y_batch.to(pytorchtools.device).float()
        loss = train_step(x_batch, y_batch)
        losses.append(loss)

    return np.average(losses)


def initialization1(weights, units):
    return nn.Parameter(torch.normal(0, 1 / (units ** 0.5), weights.shape))

def initialization2(weights, units):
    return nn.Parameter(weights/np.sqrt(units))
    return nn.Parameter(torch.FloatTensor(weights.shape).uniform_(-1 / (units ** 1), 1 / (units ** 1)))



class PTModelArchitecture(RNNModel):
    @property
    @abstractmethod
    def rnn_func(self):
        pass

    def create_model(
            self,
            steps,
            initial_state=None,
            return_states=False,
            return_output=True,
            return_sequences=True,
            kill_neuron=None):
        if initial_state is None and self.initial_state is not None:
            initial_state = self.initial_state.to(pytorchtools.device)
        model = self.rnn_func(self.inputs, self.outputs, self.units, return_states, initial_state,
                             kill_neuron, self.bias, self.num_layers)
        model.rnn.weight_hh_l0 = initialization2(model.rnn.weight_hh_l0, self.units)
        model.rnn.weight_ih_l0 = initialization2(model.rnn.weight_ih_l0, self.units)
        model.rnn.bias_hh_l0 = initialization2(model.rnn.bias_hh_l0, self.units)
        model.rnn.bias_ih_l0 = initialization2(model.rnn.bias_ih_l0, self.units)
        model.fc.weight = initialization2(model.fc.weight, self.units)
        model.fc.bias = initialization2(model.fc.bias, self.units)
        return model

    def np_to_torch(self, x, train=False):
        temp = torch.from_numpy(x).float()  # .to(pytorchtools.device)
        if train:
            temp = temp.to(pytorchtools.device)

        return temp

    def torch_to_np(self, x):
        return x.cpu().detach().numpy()

    def get_weights(self):
        weights = torch.load(self.model_dir + '/weights.pt', map_location='cpu')
        weights = {key.replace('rnn.rnn.', 'rnn.'): val for key, val in weights.items()}
        return weights

    def load_weights(self, weights=None):
        if weights is None:
            weights = self.get_weights()

        self.weights = weights

    def initial_weights(self):
        weights = torch.load(self.model_dir + '/initial_weights.pt', map_location='cpu')
        weights = {key.replace('rnn.rnn.', 'rnn.'): val for key, val in weights.items()}

        return weights

    def assign_weights(self, model, weights=None):
        if weights is None:
            # if os.path.isfile(self.model_dir + '/weights.pt'):
            #     weights = torch.load(self.model_dir + '/weights.pt', map_location='cpu')
            #     weights = {key.replace('rnn.rnn.', 'rnn.'): val for key, val in weights.items()}
            # else:
            #     print(self.model_dir)
            #     exit()
                # weights_tf = load_pickle(self.model_dir + '/weights.pkl')
                # weights = convert_tensorflow_to_pytorch_weights(weights_tf)
            weights = self.weights
        model.load_state_dict(weights)

    def run_system_from_input(self, inputs):
        return self.predict(inputs)

    def run_system_from_inits(
            self,
            init_states,
            steps,
            input_value=0
    ):
        initial_states = self.np_to_torch(init_states)
        batch_size = init_states.shape[0]
        if type(input_value) == int:
            x = input_value * np.ones((batch_size, steps, self.inputs))
        else:
            x = input_value

        #            x = np.concatenate([val*np.ones((batch_size, steps, 1)) for val in input_value], axis=-1)
        return self.predict(x, initial_states)

    def predict(self, x, initial_states=None, kill_neuron=None, weights=None):
        # torch.cuda.empty_cache()
        steps = x.shape[1]
        model = self.create_model(steps=steps, return_states=True, initial_state=initial_states,
                                  kill_neuron=kill_neuron).cpu()
        self.assign_weights(model, weights)
        pred = model(self.np_to_torch(x))
        output, state = self.torch_to_np(pred[0]), self.torch_to_np(pred[1])
        predictions = {'output': output.squeeze(),
                       'state': state[:,::self.num_layers]}
        if self.num_layers > 1:
            for i in range(1, self.num_layers):
                predictions[f'state{i}'] = state[:,i::self.num_layers]

        return predictions

    def train(self, x_train, y_train, x_val, y_val, params, weights=None, shuffle=True):
        # x_train, y_train = x_val, y_val
        dump_pickle('data/x_train.pkl', x_train)
        dump_pickle('data/y_train.pkl', y_train)
        epochs, batch_size, _, minimal_loss = params.epochs, params.batch_size, params.losses[0], params.minimal_loss
        batch_size = 32

        x_train = self.np_to_torch(x_train, train=True)
        y_train = self.np_to_torch(y_train, train=True)
        train_loader = DataLoader(dataset=TensorDataset(x_train, y_train), batch_size=batch_size,
                                  shuffle=shuffle)
        trainer = PyTorchTrainer(self.model_dir, train_loader, epochs, minimal_loss)
        shutil.rmtree(self.model_dir)
        os.makedirs(self.model_dir)
        # files = glob.glob(f'{c}/*')
        # for f in files:
        #     os.remove(f)

        print('hi')
        pytorchtools.checkpoint_counter = 0
        # initial_state = torch.ones((1, 2)).to(pytorchtools.device)
        model = self.create_model(x_train.shape[1]).to(pytorchtools.device)
        # while True:
        #     weights = nn.Parameter(torch.normal(0, 2,  (2,2)))
        #     if weights.trace() > 2:
        #         model.rnn.weight_hh_l0 = weights
        #         break
        if weights:
            self.assign_weights(model, weights)

        trainer.train(model)
        return trainer.train_on
    # def train(self, x_train, y_train, x_val, y_val, params, weights=None, shuffle=True):
    #     pytorchtools.checkpoint_counter = 0
    #     epochs, batch_size, loss = params.epochs, params.batch_size, params.losses[0]
    #     criterion = nn.MSELoss()
    #     lrs = [j * (10 ** (-i)) for i in range(4, 6) for j in range(10, 0, -2)]
    #     #lrs = [10 ** (-3)]
    #
    #     x_train = self.np_to_torch(x_train, train=True)
    #     y_train = self.np_to_torch(y_train, train=True)
    #     train_loader = DataLoader(dataset=TensorDataset(x_train, y_train), batch_size=batch_size,
    #                               shuffle=shuffle)
    #
    #     train_on = True
    #     while train_on:
    #         print('hi')
    #         pytorchtools.checkpoint_counter = 0
    #         model = self.create_model(x_train.shape[1]).to(pytorchtools.device)
    #         if weights:
    #             self.assign_weights(model, weights)
    #         torch.save(model.state_dict(), self.model_dir + '/initial_weights.pt')
    #         train_on = False
    #         for lr in lrs:
    #             average_losses = []
    #             average_losses_val = []
    #             print(lr)
    #
    #             early_stopping = pytorchtools.EarlyStopping(path=self.model_dir + '/weights')
    #             optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    #             train_step = training_utils.make_train_step(model, criterion, optimizer)
    #             for epoch in range(1, epochs + 1):
    #                 losses = []
    #                 for x_batch, y_batch in train_loader:
    #                     x_batch = x_batch.to(pytorchtools.device).float()
    #                     y_batch = y_batch.to(pytorchtools.device).float()
    #                     loss = train_step(x_batch, y_batch)
    #                     losses.append(loss)
    #
    #                 curr_loss = np.average(losses)
    #                 average_losses.append(curr_loss)
    #                 # print(x_val.shape)
    #                 # model.eval()
    #                 # pred_val = model(x_val).float()
    #                 # val_loss = float(criterion(y_val.float(), pred_val))
    #                 # average_losses_val.append(val_loss)
    #
    #                 if epoch % 100 == 0:
    #                     print('Epoch: {}/{}.............'.format(epoch, epochs), end=' ')
    #                     print("Loss: {:e}, {:.5f}".format(curr_loss, curr_loss), end=' ')
    #                     print()
    #                     #print("Val Loss: {:.8f}".format(val_loss))
    #
    #                 early_stopping(curr_loss, model)
    #                 if early_stopping.early_stop:
    #                     print("Early stopping, epoch {}, score={:e}".format(epoch, early_stopping.best_score))
    #                     break
    #
    #             if early_stopping.stop_training or lr <= 0.0001 and -early_stopping.best_score >= 1e-2:
    #                 train_on = True
    #                 print(lr, -early_stopping.best_score, 1e-2)
    #                 break

    # def grid_search(self, x_train, y_train, weights=None):
    #     x_train = self.np_to_torch(x_train, train=True)
    #     y_train = self.np_to_torch(y_train, train=True)
    #
    #     def training_function(config):
    #         train_loader = DataLoader(dataset=TensorDataset(x_train, y_train), batch_size=config['batch_size'],
    #                                   shuffle=True)
    #         model = self.create_model(x_train.shape[1]).to(pytorchtools.device)
    #         if weights:
    #             self.assign_weights(model, weights)
    #
    #         optimizer = torch.optim.SGD(model.parameters(), lr=config["lr"], momentum=config['momentum'])
    #         train_step = training_utils.make_train_step(model, maxnorm_loss, optimizer, config['clipping_value'])
    #         for epoch in range(20000):
    #             loss = train_epoch(train_loader, train_step)
    #             tune.report(mean_loss=loss)
    #
    #     analysis = tune.run(
    #         training_function,
    #         config={
    #             "lr": tune.grid_search([0.00005, 0.00001, 0.0001, 0.0005, 0.001, 0.005, 0.01]),
    #             "momentum": tune.uniform(0,1),
    #             "clipping_value": tune.uniform(0.1,10),
    #             "batch_size": tune.choice([64]),
    #         },
    #         num_samples=10,resources_per_trial={"cpu": 0, "gpu": 1})
    #
    #     print("Best config: ", analysis.get_best_config(
    #         metric="mean_loss", mode="min"))
    #     df = analysis.results_df
    #     df.to_csv('analysis.csv')
    #     analysis2 = pd.read_csv('analysis.csv')
    #     exit()


# def train_epoch(train_loader, train_step):
#     losses = []
#     for x_batch, y_batch in train_loader:
#         x_batch = x_batch.to(pytorchtools.device).float()
#         y_batch = y_batch.to(pytorchtools.device).float()
#         loss = train_step(x_batch, y_batch)
#         losses.append(loss)
#
#     return np.average(losses)


class PyTorchTrainer:
    def __init__(self, model_dir, train_loader, epochs, minimal_loss):
        # self.model = model
        self.model_dir = model_dir
        self.train_loader = train_loader

        # self.lrs = [j * (10 ** (-i)) for i in range(5, 6) for j in range(10, 0, -2)]
        self.lrs = [1e-8]
        self.epochs = epochs
        self.minimal_loss = 2e-5#minimal_loss
        self.train_on = True

    def log_loss(self, epoch, curr_loss):
        if epoch % 10 == 0:
            print('Epoch: {}/{}.............'.format(epoch, self.epochs), end=' ')
            print("Loss: {:e}, {:.5f}".format(curr_loss, curr_loss), end=' ')
            print()

    # def train_epoch(self, train_step):
    #     losses = []
    #     for x_batch, y_batch in self.train_loader:git
    #         x_batch = x_batch.to(pytorchtools.device).float()
    #         y_batch = y_batch.to(pytorchtools.device).float()
    #         loss = train_step(x_batch, y_batch)
    #         losses.append(loss)
    #
    #     return np.average(losses)

    def check_epoch(self, early_stopping, epoch, lr):
 #       if epoch > 3000:
  #          print(early_stopping.best_score)
        if (-early_stopping.best_score > 3e-2 and epoch > 2000) or (-early_stopping.best_score > 2e-2 and epoch > 2000):
            print("EPOCH - Early stopping, epoch {}, score={:e}".format(epoch,
                                                                early_stopping.best_score))
            return True
        if early_stopping.early_stop:
            print("Early stopping, epoch {}, score={:e}".format(epoch,
                                                                early_stopping.best_score))
            return True

        if lr < 1e-6:
            print("LR - Early stopping, epoch {}, score={:e}".format(epoch,
                                                                early_stopping.best_score))
            return True

        if -early_stopping.best_score < self.minimal_loss:
            print('Minimal loss achieved! finish training')
            self.train_on = False
            return True

        return False

    def check_training(self, loss, lr):
        if lr <= 0.0001 and loss >= 1e-2:
            print('Train Again')
            print(lr, loss, 1e-2)
            return True

        return not self.train_on

    def grid_search(self, model):
        pass

    def train(self, model):
        torch.save(model.state_dict(), self.model_dir + '/initial_weights.pt')
        cooldown = 20
        #model.fc.weight.requires_grad = False
#        model.fc.bias.requires_grad = False
        #model.rnn.bias_ih_l0.requires_grad = False
        #model.rnn.weight_ih_l0.requires_grad = False
        early_stopping = pytorchtools.EarlyStopping(model, path=self.model_dir)
        epoch = 0
        optimizer = Adam(model.parameters(), lr=0.001)

        train_step = training_utils.make_train_step(model, torch.nn.MSELoss(), optimizer)
        average_losses = []
        early_stopping = pytorchtools.EarlyStopping(model, path=self.model_dir)

        for epoch in range(1, 3000):
            curr_loss = train_epoch(self.train_loader, train_step)
            average_losses.append(curr_loss)
            self.log_loss(epoch, curr_loss)
            early_stopping(curr_loss)
            if self.check_epoch(early_stopping, epoch, optimizer.param_groups[0]['lr']):
                model.load_state_dict(torch.load(self.model_dir + '/weights.pt'))
                self.train_on = False
                if min(average_losses) > 0.0002:
                    print('GGGG');
                    self.train_on = True

                return

        print('STEP1 - reload best', -early_stopping.best_score)
        early_stopping.load_best()
        train_step = training_utils.make_train_step(model, torch.nn.MSELoss(), optimizer)
        optimizer = Adam(model.parameters(), lr=0.0005)
        scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=150, verbose=True, cooldown=cooldown, factor=0.8, threshold=1e-7)
        while True:
        # for epoch in range(1, 2*self.epochs + 1):
        # for epoch in range(1, 5):
            epoch += 1
            curr_loss = train_epoch(self.train_loader, train_step)
            average_losses.append(curr_loss)
            scheduler.step(curr_loss)
            self.log_loss(epoch, curr_loss)
            early_stopping(curr_loss)
            if epoch > 1 and scheduler.cooldown_counter == cooldown:
                print('reload best')
                early_stopping.load_best()
                epoch = 0
            elif epoch > 1000 and min(average_losses[-1000:]) > 3e-2:
                early_stopping.load_best()

            if self.check_epoch(early_stopping, epoch, optimizer.param_groups[0]['lr']):
                    break

        # plt.clf()
        # plt.plot(np.log10(average_losses))
        # plt.savefig(f'{self.model_dir}/training_{lr}.png')
        model.load_state_dict(torch.load(self.model_dir + '/weights.pt'))
        self.train_on = False
        if min(average_losses) > 0.0002:
            print('GGGG');
            self.train_on = True




class VanillaArchitecture(PTModelArchitecture):
    @property
    def rnn_type(self):
        return 'vanilla'

    @property
    def rnn_func(self):
        return pt_modules.Vanilla


class LSTMArchitecture(PTModelArchitecture):
    @property
    def rnn_type(self):
        return 'lstm'

    @property
    def rnn_func(self):
        return pt_modules.LSTM


class GRUArchitecture(PTModelArchitecture):
    @property
    def rnn_type(self):
        return 'gru'

    @property
    def rnn_func(self):
        return pt_modules.GRU
