from data import data_config
from models.model_wrapper import ModelWrapper
from models import pt_models
import torch

interval_discrimination_20 = ModelWrapper(pt_models.VanillaArchitecture(units=20, inputs=2), data_config.interval_discrimination, data_config.interval_discrimination, range(100, 104))
interval_discrimination_30 = ModelWrapper(pt_models.VanillaArchitecture(units=30, inputs=2), data_config.interval_discrimination, data_config.interval_discrimination, range(100, 104))
interval_discrimination_40 = ModelWrapper(pt_models.VanillaArchitecture(units=40, inputs=2), data_config.interval_discrimination, data_config.interval_discrimination, range(100, 104))
interval_discrimination_50 = ModelWrapper(pt_models.VanillaArchitecture(units=50, inputs=2), data_config.interval_discrimination, data_config.interval_discrimination, range(100, 104))

delayed_discrimination_30 = ModelWrapper(pt_models.VanillaArchitecture(units=30, inputs=2), data_config.delayed_discrimination, data_config.delayed_discrimination, range(100, 104))
delayed_discrimination_20 = ModelWrapper(pt_models.VanillaArchitecture(units=20, inputs=2), data_config.delayed_discrimination, data_config.delayed_discrimination, range(100, 104))
delayed_discrimination_40 = ModelWrapper(pt_models.VanillaArchitecture(units=40, inputs=2), data_config.delayed_discrimination, data_config.delayed_discrimination, range(100, 104))
delayed_discrimination_50 = ModelWrapper(pt_models.VanillaArchitecture(units=50, inputs=2), data_config.delayed_discrimination, data_config.delayed_discrimination, range(100, 104))

rsg_20 = ModelWrapper(pt_models.VanillaArchitecture(units=20, inputs=2), data_config.interval_reproduction, data_config.interval_reproduction, range(100, 104))
rsg_30 = ModelWrapper(pt_models.VanillaArchitecture(units=30, inputs=2), data_config.interval_reproduction, data_config.interval_reproduction, range(100, 104))
rsg_40 = ModelWrapper(pt_models.VanillaArchitecture(units=40, inputs=2), data_config.interval_reproduction, data_config.interval_reproduction, range(100, 104))
rsg_50 = ModelWrapper(pt_models.VanillaArchitecture(units=50, inputs=2), data_config.interval_reproduction, data_config.interval_reproduction, range(100, 104))

rsg_20_gru = ModelWrapper(pt_models.GRUArchitecture(units=20, inputs=2), data_config.interval_reproduction, data_config.interval_reproduction, range(100, 104))
rsg_30_gru = ModelWrapper(pt_models.GRUArchitecture(units=30, inputs=2), data_config.interval_reproduction, data_config.interval_reproduction, range(100, 104))
rsg_40_gru = ModelWrapper(pt_models.GRUArchitecture(units=40, inputs=2), data_config.interval_reproduction, data_config.interval_reproduction, range(100, 104))
rsg_50_gru = ModelWrapper(pt_models.GRUArchitecture(units=50, inputs=2), data_config.interval_reproduction, data_config.interval_reproduction, range(100, 104))

rsg_20_lstm = ModelWrapper(pt_models.LSTMArchitecture(units=20, inputs=2), data_config.interval_reproduction, data_config.interval_reproduction, range(100, 104))
rsg_30_lstm = ModelWrapper(pt_models.LSTMArchitecture(units=30, inputs=2), data_config.interval_reproduction, data_config.interval_reproduction, range(100, 104))
rsg_40_lstm = ModelWrapper(pt_models.LSTMArchitecture(units=40, inputs=2), data_config.interval_reproduction, data_config.interval_reproduction, range(100, 104))
rsg_50_lstm = ModelWrapper(pt_models.LSTMArchitecture(units=50, inputs=2), data_config.interval_reproduction, data_config.interval_reproduction, range(100, 104))





def interval_discrimination_vanilla_models():
    return [interval_discrimination_20, interval_discrimination_30, interval_discrimination_40, interval_discrimination_50]

def delayed_discrimination_vanilla_models():
    return [delayed_discrimination_20, delayed_discrimination_30, delayed_discrimination_40, delayed_discrimination_50]

def interval_reproduction_vanilla_models():
    return [rsg_20, rsg_30, rsg_40, rsg_50]

def interval_reproduction_gru_models():
    return [rsg_20_gru, rsg_30_gru, rsg_40_gru, rsg_50_gru]

def interval_reproduction_lstm_models():
    return [rsg_20_lstm, rsg_30_lstm, rsg_40_lstm, rsg_50_lstm]


rnn_types = ['vanilla', 'gru', 'lstm']

