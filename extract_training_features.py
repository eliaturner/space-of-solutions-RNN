import models.model_config as model_config
from analysis.epoch_neural_features import FeatureExtractor, FeatureExtractorDelayedDiscrimination


def delayed_discrimination_vanilla():
    features = [features for wrapper in model_config.interval_reproduction_vanilla_models() for features in wrapper.get_analysis([FeatureExtractorDelayedDiscrimination])]

def interval_discrimination_vanilla():
    features = [features for wrapper in model_config.interval_reproduction_vanilla_models() for features in wrapper.get_analysis([FeatureExtractor])]

def interval_reproduction_vanilla():
    features = [features for wrapper in model_config.interval_reproduction_vanilla_models() for features in wrapper.get_analysis([FeatureExtractor])]

def interval_reproduction_gru():
    features = [features for wrapper in model_config.interval_reproduction_vanilla_models() for features in wrapper.get_analysis([FeatureExtractor])]

def interval_reproduction_lstm():
    features = [features for wrapper in model_config.interval_reproduction_vanilla_models() for features in wrapper.get_analysis([FeatureExtractor])]

if __name__ == '__main__':
    interval_reproduction_vanilla()


