import numpy as np
import seaborn as sns

from analysis.analyzer import Analyzer
from tools.math_utils import calc_angle, calc_normalized_q_value, curvature_2d, get_length, svm_if_separable
from scipy.stats import pearsonr
from sklearn.linear_model import LinearRegression

import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
import matplotlib.cm as cm
sns.set()

class FeatureExtractor(Analyzer):
    name = 'feature_extractor_interval'
    value_descriptor = 'none'

    def get_first_pc(self):
        all_states = self.states.reshape(-1, self.states.shape[-1])
        pca = PCA(1)
        pca.fit_transform(all_states)
        return pca.explained_variance_ratio_[0]

    def get_curvature(self, points):
        points = PCA(2).fit_transform(points)
        curvature = curvature_2d(points)

        log_curvature = np.log10(curvature)
        log_curvature = log_curvature[~np.isnan(log_curvature)]
        return np.min(log_curvature), np.max(log_curvature)

    def get_aspect_ratio(self, epoch2_time, epoch2_ts):
        len_time = get_length(epoch2_time)
        len_space = get_length(epoch2_ts)
        return len_space/len_time

    def get_speed(self, points):
        return np.average(calc_normalized_q_value(points))

    def get_correlation(self, vec1, vec2):
        return pearsonr(vec1, vec2)[0]

    def get_angle(self, vec1, vec2):
        return calc_angle(vec1, vec2)

    def get_svm_margin(self, vec1, vec2):
        if_separable = svm_if_separable(vec1, vec2)
        return if_separable

    def get_lengths(self, points_list):
        x = np.linspace(0, 1, len(points_list))
        distances = [get_length(vec) for vec in points_list]
        distances = np.array(distances)/np.min(distances) - 1
        reg = LinearRegression()
        reg.fit(x.reshape(-1, 1), distances)
        return reg.coef_[0]


    def run(self):
        epoch1 = None
        epoch2 = [None]
        epoch1, epoch2 = self.data_params.epoch1_epoch2(self.outputs, self.states)
        features = {}
        features['curvature_min'], features['curvature_max'] = self.get_curvature(epoch1)
        features['svm'] = self.get_svm_margin(epoch1, np.vstack(epoch2))
        features['speed_ratio'] = self.get_speed(epoch1)/self.get_speed(epoch2[-1])
        features['epoch1_speed_ratio'] = self.get_speed(epoch1[-5:])/self.get_speed(epoch1[:5])

        epoch1_vec = epoch1[-1] - epoch1[0]
        epoch2_vec = epoch2[-1][-1] - epoch2[-1][0]
        features['angle'] = self.get_angle(epoch1_vec, epoch2_vec)
        features['corr'] = self.get_correlation(epoch1_vec, epoch2_vec)

        features['first_pc'] = self.get_first_pc()

        epoch2_time = epoch2[-1]
        epoch2_space = np.vstack([s[-1] for s in epoch2])
        features['aspect_ratio'] = self.get_aspect_ratio(epoch2_time, epoch2_space)
        features['speed_vs_length'] = self.get_lengths(epoch2)
        features['speed_end'] = self.get_speed(self.states[:1,-3:])

        features_vec = np.array([features[k] for k in sorted(features.keys())])
        self.save_file(features_vec, 'features_vec')
        return features_vec

        pass

class FeatureExtractorDelayedDiscrimination(Analyzer):
    name = 'feature_extractor_delayed_discrimination'
    value_descriptor = 'none'

    def get_first_pc(self):
        all_states = self.states.reshape(-1, self.states.shape[-1])
        pca = PCA(1)
        pca.fit_transform(all_states)
        return pca.explained_variance_ratio_[0]

    def get_curvature(self, points):
        points = PCA(2).fit_transform(points)
        curvature = curvature_2d(points)

        log_curvature = np.log10(curvature)
        log_curvature = log_curvature[~np.isnan(log_curvature)]
        return np.min(log_curvature), np.max(log_curvature)

    def get_aspect_ratio(self, epoch2_time, epoch2_ts):
        len_time = get_length(epoch2_time)
        len_space = get_length(epoch2_ts)
        return len_space/len_time

    def get_speed(self, points):
        return np.average(calc_normalized_q_value(points))

    def get_correlation(self, vec1, vec2):
        return pearsonr(vec1, vec2)[0]

    def get_angle(self, vec1, vec2):
        return calc_angle(vec1, vec2)
        pass

    def get_svm_margin(self, vec1, vec2):
        if_separable = svm_if_separable(vec1, vec2)
        print(if_separable)
        return if_separable

    def get_lengths(self, points_list):
        x = np.linspace(0, 1, len(points_list))
        distances = [get_length(vec) for vec in points_list]
        distances = np.array(distances)/np.min(distances) - 1
        reg = LinearRegression()
        reg.fit(x.reshape(-1, 1), distances)
        return reg.coef_[0]


    def run(self):
        epochs2, epochs3 = self.data_params.epoch1_epoch2(self.outputs, self.states)
        features = {}
        features['curvature_min'], features['curvature_max'] = self.get_curvature(epochs2[5])
        features['svm'] = self.get_svm_margin(epochs2[-1], epochs2[0])
        features['epoch_speed_ratio'] = self.get_speed(epochs2[-1][-5:])/self.get_speed(epochs2[-1][:5])
        features['curvature_min_m'], features['curvature_max_m'] = self.get_curvature(epochs3[0])
        features['curvature_min_p'], features['curvature_max_p'] = self.get_curvature(epochs3[-1])
        features['svm'] = self.get_svm_margin( np.vstack(epochs3), np.vstack(epochs2))
        features['speed_ratio'] = np.mean([self.get_speed(epochs3[0]), self.get_speed(epochs3[1])])/self.get_speed(epochs2[-1])

        features['first_pc'] = self.get_first_pc()

        epoch2_time = epochs2[-1]
        epoch2_space = np.vstack([s[-1] for s in epochs2])
        features['aspect_ratio'] = self.get_aspect_ratio(epoch2_time, epoch2_space)
        features['speed_vs_length'] = self.get_lengths(epochs2)
        features['speed_end'] = self.get_speed(self.states[:1,-3:])

        features_vec = np.array([features[k] for k in sorted(features.keys())])
        self.save_file(features_vec, 'features_vec')
        return features_vec