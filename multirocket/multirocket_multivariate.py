# Chang Wei Tan, Angus Dempster, Christoph Bergmeir, Geoffrey I Webb
#
# MultiRocket: Multiple pooling operators and transformations for fast and effective time series classification
# https://arxiv.org/abs/2102.00457

import time

import numpy as np
from numba import njit, prange
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler


@njit("float32[:](float32[:,:,:],int32[:],int32[:],int32[:],int32[:],float32[:])",
      fastmath=True, parallel=False, cache=True)
def _fit_biases(X, num_channels_per_combination, channel_indices, dilations, num_features_per_dilation, quantiles):
    num_examples, num_channels, input_length = X.shape

    # equivalent to:
    # >>> from itertools import combinations
    # >>> indices = np.array([_ for _ in combinations(np.arange(9), 3)], dtype = np.int32)
    indices = np.array((
        0, 1, 2, 0, 1, 3, 0, 1, 4, 0, 1, 5, 0, 1, 6, 0, 1, 7, 0, 1, 8,
        0, 2, 3, 0, 2, 4, 0, 2, 5, 0, 2, 6, 0, 2, 7, 0, 2, 8, 0, 3, 4,
        0, 3, 5, 0, 3, 6, 0, 3, 7, 0, 3, 8, 0, 4, 5, 0, 4, 6, 0, 4, 7,
        0, 4, 8, 0, 5, 6, 0, 5, 7, 0, 5, 8, 0, 6, 7, 0, 6, 8, 0, 7, 8,
        1, 2, 3, 1, 2, 4, 1, 2, 5, 1, 2, 6, 1, 2, 7, 1, 2, 8, 1, 3, 4,
        1, 3, 5, 1, 3, 6, 1, 3, 7, 1, 3, 8, 1, 4, 5, 1, 4, 6, 1, 4, 7,
        1, 4, 8, 1, 5, 6, 1, 5, 7, 1, 5, 8, 1, 6, 7, 1, 6, 8, 1, 7, 8,
        2, 3, 4, 2, 3, 5, 2, 3, 6, 2, 3, 7, 2, 3, 8, 2, 4, 5, 2, 4, 6,
        2, 4, 7, 2, 4, 8, 2, 5, 6, 2, 5, 7, 2, 5, 8, 2, 6, 7, 2, 6, 8,
        2, 7, 8, 3, 4, 5, 3, 4, 6, 3, 4, 7, 3, 4, 8, 3, 5, 6, 3, 5, 7,
        3, 5, 8, 3, 6, 7, 3, 6, 8, 3, 7, 8, 4, 5, 6, 4, 5, 7, 4, 5, 8,
        4, 6, 7, 4, 6, 8, 4, 7, 8, 5, 6, 7, 5, 6, 8, 5, 7, 8, 6, 7, 8
    ), dtype=np.int32).reshape(84, 3)

    num_kernels = len(indices)
    num_dilations = len(dilations)

    num_features = num_kernels * np.sum(num_features_per_dilation)

    biases = np.zeros(num_features, dtype=np.float32)

    feature_index_start = 0

    combination_index = 0
    num_channels_start = 0

    for dilation_index in range(num_dilations):

        dilation = dilations[dilation_index]
        padding = ((9 - 1) * dilation) // 2

        num_features_this_dilation = num_features_per_dilation[dilation_index]

        for kernel_index in range(num_kernels):

            feature_index_end = feature_index_start + num_features_this_dilation

            num_channels_this_combination = num_channels_per_combination[combination_index]

            num_channels_end = num_channels_start + num_channels_this_combination

            channels_this_combination = channel_indices[num_channels_start:num_channels_end]

            _X = X[np.random.randint(num_examples)][channels_this_combination]

            A = -_X  # A = alpha * X = -X
            G = _X + _X + _X  # G = gamma * X = 3X

            C_alpha = np.zeros((num_channels_this_combination, input_length), dtype=np.float32)
            C_alpha[:] = A

            C_gamma = np.zeros((9, num_channels_this_combination, input_length), dtype=np.float32)
            C_gamma[9 // 2] = G

            start = dilation
            end = input_length - padding

            for gamma_index in range(9 // 2):
                C_alpha[:, -end:] = C_alpha[:, -end:] + A[:, :end]
                C_gamma[gamma_index, :, -end:] = G[:, :end]

                end += dilation

            for gamma_index in range(9 // 2 + 1, 9):
                C_alpha[:, :-start] = C_alpha[:, :-start] + A[:, start:]
                C_gamma[gamma_index, :, :-start] = G[:, start:]

                start += dilation

            index_0, index_1, index_2 = indices[kernel_index]

            C = C_alpha + C_gamma[index_0] + C_gamma[index_1] + C_gamma[index_2]
            C = np.sum(C, axis=0)

            biases[feature_index_start:feature_index_end] = np.quantile(C, quantiles[
                                                                           feature_index_start:feature_index_end])

            feature_index_start = feature_index_end

            combination_index += 1
            num_channels_start = num_channels_end

    return biases


def _fit_dilations(input_length, num_features, max_dilations_per_kernel):
    num_kernels = 84

    num_features_per_kernel = num_features // num_kernels
    true_max_dilations_per_kernel = min(num_features_per_kernel, max_dilations_per_kernel)
    multiplier = num_features_per_kernel / true_max_dilations_per_kernel

    max_exponent = np.log2((input_length - 1) / (9 - 1))
    dilations, num_features_per_dilation = \
        np.unique(np.logspace(0, max_exponent, true_max_dilations_per_kernel, base=2).astype(np.int32),
                  return_counts=True)
    num_features_per_dilation = (num_features_per_dilation * multiplier).astype(np.int32)  # this is a vector

    remainder = num_features_per_kernel - np.sum(num_features_per_dilation)
    i = 0
    while remainder > 0:
        num_features_per_dilation[i] += 1
        remainder -= 1
        i = (i + 1) % len(num_features_per_dilation)

    return dilations, num_features_per_dilation


# low-discrepancy sequence to assign quantiles to kernel/dilation combinations
def _quantiles(n):
    return np.array([(_ * ((np.sqrt(5) + 1) / 2)) % 1 for _ in range(1, n + 1)], dtype=np.float32)


def fit(X, num_features=10_000, max_dilations_per_kernel=32):
    _, num_channels, input_length = X.shape

    num_kernels = 84

    dilations, num_features_per_dilation = _fit_dilations(input_length, num_features, max_dilations_per_kernel)

    num_features_per_kernel = np.sum(num_features_per_dilation)

    quantiles = _quantiles(num_kernels * num_features_per_kernel)

    num_dilations = len(dilations)
    num_combinations = num_kernels * num_dilations

    max_num_channels = min(num_channels, 9)
    max_exponent = np.log2(max_num_channels + 1)

    num_channels_per_combination = (2 ** np.random.uniform(0, max_exponent, num_combinations)).astype(np.int32)

    channel_indices = np.zeros(num_channels_per_combination.sum(), dtype=np.int32)

    num_channels_start = 0
    for combination_index in range(num_combinations):
        num_channels_this_combination = num_channels_per_combination[combination_index]
        num_channels_end = num_channels_start + num_channels_this_combination
        channel_indices[num_channels_start:num_channels_end] = np.random.choice(num_channels,
                                                                                num_channels_this_combination,
                                                                                replace=False)

        num_channels_start = num_channels_end

    biases = _fit_biases(X, num_channels_per_combination, channel_indices,
                         dilations, num_features_per_dilation, quantiles)

    return num_channels_per_combination, channel_indices, dilations, num_features_per_dilation, biases


@njit("float32[:,:](float32[:,:,:],float32[:,:,:],Tuple((int32[:],int32[:],int32[:],int32[:],float32[:])),Tuple((int32[:],int32[:],int32[:],int32[:],float32[:])),int32)",
      fastmath=True, parallel=True, cache=True)
def transform(X, X1, parameters, parameters1, n_features_per_kernel=4):
    num_examples, num_channels, input_length = X.shape

    num_channels_per_combination, channel_indices, dilations, num_features_per_dilation, biases = parameters
    _, _, dilations1, num_features_per_dilation1, biases1 = parameters1

    # equivalent to:
    # >>> from itertools import combinations
    # >>> indices = np.array([_ for _ in combinations(np.arange(9), 3)], dtype = np.int32)
    indices = np.array((
        0, 1, 2, 0, 1, 3, 0, 1, 4, 0, 1, 5, 0, 1, 6, 0, 1, 7, 0, 1, 8,
        0, 2, 3, 0, 2, 4, 0, 2, 5, 0, 2, 6, 0, 2, 7, 0, 2, 8, 0, 3, 4,
        0, 3, 5, 0, 3, 6, 0, 3, 7, 0, 3, 8, 0, 4, 5, 0, 4, 6, 0, 4, 7,
        0, 4, 8, 0, 5, 6, 0, 5, 7, 0, 5, 8, 0, 6, 7, 0, 6, 8, 0, 7, 8,
        1, 2, 3, 1, 2, 4, 1, 2, 5, 1, 2, 6, 1, 2, 7, 1, 2, 8, 1, 3, 4,
        1, 3, 5, 1, 3, 6, 1, 3, 7, 1, 3, 8, 1, 4, 5, 1, 4, 6, 1, 4, 7,
        1, 4, 8, 1, 5, 6, 1, 5, 7, 1, 5, 8, 1, 6, 7, 1, 6, 8, 1, 7, 8,
        2, 3, 4, 2, 3, 5, 2, 3, 6, 2, 3, 7, 2, 3, 8, 2, 4, 5, 2, 4, 6,
        2, 4, 7, 2, 4, 8, 2, 5, 6, 2, 5, 7, 2, 5, 8, 2, 6, 7, 2, 6, 8,
        2, 7, 8, 3, 4, 5, 3, 4, 6, 3, 4, 7, 3, 4, 8, 3, 5, 6, 3, 5, 7,
        3, 5, 8, 3, 6, 7, 3, 6, 8, 3, 7, 8, 4, 5, 6, 4, 5, 7, 4, 5, 8,
        4, 6, 7, 4, 6, 8, 4, 7, 8, 5, 6, 7, 5, 6, 8, 5, 7, 8, 6, 7, 8
    ), dtype=np.int32).reshape(84, 3)

    num_kernels = len(indices)
    num_dilations = len(dilations)
    num_dilations1 = len(dilations1)

    num_features = num_kernels * np.sum(num_features_per_dilation)
    num_features1 = num_kernels * np.sum(num_features_per_dilation1)

    features = np.zeros((num_examples, (num_features + num_features1) * n_features_per_kernel), dtype=np.float32)
    n_features_per_transform = np.int64(features.shape[1] / 2)

    for example_index in prange(num_examples):

        _X = X[example_index]

        A = -_X  # A = alpha * X = -X
        G = _X + _X + _X  # G = gamma * X = 3X

        # Base series
        feature_index_start = 0

        combination_index = 0
        num_channels_start = 0

        for dilation_index in range(num_dilations):

            _padding0 = dilation_index % 2

            dilation = dilations[dilation_index]
            padding = ((9 - 1) * dilation) // 2

            num_features_this_dilation = num_features_per_dilation[dilation_index]

            C_alpha = np.zeros((num_channels, input_length), dtype=np.float32)
            C_alpha[:] = A

            C_gamma = np.zeros((9, num_channels, input_length), dtype=np.float32)
            C_gamma[9 // 2] = G

            start = dilation
            end = input_length - padding

            for gamma_index in range(9 // 2):
                C_alpha[:, -end:] = C_alpha[:, -end:] + A[:, :end]
                C_gamma[gamma_index, :, -end:] = G[:, :end]

                end += dilation

            for gamma_index in range(9 // 2 + 1, 9):
                C_alpha[:, :-start] = C_alpha[:, :-start] + A[:, start:]
                C_gamma[gamma_index, :, :-start] = G[:, start:]

                start += dilation

            for kernel_index in range(num_kernels):

                feature_index_end = feature_index_start + num_features_this_dilation

                num_channels_this_combination = num_channels_per_combination[combination_index]

                num_channels_end = num_channels_start + num_channels_this_combination

                channels_this_combination = channel_indices[num_channels_start:num_channels_end]

                _padding1 = (_padding0 + kernel_index) % 2

                index_0, index_1, index_2 = indices[kernel_index]

                C = C_alpha[channels_this_combination] + \
                    C_gamma[index_0][channels_this_combination] + \
                    C_gamma[index_1][channels_this_combination] + \
                    C_gamma[index_2][channels_this_combination]
                C = np.sum(C, axis=0)

                if _padding1 == 0:
                    for feature_count in range(num_features_this_dilation):
                        feature_index = feature_index_start + feature_count
                        _bias = biases[feature_index]

                        ppv = 0
                        last_val = 0
                        max_stretch = 0.0
                        mean_index = 0
                        mean = 0

                        for j in range(C.shape[0]):
                            if C[j] > _bias:
                                ppv += 1
                                mean_index += j
                                mean += C[j] + _bias
                            elif C[j] < _bias:
                                stretch = j - last_val
                                if stretch > max_stretch:
                                    max_stretch = stretch
                                last_val = j
                        stretch = C.shape[0] - 1 - last_val
                        if stretch > max_stretch:
                            max_stretch = stretch

                        end = feature_index
                        features[example_index, end] = ppv / C.shape[0]
                        end = end + num_features
                        features[example_index, end] = max_stretch
                        end = end + num_features
                        features[example_index, end] = mean / ppv if ppv > 0 else 0
                        end = end + num_features
                        features[example_index, end] = mean_index / ppv if ppv > 0 else -1
                else:
                    _c = C[padding:-padding]

                    for feature_count in range(num_features_this_dilation):
                        feature_index = feature_index_start + feature_count
                        _bias = biases[feature_index]

                        ppv = 0
                        last_val = 0
                        max_stretch = 0.0
                        mean_index = 0
                        mean = 0

                        for j in range(_c.shape[0]):
                            if _c[j] > _bias:
                                ppv += 1
                                mean_index += j
                                mean += _c[j] + _bias
                            elif _c[j] < _bias:
                                stretch = j - last_val
                                if stretch > max_stretch:
                                    max_stretch = stretch
                                last_val = j
                        stretch = _c.shape[0] - 1 - last_val
                        if stretch > max_stretch:
                            max_stretch = stretch

                        end = feature_index
                        features[example_index, end] = ppv / _c.shape[0]
                        end = end + num_features
                        features[example_index, end] = max_stretch
                        end = end + num_features
                        features[example_index, end] = mean / ppv if ppv > 0 else 0
                        end = end + num_features
                        features[example_index, end] = mean_index / ppv if ppv > 0 else -1

                feature_index_start = feature_index_end

                combination_index += 1
                num_channels_start = num_channels_end

        # First order difference
        _X1 = X1[example_index]
        A1 = -_X1  # A = alpha * X = -X
        G1 = _X1 + _X1 + _X1  # G = gamma * X = 3X

        feature_index_start = 0

        combination_index = 0
        num_channels_start = 0

        for dilation_index in range(num_dilations1):

            _padding0 = dilation_index % 2

            dilation = dilations1[dilation_index]
            padding = ((9 - 1) * dilation) // 2

            num_features_this_dilation = num_features_per_dilation1[dilation_index]

            C_alpha = np.zeros((num_channels, input_length - 1), dtype=np.float32)
            C_alpha[:] = A1

            C_gamma = np.zeros((9, num_channels, input_length - 1), dtype=np.float32)
            C_gamma[9 // 2] = G1

            start = dilation
            end = input_length - padding

            for gamma_index in range(9 // 2):
                C_alpha[:, -end:] = C_alpha[:, -end:] + A1[:, :end]
                C_gamma[gamma_index, :, -end:] = G1[:, :end]

                end += dilation

            for gamma_index in range(9 // 2 + 1, 9):
                C_alpha[:, :-start] = C_alpha[:, :-start] + A1[:, start:]
                C_gamma[gamma_index, :, :-start] = G1[:, start:]

                start += dilation

            for kernel_index in range(num_kernels):

                feature_index_end = feature_index_start + num_features_this_dilation

                num_channels_this_combination = num_channels_per_combination[combination_index]

                num_channels_end = num_channels_start + num_channels_this_combination

                channels_this_combination = channel_indices[num_channels_start:num_channels_end]

                _padding1 = (_padding0 + kernel_index) % 2

                index_0, index_1, index_2 = indices[kernel_index]

                C = C_alpha[channels_this_combination] + \
                    C_gamma[index_0][channels_this_combination] + \
                    C_gamma[index_1][channels_this_combination] + \
                    C_gamma[index_2][channels_this_combination]
                C = np.sum(C, axis=0)

                if _padding1 == 0:
                    for feature_count in range(num_features_this_dilation):
                        feature_index = feature_index_start + feature_count
                        _bias = biases1[feature_index]

                        ppv = 0
                        last_val = 0
                        max_stretch = 0.0
                        mean_index = 0
                        mean = 0

                        for j in range(C.shape[0]):
                            if C[j] > _bias:
                                ppv += 1
                                mean_index += j
                                mean += C[j] + _bias
                            elif C[j] < _bias:
                                stretch = j - last_val
                                if stretch > max_stretch:
                                    max_stretch = stretch
                                last_val = j
                        stretch = C.shape[0] - 1 - last_val
                        if stretch > max_stretch:
                            max_stretch = stretch

                        end = feature_index + n_features_per_transform
                        features[example_index, end] = ppv / C.shape[0]
                        end = end + num_features
                        features[example_index, end] = max_stretch
                        end = end + num_features
                        features[example_index, end] = mean / ppv if ppv > 0 else 0
                        end = end + num_features
                        features[example_index, end] = mean_index / ppv if ppv > 0 else -1
                else:
                    _c = C[padding:-padding]

                    for feature_count in range(num_features_this_dilation):
                        feature_index = feature_index_start + feature_count
                        _bias = biases1[feature_index]

                        ppv = 0
                        last_val = 0
                        max_stretch = 0.0
                        mean_index = 0
                        mean = 0

                        for j in range(_c.shape[0]):
                            if _c[j] > _bias:
                                ppv += 1
                                mean_index += j
                                mean += _c[j] + _bias
                            elif _c[j] < _bias:
                                stretch = j - last_val
                                if stretch > max_stretch:
                                    max_stretch = stretch
                                last_val = j
                        stretch = _c.shape[0] - 1 - last_val
                        if stretch > max_stretch:
                            max_stretch = stretch

                        end = feature_index + n_features_per_transform
                        features[example_index, end] = ppv / _c.shape[0]
                        end = end + num_features
                        features[example_index, end] = max_stretch
                        end = end + num_features
                        features[example_index, end] = mean / ppv if ppv > 0 else 0
                        end = end + num_features
                        features[example_index, end] = mean_index / ppv if ppv > 0 else -1

                feature_index_start = feature_index_end

    return features


class MultiRocket:

    def __init__(
            self,
            num_features=10000,
            verbose=0,
            foldIds=None,
            output_model='LogisticRegression',
            max_iter=5,
            class_weight=None,
            alpha=1e-1,
            normalize=True
    ):
        self.name = "MultiRocket"

        self.base_parameters = None
        self.diff1_parameters = None

        self.n_features_per_kernel = 4
        self.num_features = num_features / 2  # 1 per transformation
        self.num_kernels = int(self.num_features / self.n_features_per_kernel)

        if verbose > 1:
            print('[{}] Creating {} with {} kernels'.format(self.name, self.name, self.num_kernels))

        self.foldIds = foldIds

        self.output_model = output_model
        if output_model == 'RidgeClassifier':
            self.classifier = RidgeClassifier(alpha=alpha,
                                            normalize=True)
        elif output_model == 'LogisticRegression':
            self.classifier = LogisticRegression(C=alpha,
                                                 multi_class='multinomial',
                                                 class_weight=class_weight,
                                                 tol=1e-1,
                                                 max_iter=max_iter,
                                                 solver='lbfgs')
        elif output_model == 'SGDClassifier':
            self.classifier = SGDClassifier(loss='log',
                                            alpha=alpha,
                                            penalty='l2',
                                            class_weight='balanced',
                                            tol=1e-3,
                                            max_iter=max_iter,
                                            n_jobs=10)
        elif output_model == 'RidgeRegression':
            self.regressor = Ridge(alpha=alpha,
                                   normalize=True)
        elif output_model == 'ElasticNet':
            self.regressor = ElasticNet(alpha=alpha,
                                        l1_ratio=l1_ratio,
                                        normalize=True)
        else:
            print(f'Unknown output_model {output_model}')
            exit

        if normalize:
            self.scaler = StandardScaler(copy=False)

        self.train_duration = 0
        self.test_duration = 0
        self.generate_kernel_duration = 0
        self.train_transforms_duration = 0
        self.test_transforms_duration = 0
        self.apply_kernel_on_train_duration = 0
        self.apply_kernel_on_test_duration = 0

        self.verbose = verbose


    def fit_all_kernels(self, x_train):
        N = len(self.foldIds)
        x_train_transform_folds=[]
        x_test_transform_folds=[]
        for i, fold in enumerate(self.foldIds):
            self._start_time = time.perf_counter()
            print(f'Fitting kernel to fold {i+1}/{N}')
            x_train_transform = self.fit_kernels_transform_x(x_train[fold[0]])
            x_train_transform_folds.append(x_train_transform)

            x_test_transform = self.transform_x(x_train[fold[1]])
            x_test_transform_folds.append(x_test_transform)
        return x_train_transform_folds, x_test_transform_folds


    def transform_x(self, x, xx=None, scale=True):
        if xx is None:
            xx = np.diff(x, 1)
        x_transform = transform(x, xx,
                                self.base_parameters,
                                self.diff1_parameters,
                                self.n_features_per_kernel)
        if scale:
            x_transform = self.scaler.transform(x_transform)
        return np.nan_to_num(x_transform)


    def fit_kernels_transform_x(self, x_train):
        start_time = time.perf_counter()
        if self.verbose > 1:
            print('[{}] Training with training set of {}'.format(self.name, x_train.shape))

        _start_time = time.perf_counter()
        xx = np.diff(x_train, 1)
        self.train_transforms_duration += time.perf_counter() - _start_time

        _start_time = time.perf_counter()
        self.base_parameters = fit(
            x_train,
            num_features=self.num_kernels
        )
        self.diff1_parameters = fit(
            xx,
            num_features=self.num_kernels
        )
        self.generate_kernel_duration += time.perf_counter() - _start_time

        _start_time = time.perf_counter()
        x_train_transform = self.transform_x(x_train, xx, scale=False)
        x_train_transform = self.scaler.fit_transform(x_train_transform)
        self.apply_kernel_on_train_duration += time.perf_counter() - _start_time

        elapsed_time = time.perf_counter() - start_time
        if self.verbose > 1:
            print('[{}] Kernels applied!, took {}s'.format(self.name, elapsed_time))
            print('[{}] Transformed Shape {}'.format(self.name, x_train_transform.shape))

        return x_train_transform


    def train_fold_on_kernel(self, x, y, x_test=None, yhat=None):
        if self.output_model in ['LogisticRegression', 'SGDClassifier']:
            self.classifier.fit(x, y)
            yhat = self.classifier.predict_proba(x_test)
        elif self.output_model in ['RidgeRegression', 'ElasticNet']:
            self.regressor.fit(x, y)
            yhat = self.regressor.predict(x_test)
        return yhat


    def fit_cv_to_kernels(self, x_train, x_test, y_train):

        print('Training')
        self._start_time = time.perf_counter()

        yhat=[]
        N = len(self.foldIds)
        for i, fold in enumerate(self.foldIds):
            if self.output_model == 'LogisticRegression':
                if i == 0:
                    self.classifier.warm_start = False
                else:
                    self.classifier.warm_start = True
            print(f'Traning fold {i+1}/{N}')
            yhat.append(self.train_fold_on_kernel(x=x_train[i],
                                             y=y_train[fold[0]],
                                             x_test=x_test[i],
                                             yhat=yhat))

        self.train_duration = time.perf_counter() - self._start_time

        print('Training done!, took {:.3f}s'.format(self.train_duration))

        return yhat
        

    def predict_proba(self, x):
        print('[{}] Predicting'.format(self.name))
        self._start_time = time.perf_counter()
        x_test_transform = self.transform_x(x, scale=True)

        if self.output_model in ['LogisticRegression', 'SGDClassifier']:
            yhat = self.classifier.predict_proba(x_test_transform)
        else:
            yhat = self.classifier._predict_proba_lr(x_test_transform)
        self.test_duration = time.perf_counter() - self._start_time

        print("[{}] Predicting completed, took {:.3f}s".format(self.name, self.test_duration))

        return yhat


    def fit(self, x_train, y_train, predict_on_train=True):
        self.generate_kernel_duration = 0
        self.apply_kernel_on_train_duration = 0
        self.train_transforms_duration = 0

        start_time = time.perf_counter()

        _start_time = time.perf_counter()

        if self.verbose > 1:
            print('[{}] Training with training set of {}'.format(self.name, x_train.shape))

        x_train_transform = self.fit_kernels_transform_x(x_train)
        elapsed_time = time.perf_counter() - start_time
        if self.verbose > 1:
            print('[{}] Kernels applied!, took {}s'.format(self.name, elapsed_time))
            print('[{}] Transformed Shape {}'.format(self.name, x_train_transform.shape))

        _start_time = time.perf_counter()
        if self.verbose > 1:
            print('[{}] Training'.format(self.name))
        if self.output_model in ['LogisticRegression', 'SGDClassifier']:
            self.classifier.fit(x_train_transform, y_train)
            if predict_on_train:
                yHat = self.classifier.predict_proba(x_train_transform)
            else:
                yHat = []
        self.train_duration = time.perf_counter() - _start_time
        if self.verbose > 1:
            print('[{}] Training done!, took {:.3f}s'.format(self.name, self.train_duration))
        return yHat


    def predict(self, x):
        if self.verbose > 1:
            print('[{}] Predicting'.format(self.name))

        self.apply_kernel_on_test_duration = 0
        self.test_transforms_duration = 0

        _start_time = time.perf_counter()
        xx = np.diff(x, 1)
        self.test_transforms_duration += time.perf_counter() - _start_time

        _start_time = time.perf_counter()
        x_transform = transform(
            x, xx,
            self.base_parameters, self.diff1_parameters,
            self.n_features_per_kernel
        )
        self.apply_kernel_on_test_duration += time.perf_counter() - _start_time

        x_transform = np.nan_to_num(x_transform)
        if self.verbose > 1:
            print('Kernels applied!, took {:.3f}s. Transformed shape: {}.'.format(self.apply_kernel_on_test_duration,
                                                                                  x_transform.shape))

        start_time = time.perf_counter()
        yhat = self.classifier.predict(x_transform)
        self.test_duration = time.perf_counter() - start_time
        if self.verbose > 1:
            print("[{}] Predicting completed, took {:.3f}s".format(self.name, self.test_duration))

        return yhat
