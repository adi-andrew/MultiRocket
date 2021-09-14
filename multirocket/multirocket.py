# Chang Wei Tan, Angus Dempster, Christoph Bergmeir, Geoffrey I Webb
#
# MultiRocket: Effective summary statistics for convolutional outputs in time series classification
# https://arxiv.org/abs/2102.00457

import time

import numpy as np
from sklearn.linear_model import Ridge
from sklearn.linear_model import RidgeClassifier
from sklearn.linear_model import ElasticNet
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import SGDClassifier
from sklearn.preprocessing import StandardScaler
# from sklearn.model_selection import TimeSeriesSplit

from multirocket import feature_names, get_feature_set
# from multirocket import minirocket as minirocket
from multirocket import minirocket_multivariate as minirocket  # use multivariate version.
from multirocket import rocket as rocket


class MultiRocket:

    def __init__(self,
                 num_features=10000,
                 feature_id=22,
                 kernel_selection=0,
                 foldIds=None,
                 output_model='RidgeClassifier',
                 alpha=1e-1,
                 l1_ratio=0.5,
                 max_iter=5,
                 normalize=True,
                 class_weight=None):
        """
        MultiRocket
        :param num_features: number of features
        :param feature_id: feature id to identify the feature combinations
        :param kernel_selection: 0 = minirocket kernels, 1 = rocket kernels
        """
        if kernel_selection == 0:
            self.name = "MiniRocket_{}_{}".format(feature_id, num_features)
        else:
            self.name = "Rocket_{}_{}".format(feature_id, num_features)

        self.kernels = None
        self.kernel_selection = kernel_selection
        self.num_features = num_features
        self.feature_id = feature_id

        # get parameters based on feature id
        fixed_features, optional_features, num_random_features = get_feature_set(feature_id)
        self.fixed_features = fixed_features
        self.optional_features = optional_features
        self.num_random_features = num_random_features
        self.n_features_per_kernel = len(fixed_features) + num_random_features
        self.num_kernels = int(num_features / self.n_features_per_kernel)

        fixed_features_list = [feature_names[x] for x in self.fixed_features]
        optional_features_list = [feature_names[x] for x in self.optional_features]
        self.feature_list = fixed_features_list + optional_features_list
        print('FeatureID: {} -- features for each kernel: {}'.format(self.feature_id, self.feature_list))

        print('Creating {} with {} kernels'.format(self.name, self.num_kernels))
        # print('Using time series split')
        # cv = TimeSeriesSplit(5)
        # self.classifier = RidgeClassifierCV(alphas=np.logspace(-3, 3, 10),
        #                                     normalize=True,
        #                                     cv=cv)
        self.foldIds = foldIds

        self.output_model = output_model
        if output_model == 'RidgeClassifier':
            self.classifier = RidgeClassifier(alpha=alpha,
                                            normalize=True)
        elif output_model == 'LogisticRegression':
            self.classifier = LogisticRegression(C=alpha,
                                                 multi_class='multinomial',
                                                 class_weight=class_weight,
                                                 tol=1e-2,
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
        self.apply_kernel_on_train_duration = 0
        self.apply_kernel_on_test_duration = 0

    def fit_kernels(self, x_train):
        start_time = time.perf_counter()

        print('[{}] Training with training set of {}'.format(self.name, x_train.shape))
        if self.kernel_selection == 0:
            # swap the axes for minirocket kernels. will standardise the axes in future.
            x_train = x_train.swapaxes(1, 2)

            _start_time = time.perf_counter()
            self.kernels = minirocket.fit(x_train,
                                          self.feature_id,
                                          self.n_features_per_kernel,
                                          num_features=self.num_kernels)
            self.generate_kernel_duration = time.perf_counter() - _start_time

            x_train_transform = minirocket.transform(x_train, self.kernels)
            self.apply_kernel_on_train_duration = time.perf_counter() - _start_time - self.generate_kernel_duration
        else:
            _start_time = time.perf_counter()
            self.kernels = rocket.generate_kernels(x_train.shape[1],
                                                   num_kernels=self.num_kernels,
                                                   feature_id=self.feature_id,
                                                   num_features=self.n_features_per_kernel,
                                                   num_channels=x_train.shape[2])
            self.generate_kernel_duration = time.perf_counter() - _start_time

            x_train_transform = rocket.apply_kernels(x_train, self.kernels, self.feature_id)
            self.apply_kernel_on_train_duration = time.perf_counter() - start_time - self.generate_kernel_duration

        x_train_transform = np.nan_to_num(x_train_transform)

        elapsed_time = time.perf_counter() - start_time
        print('Kernels applied!, took {}s'.format(elapsed_time))
        print('Transformed Shape {}'.format(x_train_transform.shape))
        return x_train_transform


    def fit_all_kernels(self, x_train):
        N = len(self.foldIds)
        x_train_transform_folds=[]
        x_test_transform_folds=[]
        for i, fold in enumerate(self.foldIds):
            print(f'Fitting kernel to fold {i+1}/{N}')
            x_train_transform = self.fit_kernels(x_train[fold[0]])
            x_train_transform = self.scaler.fit_transform(x_train_transform)
            x_train_transform_folds.append(x_train_transform)

            self._start_time = time.perf_counter()
            x_test_transform = self.transform_x(x_train[fold[1]])
            x_test_transform = self.scaler.transform(x_test_transform)
            x_test_transform_folds.append(x_test_transform)
        return x_train_transform_folds, x_test_transform_folds


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
            print(f'Traning fold {i+1}/{N}')
            yhat.append(self.train_fold_on_kernel(x=x_train[i],
                                             y=y_train[fold[0]],
                                             x_test=x_test[i],
                                             yhat=yhat))

        self.train_duration = time.perf_counter() - self._start_time

        print('Training done!, took {:.3f}s'.format(self.train_duration))

        return yhat


    def fit_cv(self, x_train, y_train):

        print('Training')
        _start_time = time.perf_counter()

        yhat=[]
        N = len(self.foldIds)
        for i, fold in enumerate(self.foldIds):
            print(f'Traning fold {i+1}/{N}')
            yhat = self.train_fold(x=x_train[fold[0]],
                                   y=y_train[fold[0]],
                                   x_test=x_train[fold[1]],
                                   yhat=yhat)

        self.train_duration = time.perf_counter() - _start_time

        print('Training done!, took {:.3f}s'.format(self.train_duration))

        return yhat


    def train_fold(self, x, y, x_test=None, yhat=None):
        x_train_transform = self.fit_kernels(x)
        if self.output_model == 'RidgeClassifier':
            self.classifier.fit(x_train_transform, y)
            if x_test is None:
                yhat = self.classifier._predict_proba_lr(x_train_transform)
            else:
                yhat.append(self.predict_proba(x_test))
        elif self.output_model in ['LogisticRegression', 'SGDClassifier']:
            x_train_transform = self.scaler.fit_transform(x_train_transform)
            self.classifier.fit(x_train_transform, y)
            if x_test is None:
                yhat = self.classifier.predict_proba(x_train_transform)
            else:
                yhat.append(self.predict_proba(x_test))
        elif self.output_model in ['RidgeRegression', 'ElasticNet']:
            self.regressor.fit(x_train_transform, y)
            if x_test is None:
                yhat = self.regressor.predict(x_train_transform)
            else:
                yhat.append(self.predict_linear(x_test))
        return yhat


    def fit(self, x_train, y_train):
        print('Training')
        self._start_time = time.perf_counter()
        yhat=self.train_fold(x_train, y_train)
        self.train_duration = time.perf_counter() - self._start_time
        print('Training done!, took {:.3f}s'.format(self.train_duration))
        return yhat


    def predict(self, x):
        print('[{}] Predicting'.format(self.name))
        start_time = time.perf_counter()

        if self.kernel_selection == 0:
            # swap the axes for minirocket kernels. will standardise the axes in future.
            x = x.swapaxes(1, 2)

            x_test_transform = minirocket.transform(x, self.kernels)
        else:
            x_test_transform = rocket.apply_kernels(x, self.kernels, self.feature_id)

        self.apply_kernel_on_test_duration = time.perf_counter() - start_time
        x_test_transform = np.nan_to_num(x_test_transform)
        print('Kernels applied!, took {:.3f}s. Transformed shape: {}. '.format(self.apply_kernel_on_test_duration,
                                                                               x_test_transform.shape))

        yhat = self.classifier.predict(x_test_transform)
        self.test_duration = time.perf_counter() - start_time

        print("[{}] Predicting completed, took {:.3f}s".format(self.name, self.test_duration))

        return yhat


    def transform_x(self, x):
        if self.kernel_selection == 0:
            # swap the axes for minirocket kernels. will standardise the axes in future.
            x = x.swapaxes(1, 2)
            x_test_transform = minirocket.transform(x, self.kernels)
        else:
            x_test_transform = rocket.apply_kernels(x, self.kernels, self.feature_id)
        self.apply_kernel_on_test_duration = time.perf_counter() - self._start_time
        x_test_transform = np.nan_to_num(x_test_transform)
        print('Kernels applied!, took {:.3f}s. Transformed shape: {}. '.format(self.apply_kernel_on_test_duration,
                                                                               x_test_transform.shape))
        return x_test_transform


    def predict_proba(self, x):
        print('[{}] Predicting'.format(self.name))
        self._start_time = time.perf_counter()
        x_test_transform = self.transform_x(x)

        if self.output_model in ['LogisticRegression', 'SGDClassifier']:
            x_test_transform = self.scaler.transform(x_test_transform)
            yhat = self.classifier.predict_proba(x_test_transform)
        else:
            yhat = self.classifier._predict_proba_lr(x_test_transform)
        self.test_duration = time.perf_counter() - self._start_time

        print("[{}] Predicting completed, took {:.3f}s".format(self.name, self.test_duration))

        return yhat


    def predict_linear(self, x):
        print('[{}] Predicting'.format(self.name))
        start_time = time.perf_counter()

        if self.kernel_selection == 0:
            # swap the axes for minirocket kernels. will standardise the axes in future.
            x = x.swapaxes(1, 2)

            x_test_transform = minirocket.transform(x, self.kernels)
        else:
            x_test_transform = rocket.apply_kernels(x, self.kernels, self.feature_id)

        self.apply_kernel_on_test_duration = time.perf_counter() - start_time
        x_test_transform = np.nan_to_num(x_test_transform)
        print('Kernels applied!, took {:.3f}s. Transformed shape: {}. '.format(self.apply_kernel_on_test_duration,
                                                                               x_test_transform.shape))

        yhat = self.regressor.predict(x_test_transform)
        self.test_duration = time.perf_counter() - start_time

        print("[{}] Predicting completed, took {:.3f}s".format(self.name, self.test_duration))

        return yhat
