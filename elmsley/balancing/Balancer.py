from imblearn.under_sampling import RandomUnderSampler
from imblearn.over_sampling import SMOTE, ADASYN

class Balancer:
    def __init__(self, oversampling_method=None, undersample=False):
        self.oversampling_method = oversampling_method
        self.undersample = undersample


    def resample(self, x_train, y_train):
        if self.oversampling_method is not None:
            if self.oversampling_method == 'ADASYN':
                print('Oversampling with ADASYN algorithm started...')
                sampler = ADASYN(random_state=42, sampling_strategy='not majority')
            elif self.oversampling_method == 'SMOTE':
                print('Oversampling with SMOTE algorithm...')
                sampler = SMOTE(random_state=42, sampling_strategy='not majority')
            else:
                print('Oversampling method unspecified or not implemented')
                return x_train, y_train
        elif self.undersample:
            print('Random Undersampling algorithm...')
            sampler = RandomUnderSampler(random_state=42)
        else:
            return x_train, y_train  # No resampling

        return sampler.fit_resample(x_train, y_train)