import kagglegym
import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge, LinearRegression
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import make_pipeline
from sklearn.metrics import r2_score

np.random.seed(17)


class Experiment:
    """
    Influenced by the1owl's initial_script kernel as well as
    other kaggle kernels/discussions
    """
    def __init__(self, validate=True):
        self.validate = validate
        self.env = kagglegym.make()
        self.observation = self.env.reset()
        self.data = self.observation.train
        del self.observation.train

        self.model1, self.model2 = None, None

        # Outlier removal for polynomial regression
        self.y_max = 0.055
        self.y_min = -0.055
        self.cutoff = (self.data.y > self.y_min) & (self.data.y < self.y_max)

        self.key = ['id', 'timestamp', 'y']
        self.l_features = ['technical_20']
        self.features = [x for x in list(self.data.columns) if x not in self.key]
        self.n_features = [c + '_nan' for c in self.features] + ['n_nan']

        # Data cleaning and Nan count for extra trees
        self.fill_vals = self.data[self.features].median()
        self.data['n_nan'] = self.data.isnull().sum(axis=1)
        for c in self.features:
            self.data[c + '_nan'] = self.data[c].isnull().astype('float32')
        self.data = self.data.fillna(self.fill_vals)

        self.val_data = None

    def train(self):
        self.model1 = ExtraTreesRegressor(n_estimators=120, max_depth=4, n_jobs=-1)
        self.model2 = make_pipeline(PolynomialFeatures(3), Ridge(alpha=0.0001))
        self.model1.fit(self.data[self.features + self.n_features], self.data.y)
        self.model2.fit(self.data[self.l_features][self.cutoff], self.data[self.cutoff].y)

    def val(self, y_true_list, y_pred_list):
        """Validation method

        Due to the significant divergence of R values for different time
        periods, this funcion calculates the R values for different periods
        separately, and is used to analyse model performance for various
        distributions.
        """
        r_score = lambda x: np.sign(x) * np.sqrt(abs(x))

        y_true_total = np.concatenate(y_true_list)
        y_pred_total = np.concatenate(y_pred_list)
        print('total', r_score(r2_score(y_true_total,y_pred_total)))

        step = round(len(y_pred_list)/5) + 1
        r2_list = []
        for i in range(0, len(y_pred_list), step):
            y_true_part = np.concatenate(y_true_list[i:i+step])
            y_pred_part = np.concatenate(y_pred_list[i:i+step])
            r2 = r_score(r2_score(y_true_part, y_pred_part))
            r2_list.append(r2)
            print('part', r2)
        print('sum', sum(r for r in r2_list))

    def predict(self):
        done = False
        info = None
        y_true_list, y_pred_list = [], []
        if self.validate:
            self.val_data = pd.HDFStore("data/train.h5", "r").get("train")[['timestamp', 'y']]
            self.val_data = self.val_data[self.val_data.timestamp >= 906]
        while not done:
            # Data cleaning and Nan count for test data
            test = self.observation.features
            test['n_nan'] = test.isnull().sum(axis=1)
            for c in self.features:
                test[c + '_nan'] = test[c].isnull().astype('float32')
            test = test.fillna(self.fill_vals)

            prediction_1 = self.model1.predict(test[self.features + self.n_features])
            prediction_2 = self.model2.predict(test[self.l_features])

            target = self.observation.target
            target['y'] = prediction_1 * 0.65 + prediction_2 * 0.35

            if self.validate:
                part = self.val_data[self.val_data.timestamp == self.observation.features.timestamp[0]].y
                y_true_list.append(part)
                y_pred_list.append(target['y'])
            self.observation, reward, done, info = self.env.step(target)
        print(info)
        if self.validate:
            self.val(y_true_list, y_pred_list)

exp = Experiment(validate=True)
exp.train()
exp.predict()
