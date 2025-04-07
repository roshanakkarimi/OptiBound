import os 
import pandas as pd
import numpy as np
from SALib.analyze import sobol

from sklearn.metrics import mean_squared_error, r2_score, root_mean_squared_error

from autogluon.tabular import TabularPredictor
from autogluon.common.utils.utils import setup_outputdir
from autogluon.core.metrics import make_scorer


class AutoGluon:
    def __init__(self, data_path:str, parameters: list, targets: list):
        self.train_dataset = pd.read_csv(data_path) 
        self.inp_parameters = parameters
        self.targets = targets
        self.bm_path = setup_outputdir('./bestmodel/', warn_if_exist=False) # main path of best model(s) 
        self.post_tr = [s.replace(' ', '').replace(':','') for s in self.targets] # bestmodel final path for target(s)
    
    def sobol_sa(self, ranges: dict):
        #Sobol sensitivity analysis
        df = self.train_dataset.dropna()
        X = df[self.inp_parameters]
        Y = df['c0: Heating [kWh]']

        problem = {
        'num_vars': X.shape[1],
        'names': df.columns[:19].tolist(),
        'bounds': ranges.values()
        }
        Si = sobol.analyze(problem, Y)

        print("First-order indices:")
        for name, val in zip(problem['names'], Si['S1']):
            print(f"{name}: {val:.4f}")

        print("\nTotal-order indices:")
        for name, val in zip(problem['names'], Si['ST']):
            print(f"{name}: {val:.4f}")


    def cvrmse_score(self, y_true, y_pred) -> float:
        """
        The Coefficient of variation of the Root Mean Squared Error 
        """        
        rmse = np.sqrt(mean_squared_error(y_true, y_pred))
        mean_y = np.mean(y_true)
        return (rmse / mean_y) * 100 
    
    def load_data_XY(self) -> pd.DataFrame:
        """
        Loading train dataset parameters
        """
        df = self.train_dataset.dropna()
        X = df[self.inp_parameters]
        Y = df[self.targets]
        
        return X, Y

    def train_model(self, k:int = None, training_time:int = None, save_log:bool = False):
        """
        Train best model(s) using AutoGluon TabularPredictor and save them in the 'bestmodel' folder.

        Args:
            k (int): wanted number of folds
            training_time (int): time limit for trainig
            save_log (boolean): save log to file
        
        """
        X, y = self.load_data_XY()

        coefficient_variation_rmse_scorer = make_scorer(name='cvrmse', 
                                                        score_func=self.cvrmse_score,
                                                        optimum=1,
                                                        greater_is_better=False, needs_proba=False)
        
        for i in range(len(self.targets)):
            label = self.targets[i]
            path_i = os.path.join(self.bm_path, "Predictor_" + self.post_tr[i])

            train_data = pd.concat(([X,y[label]]), axis=1)

            TabularPredictor(label=label,eval_metric = coefficient_variation_rmse_scorer, #'r2', 'root_mean_squared_error'
                             path=path_i, log_to_file = save_log).fit(train_data, 
                                                                      fit_weighted_ensemble = False, 
                                                                      fit_full_last_level_weighted_ensemble = False,
                                                                      presets='high_quality',
                                                                      time_limit=training_time, 
                                                                      num_bag_folds=k)

    def predict(self, data_path:str):
        """
        Predict the output(s) of the given data_path and save them as a single dataset in data/LHS_out.csv.
        """
        predict_data = pd.read_csv(data_path)
        X = predict_data[self.inp_parameters]

        output = pd.DataFrame()

        for i in range(len(self.targets)):
            label = self.targets[i]
            path_i = os.path.join(self.bm_path, "Predictor_" + self.post_tr[i])
            predictor = TabularPredictor.load(path_i)
            ev = predictor.predict(X)
            output[label] = ev
        
        pd.concat([X,output], axis=1).to_csv('data/LHS_out.csv')
    
    def permutation_importance(self, target, X, y, metric='cvrmse', n_repeats=5, random_state=None):
        
        np.random.seed(random_state)
        model = TabularPredictor.load(os.path.join(self.bm_path, 'Predictor_' + target))
        baseline_score = self._score_model(model, X, y, metric)
        importance = {col: [] for col in X.columns}

        for col in importance.keys():
            for _ in range(n_repeats):
                X_shuffled = X.copy()
                X_shuffled.loc[:, col] = np.random.permutation(X_shuffled.loc[:, col])
                score = self._score_model(model, X_shuffled, y, metric)
                importance[col].append(baseline_score - score)

        # Calculate mean and std
        result = {
            'importance_mean of features': [f'Feature_{col}: {np.mean(importance[col])}' for col in importance],
            'importance_std of features': [f'Feature_{col}: {np.std(importance[col])}' for col in importance],
        }
        return result
    
    def _score_model(self, model, X, y, metric):
        """Helper to compute model score."""
        pred = model.predict(X)
        if metric == 'r2':
            return r2_score(y, pred)
        elif metric == 'rmse':
            return root_mean_squared_error(y, pred)
        elif metric == 'cvrmse':
            return self.cvrmse_score(y, pred)
        else:
            raise ValueError("Unsupported metric. Use 'rmse', 'r2' or 'cvrmse'.")

# To be removed 
class AutoGluonPrediction:
    def __init__(self, input_params, input_values:np.ndarray):
        self.row = pd.DataFrame([input_values], columns=input_params) 
        self.bm_path = setup_outputdir('./bestmodel/', warn_if_exist=False) # main path of best model(s) 

    def predict_obj(self, objective) -> float:
        path_i = os.path.join(self.bm_path, "Predictor_" + objective)
        predictor = TabularPredictor.load(path_i)
        ev = predictor.predict(self.row)[0] # change ds or function
        return ev