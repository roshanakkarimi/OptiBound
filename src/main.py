import yaml
import pandas as pd

from AutoGluonAML import AutoGluon
from LHS import LatinHypercubeSampler
from StatUtils import StatUtils

    
def main(main_loaded_data, lhs_loaded_data):

    # Model training inputs
    data_path = main_loaded_data['training_dict']['data_path']
    inp_params = main_loaded_data['training_dict']['inp_params']
    targets = main_loaded_data['training_dict']['targets']
    # n_folds = main_loaded_data['training_dict']['n_folds'] not optimized in trained models
    training_time = main_loaded_data['training_dict']['training_time']
    
    # Initiating AG instance and train best model
    ag_session = AutoGluon(data_path, inp_params, targets)
    ag_session.sobol_sa(lhs_loaded_data['param_ranges'])
    #ag_session.train_model(training_time=training_time)

    # Predict the LHS sample
    ls = LatinHypercubeSampler(lhs_loaded_data)
    ls.generate_dataset()
    ag_session.predict(ls.out_path)

    # outputs plots
    out_csv = pd.read_csv('./data/LHS_out.csv')
    if not out_csv.empty:

        df = pd.DataFrame(out_csv)
        res = ''
        for t in targets:
            print(f'Feature Importance for {t} is being computed...')
            obj = t.replace(' ', '').replace(':','')
            
            res += f'\nFeatures importance results for {t} is:' + str(ag_session.permutation_importance(obj, X=df[inp_params], y=df[t], metric='r2'))
    
            s = StatUtils()
            s.nd_hist_plot(df[t], color="red")

        with open('./logs/feature_importance_out.txt', 'w') as fi_file:
            fi_file.write(str(res))

if __name__ == "__main__":

    with open("./src/main_input.yml", 'r') as main_yml:
        main_loaded_data = yaml.safe_load(main_yml)
    with open("./src/lhs_input.yml", 'r') as lhs_yml:
        lhs_loaded_data = yaml.safe_load(lhs_yml)

    main(main_loaded_data, lhs_loaded_data)
