import os
import pygmo as pg
import pandas as pd
from matplotlib import pyplot as plt
from matplotlib import cm
import numpy as np

#from concurrent.futures import ProcessPoolExecutor
#from multiprocessing import Pool, Lock
#from AutoGluonAML import AutoGluonPrediction

from autogluon.tabular import TabularPredictor
from autogluon.common.utils.utils import setup_outputdir
        
class MyProblem:
    def __init__(self):
        self.dimension = 19  # Number of decision variables
        self.nobj = 3        # Number of objectives
        self.objs = ['c0Heating[kWh]', 'c2Kitchen_thermaldiscomfort', 'c16co2_kitchen']
        self.bm_path = setup_outputdir('./bestmodel/', warn_if_exist=False) 
        self.predictors = {} 
        self.best_models = {}
        for objective in self.objs:  
            path = os.path.join(self.bm_path, f"Predictor_{objective}")
            self.predictors[objective] = TabularPredictor.load(path)
            self.best_models[objective] = self.predictors[objective].model_best
            
    def fitness(self, x):
        
        inp_params = ['@@window@@', '@@floor_insulation@@','@@roof_insulation@@', '@@orientation@@',
                    '@@lighting_density@@', '@@occupancy_density@@', '@@equipment_density@@', '@@HSPT@@',
                    '@@hvac_efficiency@@' , '@@flowINF@@', '@@wall_insulation@@', '@@met_rate@@',
                    '@@clo@@' ,'@@shgc@@', '@@wwr@@', '@@overhang@@', '@@OpenTime@@', '@@CloseTime@@', '@@WindowOpen@@']
        
        row = pd.DataFrame([x], columns=inp_params) 

        return [self.predictors[objective].predict(row, model=self.best_models[objective])[0] 
                for objective in self.predictors]

        # To be fixed return

        # f = lambda row, objective: self.predictors[objective].predict(row, model='WeightedEnsemble_L2')[0] 

        # with ProcessPoolExecutor() as p:
        #     res = list(p.map(f, row, self.objs))
        # print(res)
        # return list(res)

        
    # For faster fitness computation (to be fixed)

    # def batch_fitness(self, dvs):
    #     print(dvs)
    #     with ProcessPoolExecutor(max_workers=4) as executor:
    #         results = executor.map(self.fitness, dvs)
    #     return np.vstack(results)
    

    def get_bounds(self):
        return ([0.53, 0.06, 0.06, 0, 1, 0.125, 16, 18, 0, 0.006, 0.12, 60, 0, 0.1, 40, 0.5, 10, 10, 0.10],
                [3.21, 0.88, 0.88, 360, 6, 0.3289, 24, 22, 1, 0.02, 0.85, 360, 1, 1, 90, 2, 180, 180, 1])

    def get_nobj(self):
        return self.nobj  

    
if __name__ == '__main__':

    # Create the problem
    import datetime
    p_start = datetime.datetime.now()

    prob = pg.problem(MyProblem())
    uda = pg.nsga2(gen=12) #20
    # Create the NSGA-II algorithm
    algo = pg.algorithm(uda) 
    # Create a population     
    pop = pg.population(prob, size=20, seed=42) #120

    # Evolve the population
    pop = algo.evolve(pop)

    p_finish = datetime.datetime.now()
    print(p_finish - p_start)

    # Get the non-dominated solutions (Pareto front)
    non_dominated = pg.fast_non_dominated_sorting(pop.get_f())
    
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    f = pop.get_f()
    ax.scatter(f[:,0], f[:,1], f[:,2], c='gray', alpha=0.3)
    ax.set_xlabel('Heating [kWh]')
    ax.set_ylabel('Thermaldiscomfort')
    ax.set_zlabel('co2_kitchen')
    # pg.plot_non_dominated_fronts(pop.get_f() ,axes=ax)
    # plt.show()
    surf = ax.plot_surface(pd.DataFrame(f[:,0]), 
                           pd.DataFrame(f[:,1]), 
                           pd.DataFrame(f[:,2]), 
                           rstride=1, cstride=1, cmap=cm.coolwarm,
                           linewidth=0, antialiased=False)
    fig.colorbar(surf, shrink=0.5, aspect=10)
    plt.show()

    # Save results csv
    best_res = pd.DataFrame()
    best_res = pd.concat([pd.DataFrame(pop.get_x()), pd.DataFrame(pop.get_f())], axis=1)
    best_res.columns = ['@@window@@', '@@floor_insulation@@','@@roof_insulation@@', '@@orientation@@',
                        '@@lighting_density@@', '@@occupancy_density@@', '@@equipment_density@@', '@@HSPT@@',
                        '@@hvac_efficiency@@' , '@@flowINF@@', '@@wall_insulation@@', '@@met_rate@@',
                        '@@clo@@' ,'@@shgc@@', '@@wwr@@', '@@overhang@@', '@@OpenTime@@', '@@CloseTime@@', '@@WindowOpen@@', 
                        'c0Heating[kWh]', 'c2Kitchen_thermaldiscomfort', 'c16co2_kitchen']
    
    best_res.to_csv('data/pfs.csv')
