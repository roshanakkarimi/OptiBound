import numpy as np
import pandas as pd 
from scipy.stats.qmc import LatinHypercube, scale

class LatinHypercubeSampler:
    def __init__(self, inp):
        self.param_ranges = inp['param_ranges']
        self.N = inp['N']
        self.out_path = inp['out_dir']

    def generate_dataset(self):
        n_samples = self.N

        sampler = LatinHypercube(d=len(self.param_ranges))

        lhs_samples = sampler.random(n=n_samples)

        scaled_samples = scale(lhs_samples, l_bounds = [v[0] for v in self.param_ranges.values()],
                                            u_bounds = [v[1] for v in self.param_ranges.values()])

        lhs_df = pd.DataFrame(scaled_samples, columns=self.param_ranges.keys())

        lhs_df.to_csv(self.out_path, index=False)



