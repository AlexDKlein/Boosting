import numpy as np
from bootstrap import Bootstrapper
from tree import Tree

class RandomForest(Bootstrapper):
    def __init__(self, n_estimators=500, metric='entropy', max_depth=3, min_leaf_size=1, classifier=True, n_jobs=1):
        super().__init__(Tree, n_estimators, metric=metric, max_depth=max_depth, 
                        min_leaf_size=min_leaf_size, classifier=classifier, n_jobs=n_jobs)
    