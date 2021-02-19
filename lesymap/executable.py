%load_ext autoreload
%autoreload 2
from lesymap_utils import *
from lesymap_experiments import *


X = build_lesion_dataset()

X_culled = cull_dataset(X, threshold=True)

area1 = bootstrap_AUCs(X_culled, method='RF', 1, n_bs=5, scenario='AND', n_jobs=30)
print(area1)
print(np.mean(area1))