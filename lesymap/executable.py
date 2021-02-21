from lesymap_utils import *
from lesymap_experiments import *
import sys



def AUC(model="SVR", n_bs=5, rois=[100, 101], scenario="AND"):
    
    X_culled = pd.read_excel("X_culled.xlsx")
    
    area1 = bootstrap_AUCs(X_culled, model=model, n_bs=n_bs, rois=rois, scenario=scenario)
    print(area1)
    print(np.mean(area1))
    

if __name__ == '__main__':
    AUC(sys.argv[1], int(sys.argv[2]), list(sys.argv[3].split(",")), sys.argv[4])