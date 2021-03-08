from lesymap_utils import *
from lesymap_experiments import *
import sys


'''Exemple de syntaxe : !python executable3.py 100,101,102 5    '''


def AUC(rois, SNR):
    
    X_culled = pd.read_excel("X_culled.xlsx")
    models = ["DR", "SVR", "DLASSO", "RF", "RF+SHAP"]
    scenarii = ["single", "AND", "OR", "SUM"]
    n_bs = 100

    print("rois : ", rois)
    print("SNR : ", SNR)
    
    for model in models:
        for scenario in scenarii:
            print("Méthode : ", model)
            print("Scenario : ", scenario)
            area = bootstrap_AUCs(X_culled, model=model, SNR=SNR, n_bs=n_bs,
                               rois=rois, scenario=scenario)
            print("AUC : ", np.mean(area))
  



if __name__ == '__main__':
    
    AUC(list(map(int, sys.argv[1].split(","))), int(sys.argv[2]))