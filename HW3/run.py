from pandas import DataFrame
from functions import *

source_path = "c:/datasets/source_train.csv"
target_path = "c:/datasets/target_train.csv"
source_ds = pd.read_csv(source_path, header=None)  
target_ds = pd.read_csv(target_path, header=None)  

xsrc1,ysrc1,xtar,ytar = split_dataset_np(source_ds,target_ds)

# B=0, ws = np.sum(alph * y[i] * x[i])
ws = qp_svm(xsrc1,ysrc1,1,0,0)

c = 100
# B=1, wt = B*wt + np.sum(alph * y[i] * x[i])
wt = qp_svm(xtar,ytar,c,.5,ws)

plot_svm(xtar,ytar,wt,c)

