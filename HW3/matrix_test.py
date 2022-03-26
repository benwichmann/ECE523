from functions import *

a = np.array([[1,2,3],[1,1,3]])
b = np.array([[1,1,3],[1,2,3]])
c = np.array([1,2,3])
cT = np.array([1,2])


da = pd.DataFrame(data=a)
db = pd.DataFrame(data=b)
dc = pd.DataFrame(data=c)

print(da.T)
print(kernel_phi(dc,da.T))
print(kernal_mul(c,a.T))