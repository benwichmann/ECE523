from functions import *

N=1500
x,y = gen_checkerboard(N,.25,3.14159/4)
xtest,yt = gen_checkerboard(N,.25,3.14159/4)
k = int(np.sqrt(N))

#organize training dataset
dataset = pd.DataFrame(x,columns=['x1','x2'])
dataset['y1'] = y

#Create y test output set
ytest = np.zeros(len(yt))
#Go through each test point and run Knn
i=0
for xp in xtest:
    ytest[i],pnan,qnan = algorithm_k_NN(k,xp,dataset) 
    i+=1 

# Using the bounds of the test feature vector,
# generate 50x50 set for each possible point
xmin, xmax = np.min(xtest[:,0]),np.max(xtest[:,0])
ymin, ymax = np.min(xtest[:,1]),np.max(xtest[:,1])
xm, ym = np.mgrid[xmin:xmax:50j, ymin:ymax:50j]

#Generate empty posterior sets same shape as xm,ym (50x50)
pxy1 = np.zeros(xm.shape)
pxy2 = np.zeros(xm.shape)

#Iterate through each point on grid, and collect the posterior for 1 or 2 
for i in range(len(xm[0])):
    for j in range(len(xm[1])):
        ynan, pxy1[i,j],pxy2[i,j] = algorithm_k_NN(k,[xm[i,j],ym[i,j]],dataset)

#Plot
plt.figure()
plt.plot(x[np.where(y==2)[0],0],x[np.where(y==2)[0],1],'s',c = 'r')
plt.plot(x[np.where(y==1)[0],0],x[np.where(y==1)[0],1],'o')
plt.xlabel("X1")
plt.ylabel("X2")
plt.title('Original Training Data')
plt.show()
plt.plot(xtest[np.where(ytest==2)[0],0],xtest[np.where(ytest==2)[0],1],'s',c = 'r')
plt.plot(xtest[np.where(ytest==1)[0],0],xtest[np.where(ytest==1)[0],1],'o')
plt.title('Test Training Data')
plt.xlabel("X1")
plt.ylabel("X2")
plt.show()

#Plot posterior probabilities for each point in grids
plt.figure()
plt.pcolormesh(xm,ym,pxy1,cmap='jet')
plt.colorbar()
plt.title('P(y=1|x) Mesh Data')
plt.xlabel("X1")
plt.ylabel("X2")
plt.show()

plt.figure()
plt.pcolormesh(xm,ym,pxy2,cmap='jet')
plt.colorbar()
plt.title('P(y=2|x) Mesh Data')
plt.xlabel("X1")
plt.ylabel("X2")
plt.show()
