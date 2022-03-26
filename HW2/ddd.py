import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import cm
import matplotlib.mlab as ml
from scipy.interpolate import griddata


def gen_cb(N, a, alpha): 
    """
    N: number of points on the checkerboard
    a: width of the checker board (0<a<1)
    alpha: rotation of the checkerboard in radians 
    """
    d = np.random.rand(N, 2).T  # THIS IS THE LINE OF CODE THAT IS DIFFERENT
    d_transformed = np.array([d[0]*np.cos(alpha)-d[1]*np.sin(alpha), 
                              d[0]*np.sin(alpha)+d[1]*np.cos(alpha)]).T
    s = np.ceil(d_transformed[:,0]/a)+np.floor(d_transformed[:,1]/a)
    lab = 2 - (s%2)
    data = d.T
    return data, lab 

def knn(k, X ,y, point):
    x1, x2 = X[y==1], X[y==2] #x1blue x2red
    n1 = len(x1)
    n2 = len(x2)
    
    dataset = X.tolist()
    for i in range(len(dataset)):
        dataset[i].append(y[i])
    
    n=len(dataset)
    dist_list = [] 
    for i in dataset:
        dist = np.linalg.norm(point-np.array(i[:-1])) #find the euclidean distance
        i.append(dist)
    dataset.sort(key=lambda tup: tup[3]) #sort
    k_nearest = dataset[:k] #take only the first k elements
    largest_k = max(k_nearest, key=lambda x: x[3])
    
    #now find how many neighbor in class blue 1
    k_1 = [x for x in k_nearest if x[2] == 1.0]
    
    #now find how many neighbor in class blue 2
    k_2 = [x for x in k_nearest if x[2] == 2.0]
    
    #calculating the volume
    r = largest_k[-1]
    
    #radius of the circle
    v = np.pi * (r**2)
    
    #p(x|y=blue)
    pb = (len(k_1)/(n1*v))
    
    #p(x|y=red)
    pr = (len(k_2)/(n2*v))
    
    #p(x)
    px= len(k_nearest)/(n*v)
    
    return pr,pb,px

X, y = gen_cb(1000, .25, np.pi / 3)


x1, x2 = X[y==1], X[y==2] #x1blue x2red
#py(blue)
p_blue = len(x1)/len(X) 
#py(red)
p_red = len(x2)/len(X)

X_t, y_t = gen_cb(1000, .25, np.pi / 3)
#testing the model
result = X_t.tolist()
for i in range(len(X_t)):
    px_r,px_b,px = knn(15,X, y, X_t[i,:])
    p_1 = px_b*p_blue/px
    p_2 = px_r*p_red/px
    if p_1 > p_2:
        result[i].append(1.0)
    else:
        result[i].append(2.0)

#convert to np array
r = np.array(result)
x_r = r[:,:2]
y_r = r[:,-1]


def plot_contour(x,y,z,resolution = 50,contour_method='linear'):
    resolution = str(resolution)+'j'
    X,Y = np.mgrid[min(x):max(x):complex(resolution),   min(y):max(y):complex(resolution)]
    points = [[a,b] for a,b in zip(x,y)]
    Z = griddata(points, z, (X, Y), method=contour_method)
    return X,Y,Z

X,Y,Z = plot_contour(r[:,0],r[:,1],r[:,2],resolution = 50,contour_method='linear')

#ploting P(x|y)with pcolor
plt.pcolor(X,Y, Z, cmap = 'jet')  
plt.colorbar() 
plt.xlabel("x1")
plt.ylabel("x2")
plt.show()