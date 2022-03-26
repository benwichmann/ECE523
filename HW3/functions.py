from constants import *


def plot_svm(x,y,w,c):
    fig, ax = plt.subplots(1, 2)
    ax[0].scatter(x[:,0],x[:,1],c = y, cmap ='Dark2',edgecolors="k")
    
    # fit the model, don't regularize for illustration purposes
    clf = svm.SVC(kernel="linear", C=c)
    clf.fit(x, y)

    ax[1].scatter(x[:,0],x[:,1],c = y, cmap ='Dark2',edgecolors="k")

    # plot the decision function
    ax1 = plt.gca()
    xlim = ax1.get_xlim()
    ylim = ax1.get_ylim()

    # create grid to evaluate model
    xx = np.linspace(xlim[0], xlim[1], 30)
    yy = np.linspace(ylim[0], ylim[1], 30)
    YY, XX = np.meshgrid(yy, xx)
    xy = np.vstack([XX.ravel(), YY.ravel()]).T
    Z = clf.decision_function(xy).reshape(XX.shape)

    # plot decision boundary and margins
    ax1.contour(
        XX, YY, Z, colors="k", levels=[-1, 0, 1], alpha=0.5, linestyles=["--", "-", "--"]
    )
    # plot support vectors
    ax1.scatter(
        clf.support_vectors_[:, 0],
        clf.support_vectors_[:, 1],
        s=100,
        linewidth=1,
        facecolors="none",
        edgecolors="k",
    )


    plt.grid()
    plt.show()
    plt.clf()

def plot_data(x,y,w,c):
    plt.figure()

    clf = svm.SVC(kernel="linear",C=c)
    clf.fit(x,y)
    a = -w[0] / w[1]
    xx = np.linspace(-5, 5)
    yy = a * xx - (clf.intercept_[0]) / w[1]
    margin = 1 / np.sqrt(np.sum(clf.coef_ ** 2))
    yy_down = yy - np.sqrt(1 + a ** 2) * margin
    yy_up = yy + np.sqrt(1 + a ** 2) * margin
    plt.clf()
    plt.plot(xx, yy, "k-")
    plt.plot(xx, yy_down, "k--")
    plt.plot(xx, yy_up, "k--")
    plt.scatter(
        clf.support_vectors_[:, 0],
        clf.support_vectors_[:, 1],
        s=80,
        facecolors="none",
        #zorder=10,
        edgecolors="k",
        cmap=plt.cm.get_cmap("Dark2"),
    )
    plt.scatter(x[:,0],x[:,1],c = y, cmap ='Dark2',edgecolors="k")
    
    plt.axis("tight")
    x_min = -3
    x_max = 3
    y_min = -4.5
    y_max = 4.5
    YY, XX = np.meshgrid(yy, xx)
    xy = np.vstack([XX.ravel(), YY.ravel()]).T
    Z = clf.decision_function(xy).reshape(XX.shape)

    # Put the result into a contour plot
    plt.contourf(XX, YY, Z, cmap=plt.cm.get_cmap("RdBu"), alpha=0.5, linestyles=["-"])

    plt.xlim(x_min, x_max)
    plt.ylim(y_min, y_max)

    plt.xticks(())
    plt.yticks(())
    return plt.figure
    #plt.show()

def split_dataset_np(source_dataset, target_dateset):
    """
    Input: two datasets
    Output: 4 dfs, x-,y-source, and x-,y-target
    """
    ysrc = source_dataset.iloc[:,-1].to_numpy()
    xsrc = source_dataset.iloc[:, 0:-1].to_numpy()
    
    ytar = target_dateset.iloc[:,-1].to_numpy()
    xtar = target_dateset.iloc[:, 0:-1].to_numpy()

    return xsrc,ysrc,xtar,ytar


def split_dataset_df(source_dataset, target_dateset):
    """
    Input: two datasets
    Output: 4 dfs, x-,y-source, and x-,y-target
    """
    ysrc = source_dataset.iloc[:,-1]
    xsrc = source_dataset.iloc[:, 0:-1]
    
    ytar = target_dateset.iloc[:,-1]
    xtar = target_dateset.iloc[:, 0:-1]

    return xsrc,ysrc,xtar,ytar

def kernal_mul(vi,vj):
    vdot = np.matmul(vi,vj)
    return vdot

def kernal_phi(x):
    kx = np.array(np.dot(x,x.T))
    return kx

def qp_svm(x,y,C_reg,B_reg,ws):
    '''
    '''
    n = x.shape[0]

    #P = yi*yj*k(x,x) 
    P = np.outer(y,y) * kernal_phi(x)
    P = cx.matrix(P, tc="d")

    #q term -1 + ByiwsTx
    f1 = -1*np.ones(n)
    if B_reg > 0:
        for i in range(n):
            f1[i] = -1+B_reg*y[i]*np.dot(ws,x[i]) 
    q = cx.matrix(f1,tc="d")

    #C equaility term (Ax = b)
    A = cx.matrix(y, (1, n), tc="d")
    b = cx.matrix(0.0,tc="d")

    #G inequality term (Gx <= h)
    G = cx.matrix(np.vstack((np.diag(-1 * np.ones(n)),np.diag(np.ones(n)))),tc="d")
    h = cx.matrix(np.hstack((np.zeros(n),C_reg*np.ones(n))),tc="d")

    #alpha values
    sol = cx.solvers.qp(P,q, G, h, A, b)
    #alpha = np.ravel(sol["x"])
    alpha = sol["x"].T

    w = np.array([0.,0.])
    if B_reg > 0:
        w[0]= B_reg*ws[0] + np.sum(alpha*y*x[:,0])
        w[1]=B_reg*ws[1] + np.sum(alpha*y*x[:,1])

    else:
        w[0]=np.sum(alpha*y*x[:,0])
        w[1]=np.sum(alpha*y*x[:,1])

    return w
