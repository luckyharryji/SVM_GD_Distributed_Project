
from pyspark import SparkContext
import numpy as np
from scipy.optimize import fmin_l_bfgs_b
from operator import add
import pylab as plb

def pos(X):
    return np.maximum(X,np.zeros_like(X))

def parseVector(line):
    return np.array([float(x) for x in line.split(' ')])

def merge(a,b):
    return np.vstack((a,b))
    
def combineLocally(iterator):
    combiners = {}
    for (k, v) in iterator:
        print "label goes here: ", k, v
        if k not in combiners:
            combiners[k] = v
        else:
            combiners[k] = merge(combiners[k], v)
    return combiners.iteritems()

def get_stat(iterator): 
    yield sum(1 for _ in iterator)

#def solveLocally(iterator,x_var,rho,z,u):
def func(x_var,A,u,z,rho):
    return np.sum(pos(np.dot(A,np.atleast_2d(x_var).T) + 1)) +  (rho/float(2))*np.sum(np.power(x_var -z+u,2))  

def objective(partition_ind,iterable_data,stat,lambda_, x, z):
    val = 0
    num = sum(stat[:partition_ind])
    for arr in iterable_data:
        val+= np.sum(pos(np.dot(arr[1],x[:,num]) + 1))
        num+=1
    yield val + 1/(2*lambda_)*np.sum(z**2)

def hinge_loss(A,x):
    val = 0;
    for i in range(len(A)):
        val+= np.sum(pos(np.dot(A[i],x[:,i]) + 1))
    return val

    
def solveLocally(partition_ind,iterable_data,x_var,stat,u,z,rho):
    num = sum(stat[:partition_ind])
    #res = list()
    for arr in iterable_data:
        x_min  = fmin_l_bfgs_b(func, x_var, None, \
                               (arr[1],u[:,num],z[:,0],rho),\
                               approx_grad=1,bounds=None, m=10, factr=1e7, pgtol=1e-5,\
                               epsilon=1e-8,iprint=0, maxfun=15000, maxiter=1000,\
                               disp=0, callback=None)
        num+=1
        yield x_min[0]              

#A = [ -((ones(n,1)*y).*x)' -y'];
def transformData(arr):
    '''
      Return a tuple (class,(class*dataPoints,-class))
      arr: array-like,shape = (nFeatures,)
          
    '''
    nFeatures = np.size(arr) - 1 
    return (arr[-1],np.hstack((-arr[-1]*arr[0:nFeatures], -arr[-1])))

def plot2D(data,theta):
    '''
        theta: array-like, theta=(w,b)
    '''
    X = data[:, 0:2]
    y = data[:, 2]
    
 
    pos = np.where(y == 1)
    neg = np.where(y == -1)
    plb.scatter(X[pos, 0], X[pos, 1], marker='o', c='b')
    plb.scatter(X[neg, 0], X[neg, 1], marker='x', c='r')
    plb.xlabel('x1')
    plb.ylabel('x2')
    plb.legend(['1', '-1'])
    plotDesisionLine(data,theta[0:2],theta[2])
    plb.show()
def plotDesisionLine(data,w,b):
    x1 = np.linspace(min(data[:,0]), max(data[:,0]), 100)
    x2 = -(w[0]*x1 + b)/w[1]
    plb.plot(x1,x2)

    
if __name__ == '__main__':
    sc = SparkContext('local[3]', "Xiangyu and Nianzu: distributed SVM")
    nSlices  = 5
    line_data = sc.textFile('data.txt',nSlices)

    data = line_data.map(parseVector).map(transformData).mapPartitions(combineLocally,1).cache()

    N = data.count() #Number of the elements
    stat = data.mapPartitions(get_stat).collect()
    #------------------------------------------------------------------------------ 
    #Initialize constants and variables 
    MAX_ITER = 1000
    ABSTOL   = 1e-4
    RELTOL   = 1e-2


    nFeatures = np.size(data.first()[1],1) - 1 #number of features
    u = np.zeros((nFeatures+1,N))
    z = np.zeros((nFeatures+1,1))
    lambda_ = 1
    rho = 1
    alpha = 1.4
    x_var = np.random.rand(nFeatures+1,1)
    x = np.zeros((nFeatures+1,N))

    objval = list()
    r_norm = list()
    s_norm = list()
    eps_pri = list()
    eps_dual = list()

    stat_b = sc.broadcast(stat)
    rho_b = sc.broadcast(rho)
    x_var_b = sc.broadcast(x_var)
    lambda_b = sc.broadcast(lambda_)
    for j in (range(MAX_ITER)):
        u_b = sc.broadcast(u)
        z_b = sc.broadcast(z)
            
        res = data.mapPartitionsWithSplit(lambda ind,iterator: solveLocally(ind,iterator,x_var_b.value,stat_b.value
                                              ,u_b.value,z_b.value,rho_b.value)).collect()  
        x = np.array(res).T
        
        
        zold = z
        x_hat = alpha*x +(1-alpha)*zold
        z = np.tile(float(N*rho)/(1/lambda_ + N*rho)*np.mean( x_hat + u, 1 ),(N,1)).T
        u = u + (x_hat - z)
        
        x_b = sc.broadcast(x)
        
        obj_val = data.mapPartitionsWithSplit(lambda ind,iterator: objective(ind,iterator,stat_b.value,
                                                                             lambda_b.value,x_b.value,z_b.value[:,0])).reduce(add)
        objval.append(obj_val)
        r_norm.append(np.linalg.norm(x-z,2))
        s_norm.append(np.linalg.norm(-rho*(z-zold),2))
        
        eps_pri.append(np.sqrt(nFeatures+1)*ABSTOL + RELTOL*max(np.linalg.norm(x,2),np.linalg.norm(-z,2)))
        eps_dual.append(np.sqrt(nFeatures+1)*ABSTOL + RELTOL*np.linalg.norm(rho*u,2))

        
        # print str(j) + " " + str(objval[j])+ " " + str(r_norm[j])+ " " + str(s_norm[j])\
        # + " " + str(eps_pri[j])+ " " + str(eps_dual[j]) 
        
        if (r_norm[j] < eps_pri[j] and s_norm[j] < eps_dual[j]):
            break;
        xave = np.mean(x,1)
    print xave

    data =[parseVector(line) for line in open('data.txt')]
    data = np.array(data)
    plot2D(data,xave) 
    #(partition_ind,iterable_data,x_var,stat,u,z,rho)
                        
    #data.mapPartitionsWithSplit(f, preservesPartitioning)

    # y= line_X.map(parseVector).cache()
    #rdd.mapPartitionsWithSplit(lambda ind,iterator: f(ind,iterator,u)).collect()

