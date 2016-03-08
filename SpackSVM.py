
# coding: utf-8

# In[2]:

from __future__ import print_function

import numpy as np
import sys, os
import random

from pyspark import SparkContext, SparkConf
from pyspark.mllib.regression import LabeledPoint
from pyspark.mllib.util import MLUtils
from pyspark.mllib.linalg import SparseVector



# In[3]:


def rbfKernel(gamma, x1, x2):
    if type(x1)== SparseVector:
        return np.exp(-1 * gamma * x1.squared_distance(x2))
    return np.exp(-1 * gamma * np.square(np.linalg.norm(x1-x2)))



def sPackSVM_train(train_data, n_iter, lambda0, gamma):
    data_hash = train_data.map(lambda x: (x, 0.0)).collectAsMap()

    s = 1.
    norm = 0.
    alpha = 0.
    i,j = 0,0
 
    for t in range(1, n_iter+1):
        
        print("iteration:"+str(t))
        
        data = random.choice(data_hash.keys())
        
        model = sc.parallelize(data_hash.iteritems())
        y_u = s * (model.map(lambda kv : kv[1] * kv[0].label * rbfKernel(gamma, data.features, kv[0].features)).reduce(lambda x,y: x+y))
        
        y = data.label
        x = data.features
        #Compute sub gradient

        t = t + 1
        s = s * (1. - 1.0/t)

        if y*y_u < 1:

            norm = norm + (2. * y) / (lambda0 * t) * y_u + np.square(y / (lambda0 * t))* rbfKernel(gamma, x,x)
          
            data_hash[data] += 1. / (lambda0 * t *s)


            if norm > (1./lambda0):
                s = s * (1. / np.sqrt(lambda0 * norm))
                norm = 1./ lambda0
        
        acc_temp = getAccuracy(train_data.collect(), model, gamma, s)
        print("Accuracy: " + str(acc_temp))        
        model.unpersist()
    
    model = sc.parallelize(data_hash.iteritems()).persist()

    return model, s
                


# In[4]:

def pPackSVM_predict(data, model, gamma, s):
    
    return s * (model.map(lambda kv : kv[1] * kv[0].label * rbfKernel(gamma, data.features, kv[0].features)).reduce(lambda x,y: x+y))
    
    


# In[5]:

def getAccuracy(test_data, model, gamma,s):
    pred = map(lambda x: pPackSVM_predict(x, model, gamma, s) * x.label, test_data)
    tp = sum(1 for x in pred if x > 0)
    N = len(test)

    return float(tp) / N


if __name__ == "__main__":

    conf = (SparkConf().setAppName("SGD SVM Test").set("spark.executor.memory", "1g"))
    sc = SparkContext(conf = conf)
    
    logger = sc._jvm.org.apache.log4j
    logger.LogManager.getLogger("org").setLevel( logger.Level.OFF )
    logger.LogManager.getLogger("akka").setLevel( logger.Level.OFF )

    data = MLUtils.loadLibSVMFile(sc, 'data/heart_scale.txt')
    (training, test) = data.randomSplit([0.7, 0.3])
    

    n_train = training.count()
    test = test.collect()
    
    iterations = np.arange(1,5) * n_train / 2
    pack_size = 100
    gamma = 1.
    
    for n_iters in [100]:
        model, s = sPackSVM_train(training, n_iters, 0.3/ n_train, gamma)
        acc = getAccuracy(test, model, gamma, s)
    
    print("Accuracy: " + str(acc))

    sc.stop()






# In[ ]:



