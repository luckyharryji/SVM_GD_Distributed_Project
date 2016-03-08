from __future__ import print_function

import numpy as np
import sys, os
from pyspark import SparkContext, SparkConf
from pyspark.mllib.regression import LabeledPoint
from pyspark.mllib.util import MLUtils
from pyspark.mllib.linalg import SparseVector


# In[2]:

def rbfKernel(gamma, x1, x2):
    if type(x1)== SparseVector:
        return np.exp(-1 * gamma * x1.squared_distance(x2))
    return np.exp(-1 * gamma * np.square(np.linalg.norm(x1-x2)))


# In[13]:

def pPackSVM_train(train_data, n_iter, pack_size, lambda0, gamma):
    #model = train_data.map(lambda x: (x, 0.))
    
    
    data_hash = train_data.zipWithUniqueId().map(lambda kv: (kv[1], (kv[0], 0.0))).persist()
    s = 1.
    norm = 0.
    alpha = 0.
    i,j = 0,0
    
    #pair_idx = sc.parallelize(range(0, pack_size)).flatMap(lambda x : (range(x, pack_size).map(lambda y: (x,y)))).persist()
    pair_idx = sc.parallelize(range(0, pack_size)).flatMap(lambda x : [(x, y) for y in range(0, pack_size)]).cache()
    
    for t in range(1, n_iter+1):
        
        print("iteration:"+str(t))
        
        sample = data_hash.takeSample(True, pack_size)
        
        broad_sample = sc.broadcast(sample)
        
        #y_u = sc.parallelize(broad_sample.value).map(lambda x:  (data_hash.map(lambda kv:  (kv[1][0].lable *  kv[1][1] *  rbfKernel(gamma,kv[1][0].features, x[1][0].features))).reduce(lambda x,y: x+y)))
        
        #print(type(broad_sample.value))
        
        y_u = map(lambda x: (data_hash.map(lambda kv:\
                                           (kv[1][0].label * \
                                            kv[1][1] * \
                                            rbfKernel(gamma,kv[1][0].features, x[1][0].features))).reduce(lambda x,y: x+y)),\
                  broad_sample.value)
        
        y = map(lambda x: x[1][0].label, sample)
        local_set = {}
        
        inner_prod = pair_idx.map(lambda x: (tuple(x), rbfKernel(gamma, sample[x[0]][1][0].features, sample[x[1]][1][0].features))).cache()
        #inner_prod = pair_idx.map(lambda x: (tuple(x), 0.5)).cache()
        
        
        # print(inner_prod.count())
        # print(type(inner_prod))
        #for inner in inner_prod.take(100):
        #    print(inner)
        inner_prod = inner_prod.collectAsMap()

        
        #Compute sub gradient
        for i in range(pack_size):
            t = t + 1
            s = s * (1. - (1.0/t))
            #print(lambda0, t, s)
            
            for j in range(i+1, pack_size):
                y_u[j] *= (1 - 1.0 / t)
            
            if y[i]*y_u[i] < 1:
                
                norm = norm + (2. * y[i]) / (lambda0 * t) * y_u[i] + np.square(y[i] / (lambda0 * t))* inner_prod[(i,i)]
                
                alpha = sample[i][1][1]
                
                
                local_set[sample[i][0]] = (sample[i][1][0], alpha + (1. / (lambda0 * t *s)))
                
                for ite in range(i+1,pack_size-1):
                    y_u[ite] = y_u[ite] + y[ite] / (lambda0 * t) * inner_prod[(i,ite)]
                    
                if norm > (1./lambda0):
                    s = s * (1. / np.sqrt(lambda0 * norm))
                    norm = 1./ lambda0
                    for ite in range(i+1, pack_size):
                        y_u[ite] = y_u[ite] / np.sqrt( lambda0 * norm)
        
        
        hashtable = data_hash.collectAsMap()
        data_hash.unpersist()
        for k, v in local_set.iteritems():
            hashtable[k] = v
        data_hash = sc.parallelize(hashtable.iteritems())   
        data_hash.persist()
    
    model = data_hash.map(lambda kv: (kv[1][0], kv[1][1])).filter(lambda kv: kv[1]>0).cache()
    data_hash.unpersist()
    
    return model, s
                


# In[4]:

def pPackSVM_predict(data, model, gamma, s):
    
    return s * (model.map(lambda kv : kv[1] * kv[0].label * rbfKernel(gamma, data.features, kv[0].features)).reduce(lambda x,y: x+y))
    
    


# In[5]:

def getAccuracy(test_data, model, gamma,s):
    pred = map(lambda x: pPackSVM_predict(x, model, gamma, s) * x.label, test_data)
    tp = sum(1 for x in pred if x > 0)
    N = len(test)

    print(pred)
    print(tp, N)
    return float(tp) / N


if __name__ == "__main__":

    conf = (SparkConf().setAppName("SGD SVM Test").set("spark.executor.memory", "1g"))
    sc = SparkContext(conf = conf)
    logger = sc._jvm.org.apache.log4j
    logger.LogManager.getLogger("org").setLevel( logger.Level.OFF )
    logger.LogManager.getLogger("akka").setLevel( logger.Level.OFF )


    data = MLUtils.loadLibSVMFile(sc, 'data/heart_scale.txt')
    (training, test) = data.randomSplit([0.7, 0.3])

    # print "data is here", training
    # print("data goes here")
    # data_hash = training.zipWithUniqueId().map(lambda kv: (kv[1], (kv[0], 0.0))).persist()
    # data_hash.foreach(print)
    # print "training da is :", training
    # training = MLUtils.loadLibSVMFile('data/a9a')
    # test = MLUtils.loadLibSVMFile('data/a9a.t')

    n_train = training.count()
    test = test.collect()
    
    iterations = np.arange(1,5) * n_train / 2
    pack_size = 100
    
    for n_iters in [1000]:
        model, s = pPackSVM_train(training, n_iters, pack_size, 0.3, 0.1)
        acc = getAccuracy(test, model, 0.1, s)
    
    print("Accuracy: " + str(acc))

    sc.stop()