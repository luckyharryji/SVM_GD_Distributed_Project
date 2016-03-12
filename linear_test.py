from pyspark.mllib.classification import LogisticRegressionWithSGD
from pyspark.mllib.regression import LabeledPoint
from pyspark import SparkContext, SparkConf
from pyspark.mllib.util import MLUtils
from numpy import array

# Load and parse the data
def parsePoint(line):
    values = [float(x) for x in line.split(' ')]
    return LabeledPoint(values[0], values[1:])


conf = (SparkConf().setAppName("SGD SVM Test").set("spark.executor.memory", "1g"))
sc = SparkContext(conf = conf)
logger = sc._jvm.org.apache.log4j
logger.LogManager.getLogger("org").setLevel( logger.Level.OFF )
logger.LogManager.getLogger("akka").setLevel( logger.Level.OFF )

# parsedData = sc.textFile("data/heart_scale.txt")
parsedData = MLUtils.loadLibSVMFile(sc, 'data/heart_scale.txt')
# 
# parsedData = data.map(parsePoint)

# data_hash = train_data.zipWithUniqueId().map(lambda kv: (kv[1], (kv[0], 0.0))).persist()
# Build the model
model = LogisticRegressionWithSGD.train(parsedData)

# Evaluating the model on training data
labelsAndPreds = parsedData.map(lambda p: (p.label, model.predict(p.features)))
trainErr = labelsAndPreds.filter(lambda (v, p): v != p).count() / float(parsedData.count())
print("Training Error = " + str(trainErr))



# from __future__ import print_function

# import numpy as np
# import sys, os
# from pyspark import SparkContext, SparkConf
# from pyspark.mllib.regression import LabeledPoint
# from pyspark.mllib.util import MLUtils
# from pyspark.mllib.linalg import SparseVector


# # In[2]:

# def rbfKernel(gamma, x1, x2):
#     if type(x1)== SparseVector:
#         return np.exp(-1 * gamma * x1.squared_distance(x2))
#     return np.exp(-1 * gamma * np.square(np.linalg.norm(x1-x2)))


# # In[13]:

# def pPackSVM_train(train_data, n_iter, pack_size, lambda0, gamma, test):
#     #model = train_data.map(lambda x: (x, 0.))
    
    
#     data_hash = train_data.zipWithUniqueId().map(lambda kv: (kv[1], (kv[0], 0.0))).persist()
#     s = 1.
#     norm = 0.
#     alpha = 0.
#     i,j = 0,0
    
#     #pair_idx = sc.parallelize(range(0, pack_size)).flatMap(lambda x : (range(x, pack_size).map(lambda y: (x,y)))).persist()
#     pair_idx = sc.parallelize(range(0, pack_size)).flatMap(lambda x : [(x, y) for y in range(0, pack_size)]).cache()
    
#     for t in range(1, n_iter+1):
        
#         print("iteration:"+str(t))
        
#         sample = data_hash.takeSample(False, pack_size)
        
#         broad_sample = sc.broadcast(sample)
        
#         #y_u = sc.parallelize(broad_sample.value).map(lambda x:  (data_hash.map(lambda kv:  (kv[1][0].lable *  kv[1][1] *  rbfKernel(gamma,kv[1][0].features, x[1][0].features))).reduce(lambda x,y: x+y)))
        
#         #print(type(broad_sample.value))
        
#         y_u = map(lambda x: (data_hash.map(lambda kv:\
#                                            (kv[1][0].label * \
#                                             kv[1][1] * \
#                                             rbfKernel(gamma,kv[1][0].features, x[1][0].features))).reduce(lambda x,y: x+y)),\
#                   broad_sample.value)
        
#         y = map(lambda x: x[1][0].label, broad_sample.value)
#         local_set = {}
        
#         inner_prod = pair_idx.map(lambda x: (tuple(x), rbfKernel(gamma, broad_sample.value[x[0]][1][0].features, broad_sample.value[x[1]][1][0].features))).cache()
#         #inner_prod = pair_idx.map(lambda x: (tuple(x), 0.5)).cache()
        
        
#         # print(inner_prod.count())
#         # print(type(inner_prod))
#         #for inner in inner_prod.take(100):
#         #    print(inner)
#         inner_prod = inner_prod.collectAsMap()

        
#         #Compute sub gradient
#         for i in range(pack_size):
#             t = t * pack_size + i + 1
#             s = s * (1. - (1.0/t))
#             #print(lambda0, t, s)
            
#             for j in range(i+1, pack_size):
#                 y_u[j] *= (1 - 1.0 / t)
            
#             if y[i]*y_u[i] < 1:
                
#                 norm = norm + (2. * y[i]) / (lambda0 * t) * y_u[i] + np.square(y[i] / (lambda0 * t))* inner_prod[(i,i)]
                
#                 alpha = broad_sample.value[i][1][1]
                
                
#                 local_set[broad_sample.value[i][0]] = (broad_sample.value[i][1][0], alpha + (1. / (lambda0 * t *s)))
#                 # print(broad_sample.value[i][0])
#                 for ite in range(i+1,pack_size-1):
#                     y_u[ite] = y_u[ite] + y[ite] / (lambda0 * t) * inner_prod[(i,ite)]
                    
#                 if norm > (1./lambda0):
#                     s = s * (1. / np.sqrt(lambda0 * norm))
#                     norm = 1./ lambda0
#                     for ite in range(i+1, pack_size):
#                         y_u[ite] = y_u[ite] / np.sqrt( lambda0 * norm)
        
        
#         hashtable = data_hash.collectAsMap()
#         data_hash.unpersist()
#         for k, v in local_set.iteritems():
#             hashtable[k] = v
#         data_hash = sc.parallelize(hashtable.iteritems())   
#         data_hash.persist()
#         model_temp = data_hash.map(lambda kv: (kv[1][0], kv[1][1])).filter(lambda kv: kv[1]>0).cache()
#         acc_temp = getAccuracy(test, model_temp, 0.1, s)
#         print("Accuracy: " + str(acc_temp))
    
#     model = data_hash.map(lambda kv: (kv[1][0], kv[1][1])).filter(lambda kv: kv[1]>0).cache()
#     data_hash.unpersist()
    
#     return model, s
                


# # In[4]:

# def pPackSVM_predict(data, model, gamma, s):
    
#     return s * (model.map(lambda kv : kv[1] * kv[0].label * rbfKernel(gamma, data.features, kv[0].features)).reduce(lambda x,y: x+y))
    
    


# # In[5]:

# def getAccuracy(test_data, model, gamma,s):
#     pred = map(lambda x: pPackSVM_predict(x, model, gamma, s) * x.label, test_data)
#     tp = sum(1 for x in pred if x > 0)
#     N = len(test)

#     # print(pred)
#     # print(tp, N)
#     return float(tp) / N


# if __name__ == "__main__":

#     conf = (SparkConf().setAppName("SGD SVM Test").set("spark.executor.memory", "1g"))
#     sc = SparkContext(conf = conf)
#     logger = sc._jvm.org.apache.log4j
#     logger.LogManager.getLogger("org").setLevel( logger.Level.OFF )
#     logger.LogManager.getLogger("akka").setLevel( logger.Level.OFF )


#     data = MLUtils.loadLibSVMFile(sc, 'data/heart_scale.txt')
#     (training, test) = data.randomSplit([0.7, 0.3])

#     # print "data is here", training
#     # print("data goes here")
#     # data_hash = training.zipWithUniqueId().map(lambda kv: (kv[1], (kv[0], 0.0))).persist()
#     # data_hash.foreach(print)
#     # print "training da is :", training
#     # training = MLUtils.loadLibSVMFile('data/a9a')
#     # test = MLUtils.loadLibSVMFile('data/a9a.t')

#     n_train = training.count()
#     test = test.collect()
    
#     iterations = np.arange(1,5) * n_train / 2
#     pack_size = 100
    
#     for n_iters in [100]:
#         model, s = pPackSVM_train(training, n_iters, pack_size, float(0.3)/ n_train, 0.1, test)
#         acc = getAccuracy(test, model, 0.1, s)
    
#     print("Accuracy: " + str(acc))

#     sc.stop()



