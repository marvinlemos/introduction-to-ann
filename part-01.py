#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Sep 29 09:22:06 2018

@author: marvin
"""

import numpy as np

## 1. Data dimensions

### 2. Scalars: 0-dimension
height = np.array(1.79)
age = np.array(1.79, dtype=np.uint8) ##unsigned int

# age2 becomes int64
age2 = age + 3

age2 = age + np.array(3, dtype = np.uint8)


### 2. Vectors: 1 dimension
# Creating an sample in column-vector form (3x1)
sample = np.array([1.79, 20, 31])
# dimension (3, ): the matrix has only one dimension
sample.shape


# Creating a weigth vector
weight = np.array([3, 1, 8.5])

# linear combination
result = np.array(0)
for i in range(sample.size):
    result = result + sample[i] * weight[i]

print(result)

#### or simple
result = np.dot(sample, weight)
print(result)


# Multipliacao de matrizes
# M1 = m x p
# M2 = p x n
# The number of columns of the first Matrix should be equal to the number of rows of the
# second

samples = np.array([[1.79, 20, 79], [1.65, 28, 45.2]])

weights = np.array([[3, 1], [1, 9], [8.5, 2]])

result = np.matmul(samples, weights)


#### How to represent a neuron???

## We can add -1 to the sample
# sample: [HEIGHT, KG, AGE]
sample = np.array([1.79, 50, 40, -1])

### And add the Activation Threshold (Limiar de ativacao) to the weights. Suppose our
### Thresholds is equal to 0.5

weight = np.array([3, 3.5, -1.5, 100])

# result = x1*w1 + x2*w2 + x3*w3 -1*theta
result = np.dot(sample, weight)
print("u = %.2f" % result)

def step(x):
    return 1 if x >= 0 else 0


y = step(result)
print("y =  %.2f" % y)
