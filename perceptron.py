#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Oct  6 11:18:00 2018

@author: marvin
"""

import numpy as np
import matplotlib.pyplot as plt

def bi_step(x):
    return 1 if x >= 0 else -1



#samples = np.array([[-1, 0.1, 0.4, 0.7], 
#                    [-1, 0.3, 0.7, 0.2],
#                    [-1, 0.6, 0.9, 0.8],
#                    [-1, 0.5, 0.7, 0.1]])
    
samples = np.array([[-1, 0.1, 0.4], 
                    [-1, 0.3, 0.7],
                    [-1, 0.6, 0.9],
                    [-1, 0.5, 0.7]])    


outcomes = np.array([1, -1, -1, 1])

class_a = samples[np.argwhere(outcomes == 1)]
class_b = samples[np.argwhere(outcomes == -1)]

plt.scatter([s[0][1] for s in class_a], [s[0][2] for s in class_a], s = 25, color = 'blue')
plt.scatter([s[0][1] for s in class_b], [s[0][2] for s in class_b], s = 25, color = 'red')

learning_rate = 0.5
epoch = 0

weights = np.random.normal(size=len(samples[0]))
error = True

while error:
    error = False
    for row, sample in enumerate(samples):
        u = sample.dot(weights)
        y = bi_step(u)
        #print(y)
        if y != outcomes[row]:
            # Be careful... (y - outcomes[row]) do not converge.
            weights = weights + learning_rate * (outcomes[row] - y) * sample
            error = True
    epoch += 1
    print(epoch)

print('converged!')
print('new weights: ', weights)

u = samples[3].dot(weights)
y = bi_step(u)
print(y)
        
        
        
    
    
    
