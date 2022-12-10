import numpy as np

def get_AP(query, predict): #for each picture in query
    
    precision = []
    test = predict
    states = query
    mask = np.in1d(test, states)
    n = np.argwhere(mask == True).flatten()
    s = 0
    for i in n:
        s = s + 1
        precision.append(s/(i+1))
    AP = sum(precision)/len(n)

    return AP

def get_CMC(query, predict):

    top1 = 0
    top5 = 0
    cmc = np.zeros(len(predict))
    test = predict
    states = query
    mask = np.in1d(test, states)
    n = np.argwhere(mask == True).flatten()
    cmc[n] = 1
    if cmc[0]==1: top1 = 1
    if sum(cmc[0:5])>0: top5 = 1

    return top1, top5

