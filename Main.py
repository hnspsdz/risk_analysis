import os
import numpy as np
import time
import sys

from risk4r50 import Risk4r40


#--------------------------------------------------------------------------------   

def runTrain():
    RSENET50='RESNET-50'

    nnArchitecture = RSENET50
    nnIsTrained = True
    nnClassCount = 2  # 14

    trBatchSize = 8
    trMaxEpoch =30

    print('=== Training NN architecture = ', nnArchitecture, '===')
    Risk4r40.train(  nnIsTrained, nnClassCount, trBatchSize,
                         trMaxEpoch, 10, 'chest_RSNA100', '/chest_RSNA100.pth',None)


# --------------------------------------------------------------------------------
if __name__ == '__main__':
 runTrain()


