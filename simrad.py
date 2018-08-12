import time
import numpy as np
import os,sys
from sklearn.preprocessing import scale
from sklearn.externals import joblib
from sklearn.linear_model import LogisticRegression
import keras
from keras.models import Sequential, Model
from keras.layers import Dense, Flatten, Reshape, Dropout, Input
from keras.layers import LSTM
from keras.layers import Activation
from keras.layers import MaxPooling1D, MaxPooling2D
import random
import utils
import copy



def shuffle_dataset(data_set):

    for key in data_set:
        random.shuffle(data_set[key])
    return data_set



# ============================ main =================================

if not os.path.exists('./model/'):
    os.makedirs('./model/')



winSize = 120
n_iter = 5
levelN = 10


levelNtest = 10



data_set = utils.loadOppASeriesData("OPP-BW-aseries-data.dat")

HL_Label = {0: '-', 101: 'Relaxing', 102: 'Coffee time', 103: 'Early morning', 104: 'Cleanup', 105: 'Sandwich time'}

HL_Label_S = [0, 101, 102, 103, 104, 105]
L_Label_S = [0, 1, 2, 4, 5]
LLL_Label_S = [200, 201, 202, 203, 204, 205, 206, 207, 208, 209, 210, 211, 212, 213]
LLR_Label_S = [400, 401, 402, 403, 404, 405, 406, 407, 408, 409, 410, 411, 412, 413]

fd1 = len(L_Label_S)
fd2 = len(LLL_Label_S)
fd3 = len(LLR_Label_S)

cls_n = 5

winLa = int(winSize / 2)
winLb = int(winSize / 4)

keras_backend = keras.backend.backend()

np.random.seed(124)
random.seed(124)
rstate = random.getstate()



random.setstate(rstate)
shuffle_dataset(data_set)
rstate = random.getstate()

labels = {'L':L_Label_S, 'LLL':LLL_Label_S, 'LLR':LLR_Label_S}


CRR_AVG = np.array([0.0]*levelNtest)

for ki in range(0, 4):

    print 'Fold %d' % ki

    test_ds, train_ds = utils.KfoldCross(data_set, 4, ki)


    cam_clf_path = './model/CAM-%d-%d-%d-fold%d.clf' % (winSize, n_iter, levelN, ki)
    cam_clf = None
    if os.path.exists(cam_clf_path):
        cam_clf = joblib.load(cam_clf_path)

    asm_clf_path = './model/ASM-win%d-fold%d.cnn' % (winSize, ki)

    asm_clf = None
    if os.path.exists(asm_clf_path):
        asm_clf = keras.models.load_model(asm_clf_path)

    if asm_clf is None and cam_clf is None:
        continue

    print 'Testing SimRAD...'

    crr = 0
    tot = 0
    CRR = [0]*levelNtest

    for ca in test_ds:

        ctag = ca['tag']

        py = np.array([0.0]*cls_n)

        for levi in range(0, levelNtest):

            plevel = float(levi + 1) / levelNtest
            dend = int(len(ca['L'])*plevel)

            cca = copy.deepcopy(ca)
            utils.trimCA(cca, dend)


            F, T = utils.getActionSubsequences(cca, winLa, winLb, labels)

            testX = np.reshape(F, ( len(F), len(F[0]), len(F[0][0]) ) )
            RR = asm_clf.predict(testX)


            R = RR[:]
            for i in range(1,len(RR)):
                R[i] = py[i-1] * RR[i]
            R = np.sum(R, axis=0)


            LocA = [  L_Label_S[np.argmax(R[i][:fd1])] for i in range(0, len(R)) ]
            LeftA = [ LLL_Label_S[np.argmax(R[i][fd1:fd1+fd2])] for i in range(0, len(R)) ]
            RightA = [ LLR_Label_S[np.argmax(R[i][fd1+fd2:])] for i in range(0, len(R)) ]

            ftp = utils.getTemporalPatternFeature(LocA, LeftA, RightA, labels)

            # probability p(y | A)
            py = cam_clf.predict_proba([ftp])[0]


            if HL_Label_S[np.argmax(py)+1] == cca['tag']:
                CRR[levi] += 1



    tot = len(test_ds)
    for levi in range(0, len(CRR)):
        CRR[levi] = float(CRR[levi])/tot
    print CRR

    CRR_AVG += np.array(CRR)


print 'Overall:'
CRR_AVG /= 4
print list(CRR_AVG)
