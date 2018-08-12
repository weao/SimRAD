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

max_dlen = 0


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

    if cam_clf is not None:
        continue

    print 'Training CAM...'

    cam_clf = LogisticRegression(C=0.5, multi_class='multinomial', class_weight='balanced', solver='lbfgs', warm_start=True)


    trainR = [None]*len(train_ds)

    for si in range(0, len(train_ds)):

        ca = train_ds[si]
        F, _ = utils.getActionSubsequences(ca, winLa, winLb, labels)

        testX = np.reshape(F, ( len(F), len(F[0]), len(F[0][0]) ) )
        trainR[si] = asm_clf.predict(testX)


    for itr in range(0, n_iter):


        WL = []
        FL = []
        T = []

        for levi in range(0, levelN):

            PY = [ np.array([0.0]*(cls_n)) for si in range(len(train_ds))  ]

            plevel = float(levi + 1) / levelN

            #for each complex activity sample 'ca'
            for si in range(0, len(train_ds)):

                ca = train_ds[si]

                ctag = ca['tag']

                # RR is original result, should not be modified.
                RR = trainR[si]

                py = PY[si]

                # copy RR to R
                R = RR[:]
                for i in range(1,len(RR)):
                    R[i] = py[i-1] * RR[i]
                R = np.sum(R, axis=0)

                dend = int(len(R)*plevel)

                LocA = [  L_Label_S[np.argmax(R[i][:fd1])] for i in range(0, dend) ]
                LeftA = [ LLL_Label_S[np.argmax(R[i][fd1:fd1+fd2])] for i in range(0, dend) ]
                RightA = [ LLR_Label_S[np.argmax(R[i][fd1+fd2:])] for i in range(0, dend) ]


                ftp = utils.getTemporalPatternFeature(LocA, LeftA, RightA, labels)

                if itr >= 1:
                    py = cam_clf.predict_proba([ftp])[0] + abs(np.random.normal(0, 0.05, cls_n))
                    py = py / sum(py)
                    PY[si] = py

                FL.append(ftp)

                # penalty
                WL.append(plevel*plevel)

                ci = HL_Label_S.index(ctag)
                T.append(ci-1)

        cam_clf.fit(FL, T, sample_weight=WL)



    joblib.dump(cam_clf, cam_clf_path)



