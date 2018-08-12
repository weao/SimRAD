import time
import numpy as np
from datetime import datetime
import os,sys
from sklearn.externals import joblib
import keras
from keras.models import Sequential, Model
from keras.layers import Dense, Flatten, Reshape, Dropout, Input
from keras.layers import LSTM
from keras.layers import Activation
from keras.layers import MaxPooling1D
import random
import utils

def shuffle_dataset(data_set):

    for key in data_set:
        random.shuffle(data_set[key])
    return data_set

if not os.path.exists('./model/'):
	os.makedirs('./model/')

winSize = 120

data_set = {}

data_set = utils.loadOppASeriesData("OPP-BW-aseries-data.dat")

HL_Label = {0: '-', 101: 'Relaxing', 102: 'Coffee time', 103: 'Early morning', 104: 'Cleanup', 105: 'Sandwich time'}


HL_Label_S = [0, 101, 102, 103, 104, 105]
L_Label_S = [0, 1, 2, 4, 5]
LLL_Label_S = [200, 201, 202, 203, 204, 205, 206, 207, 208, 209, 210, 211, 212, 213]
LLR_Label_S = [400, 401, 402, 403, 404, 405, 406, 407, 408, 409, 410, 411, 412, 413]

fd1 = len(L_Label_S)
fd2 = len(LLL_Label_S)
fd3 = len(LLR_Label_S)

winLa = int(winSize / 2)
winLb = int(winSize / 4)

print keras.backend.backend(), winLa, winLb


random.seed(124)
rstate = random.getstate()

random.setstate(rstate)
shuffle_dataset(data_set)
rstate = random.getstate()


dtnow = datetime.now()

labels = {'L':L_Label_S, 'LLL':LLL_Label_S, 'LLR':LLR_Label_S}

ACC = {0:[]}
ACC_ki = [ [] for ki in range(0, 4) ]

for ki in range(0, 4):

	print 'Fold %d' % ki

	dlen = len(data_set)
	test_ds, train_ds = utils.KfoldCross(data_set, 4, ki)

	clf_path = './model/ASM-win%d-fold%d.cnn' % (winSize, ki)


	clf = None
	if os.path.exists(clf_path):
	    clf = keras.models.load_model(clf_path)


	if clf is not None:
		continue

	print 'Training ASM...'

	AF = []

	CAT = [ None ]*len(HL_Label_S)
	for ci in range(len(CAT)):
	    CAT[ci] = []



	#for each complex activity sample 'ca'
	for ca in train_ds:

	    ctag = HL_Label_S.index(ca['tag'])

	    F, T = utils.getActionSubsequences(ca, winLa, winLb, labels)

	    AF += F
	    CAT[0] += T
	    CAT[ctag] += T

	    TZ = [ [ 0 for j in range(len(T[i])) ] for i in range(len(T)) ]

	    for ci in range(1, len(CAT)):
	        if ci == ctag:
	            continue
	        CAT[ci] += TZ



	trainX = np.reshape(AF, ( len(AF), len(AF[0]), len(AF[0][0]) ) )

	trainY = [None]*len(CAT)
	for ci in range(len(CAT)):
	    AT = CAT[ci]
	    trainY[ci] = np.reshape(AT, ( len(AT), len(AT[0]) ) )



	np.random.seed(123)

	output_dim = trainY[0].shape[1]

	main_input = Input(shape=( trainX.shape[1], trainX.shape[2] ), name='main_input')
	layer = Dense(60)(main_input)
	layer = Dropout(0.1)(layer)
	layer = Flatten()(layer)
	layer = Reshape((540, 1))(layer)
	layer = MaxPooling1D(pool_size=2)(layer)
	layer = Flatten()(layer)
	layer = Dropout(0.2)(layer)
	hub = Dense(256)(layer)


	cls_outputs = [None]*len(CAT)
	for ci in range(len(CAT)):
	    layer = Dense(output_dim)(hub)
	    output_name = 'output-%d' % ci
	    cls_outputs[ci] = Activation('sigmoid', name=output_name)(layer)


	clf = Model(inputs=[main_input], outputs=cls_outputs)

	opter = keras.optimizers.Adam()
	clf.compile(loss='mean_squared_error', optimizer=opter)
	clf.fit(trainX, trainY, epochs=50, batch_size=10, verbose=2)


	clf.save(clf_path)





