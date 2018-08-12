import numpy as np


def paddingTimeSeries(X, r):

    return [0.0]*r + X + [0.0]*r


def statelize(data):
    states = []
    pt = 0
    for t in range(1, len(data)):
        if data[t] != data[t-1]:
            states.append({'a':data[pt], 't':(pt, t)})
            pt = t
    states.append({'a':data[pt], 't':(pt, len(data))})
    return states


def getIntervalSequenceSupports(aS):

    n = len(aS)
    if n == 0:
        return 0, 0

    wint = 0.0
    for a in aS:
        wint += a['t'][1] - a['t'][0]
    wint /= n

    sup = 0.0
    i = 1
    si = 0
    while i < n:

        if aS[i]['t'][0] - aS[i-1]['t'][1] >= wint:
            sup += aS[i-1]['t'][1] + wint - aS[si]['t'][0]
            si = i
        i += 1

    sup += aS[si]['t'][1] - aS[si]['t'][0] + wint
    return sup, wint


def getTemporalPatternFeature(LocA, LeftA, RightA, labels):

    L_Label = labels['L']
    LLL_Label = labels['LLL']
    LLR_Label = labels['LLR']

    LocA = statelize(LocA)
    LeftA = statelize(LeftA)
    RightA = statelize(RightA)

    fea = []

    for tag in L_Label[1:]:
        aS = [x for x in LocA if x['a'] == tag]
        f, wint = getIntervalSequenceSupports(aS)
        fea.append( f / (len(LocA) + wint) )

    for tag in LLL_Label[1:]:
        aS = [x for x in LeftA if x['a'] == tag]
        f, wint = getIntervalSequenceSupports(aS)
        fea.append( f / (len(LeftA) + wint) )

    for tag in LLR_Label[1:]:
        aS = [x for x in RightA if x['a'] == tag]
        f, wint = getIntervalSequenceSupports(aS)
        fea.append( f / (len(RightA) + wint) )

    return fea



def KfoldCross(data_set, K, Ki):

    dlen = len(data_set)

    train_data_set = []
    test_data_set = []
    cnt = 0

    for key in data_set:

        group = data_set[key]
        dlen = len(group)
        dd = int(dlen / K)
        if dd < 1:
            dd = 1
        for i in range(0, dlen):
            piece = group[i]
            if Ki * dd <= i and i < (Ki+1) * dd:
                cnt += 1
                test_data_set.append(piece)
            else:
                train_data_set.append(piece)

    return train_data_set, test_data_set


def loadOppASeriesData(path):
    D = {}
    fin = open(path, 'r')
    lines = fin.readlines()

    cai = 0
    for line in lines:
        vals = line.split(' ')
        tag = int(vals[0])
        if tag not in D:
            D[tag] = []

        uid = str(vals[1])
        ca = {'sid':cai, 'tag':tag, 'uid':uid, 'L':[], 'LLL':[], 'LLR':[], 'Bx':[], 'By':[], 'Bz':[], 'Lx':[], 'Ly':[], 'Lz':[], 'Rx':[], 'Ry':[], 'Rz':[]}

        idx = 2
        l1 = int(vals[idx])
        idx += 1
        for i in range(0, l1):
            ca['L'].append(int(vals[idx]))
            idx += 1


        l2 = int(vals[idx])
        idx += 1
        for i in range(0, l2):
            ca['LLL'].append(int(vals[idx]))
            idx += 1

        l3 = int(vals[idx])
        idx += 1
        for i in range(0, l3):
            ca['LLR'].append(int(vals[idx]))
            idx += 1

        dlen = int(vals[idx])
        idx += 1
        for i in range(0, dlen):
            x, y, z = float(vals[idx]), float(vals[idx+1]), float(vals[idx+2])
            ca['Bx'].append(x)
            ca['By'].append(y)
            ca['Bz'].append(z)
            idx += 3


        dlen = int(vals[idx])
        idx += 1
        for i in range(0, dlen):
            x, y, z = float(vals[idx]), float(vals[idx+1]), float(vals[idx+2])
            ca['Lx'].append(x)
            ca['Ly'].append(y)
            ca['Lz'].append(z)
            idx += 3


        dlen = int(vals[idx])
        idx += 1
        for i in range(0, dlen):
            x, y, z = float(vals[idx]), float(vals[idx+1]), float(vals[idx+2])
            ca['Rx'].append(x)
            ca['Ry'].append(y)
            ca['Rz'].append(z)
            idx += 3


        D[tag].append(ca)
        cai += 1
    return D


def trimCA(ca, dend):

	ca['L'] = ca['L'][:dend]
	ca['LLL'] = ca['LLL'][:dend]
	ca['LLR'] = ca['LLR'][:dend]

	ca['Bx'] = ca['Bx'][:dend]
	ca['By'] = ca['By'][:dend]
	ca['Bz'] = ca['Bz'][:dend]

	ca['Lx'] = ca['Lx'][:dend]
	ca['Ly'] = ca['Ly'][:dend]
	ca['Lz'] = ca['Lz'][:dend]

	ca['Rx'] = ca['Rx'][:dend]
	ca['Ry'] = ca['Ry'][:dend]
	ca['Rz'] = ca['Rz'][:dend]



def getActionSubsequences(ca, winLa, winLb, labels):
    F = []
    T = []

    L_Label_S = labels['L']
    LLL_Label_S = labels['LLL']
    LLR_Label_S = labels['LLR']

    aX = paddingTimeSeries(ca['Bx'], winLa)
    aY = paddingTimeSeries(ca['By'], winLa)
    aZ = paddingTimeSeries(ca['Bz'], winLa)
    dlen = len(aX)

    for i in range(winLa+1, dlen-winLa+1):
        X = aX[i-winLa-1:i+winLa]
        Y = aY[i-winLa-1:i+winLa]
        Z = aZ[i-winLa-1:i+winLa]

        f = [X, Y, Z]

        tag = ca['L'][i-winLa-1]

        y = [0.0]*len(L_Label_S)
        y[L_Label_S.index(tag)] = 1.0

        F.append(f)
        T.append(y)

    aX = paddingTimeSeries(ca['Lx'], winLb)
    aY = paddingTimeSeries(ca['Ly'], winLb)
    aZ = paddingTimeSeries(ca['Lz'], winLb)
    dlen = len(aX)

    for i in range(winLb+1, dlen-winLb+1):
        X = aX[i-winLb-1:i+winLb]
        Y = aY[i-winLb-1:i+winLb]
        Z = aZ[i-winLb-1:i+winLb]

        X += [0.0]*((winLa - winLb)*2)
        Y += [0.0]*((winLa - winLb)*2)
        Z += [0.0]*((winLa - winLb)*2)
        f = [X, Y, Z]


        tag = ca['LLL'][i-winLb-1]

        y = [0.0]*len(LLL_Label_S)
        y[LLL_Label_S.index(tag)] = 1.0

        F[i-winLb-1] += f
        T[i-winLb-1] += y


    aX = paddingTimeSeries(ca['Rx'], winLb)
    aY = paddingTimeSeries(ca['Ry'], winLb)
    aZ = paddingTimeSeries(ca['Rz'], winLb)
    dlen = len(aX)

    for i in range(winLb+1, dlen-winLb+1):
        X = aX[i-winLb-1:i+winLb]
        Y = aY[i-winLb-1:i+winLb]
        Z = aZ[i-winLb-1:i+winLb]

        X += [0.0]*((winLa - winLb)*2)
        Y += [0.0]*((winLa - winLb)*2)
        Z += [0.0]*((winLa - winLb)*2)
        f = [X, Y, Z]

        tag = ca['LLR'][i-winLb-1]

        y = [0.0]*len(LLR_Label_S)
        y[LLR_Label_S.index(tag)] = 1.0
        F[i-winLb-1] += f
        T[i-winLb-1] += y

    return F, T
