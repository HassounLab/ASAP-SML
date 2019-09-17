import random
from matplotlib import rc, rcParams
import matplotlib.pyplot as plt
import scipy.stats as sta
from sklearn.ensemble import ExtraTreesClassifier
from sklearn import svm
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from scipy import interp
from sklearn.metrics import roc_curve, auc
import numpy as np
import pandas as pd

np.random.seed(8)

BLOSUM62_DIRECT = "./data/blosum62.csv"

SET_NAME = 'MMP-cluster'
IF_ONLY_HEAVY = False
CNT_DB = 2
CNT_TARGET = 1
REFERENCE_PATH_TESTCASE = './testCase/MMP-cluster/reference-PDB/'
TARGETING_PATH_TESTCASE = './testCase/MMP-cluster/targeting-MMP/'
TARGET_DESIRE_SIZE = 166 #44 #MMP-cluster



#################################################################################################################
#  function DuplicateSelectFeature:
#  Sample with replacement, each time the selection is on the total 
# 
#  Input:   DatasetName, DatasetSize, AllFeatureVectors
#  Output: 1. X_DS
#          2. Y_DS
#          3. SeqName_DS
#################################################################################################################
def DuplicateSelectFeature(size, DatasetName, DatasetSize, AllFeatureVectors):
    X_DS = []
    Y_DS = []
    SeqName_DS =[]
    previous = 0
    for i in range(len(DatasetSize)):
        if i < CNT_DB:
            actual_size = int(size * CNT_TARGET / CNT_DB)
        else:
            actual_size = size


        if actual_size <= DatasetSize[i]:
            shuffle_x = np.array([sh_i for sh_i in range(DatasetSize[i])])
            np.random.shuffle(shuffle_x)
            for j in range(actual_size):
                # idx = np.random.randint(DatasetSize[i])
                idx = shuffle_x[j]
                SeqName_DS.append(DatasetName[i]+'_'+str(idx))
                X_DS.append(AllFeatureVectors[previous+idx])
                if i < CNT_DB:
                    Y_DS.append(0)
                else:
                    Y_DS.append(1)
            previous += DatasetSize[i]

        else:
            for j in range(actual_size):
                idx = np.random.randint(DatasetSize[i])
                SeqName_DS.append(DatasetName[i]+'_'+str(idx))
                X_DS.append(AllFeatureVectors[previous+idx])
                if i < CNT_DB:
                    Y_DS.append(0)
                else:
                    Y_DS.append(1)
            previous += DatasetSize[i]
    return X_DS, Y_DS, SeqName_DS

#################################################################################################################
#  function IterationDuplicateSelectFeature:
#  Iteratively sample with replacement
# 
#  Input:   DatasetName, DatasetSize, AllFeatureVectors
#  Output: 1. X_DS
#          2. Y_DS
#          3. SeqName_DS
#################################################################################################################
def IterationDuplicateSelectFeature(size, iterate, DatasetName, DatasetSize, AllFeatureVectors):
    X_IDS = [[] for i in range(iterate)]
    Y_IDS = [[] for i in range(iterate)]
    SeqName_IDS = [[] for i in range(iterate)]
    for i in range(iterate):
        X_DS, Y_DS, SeqName_DS = DuplicateSelectFeature(size, DatasetName, DatasetSize, AllFeatureVectors)
        X_IDS[i] = X_DS
        Y_IDS[i] = Y_DS
        SeqName_IDS[i] = SeqName_DS
    return X_IDS, Y_IDS, SeqName_IDS


#################################################################################################################
#  function normalize:
#  Normalize the distance matrix
# 
#  Input:  dist
#  Output: 1. tmp_dist
#################################################################################################################

def normalize(dist):
    tmp_min = dist.min()
    tmp_max = dist.max()
    tmp_dist = (dist-tmp_min)/(tmp_max-tmp_min)
    return tmp_dist

#################################################################################################################
#  function Draw_heatmap:
#  Draw heatmap according to the input distance matrix 
# 
#  Input:  dist, name
#################################################################################################################
def Draw_heatmap(size, dist, name, DatasetSize):
    rc('font', size=20)  
    #ticks bold
    rcParams['text.latex.preamble'] = [r'\usepackage{sfmath} \boldmath']
    # figure size
    fig, ax = plt.subplots(figsize=(10, 7))
    # heatmap with color bar

    dist = normalize(dist)
    plt.imshow(dist, cmap='Blues', interpolation='nearest')
    v = np.linspace(0.0, 1.0, 6, endpoint=True)
    cb = plt.colorbar(ticks=v)
    plt.title(name, y=1.08)

    N_DB = size * (CNT_TARGET / CNT_DB)
    N_TARGET = size

    x = []
    for i in range(CNT_DB):
        x.append((i+0.5) * N_DB)
    for i in range(CNT_TARGET):
        x.append((i + 0.5) * N_TARGET + CNT_DB * N_DB)

    y = x

    if SET_NAME == 'IGHV':
        labels = ['Reference'] + ['MMP-targeting']
    elif SET_NAME == 'MMP':
        labels = ['Human', 'Murine'] + ['MMP-targeting']
    elif SET_NAME == 'MMP-cluster':
        labels = ['Human', 'Murine'] + [str(i + 1) for i in range(CNT_TARGET)]
    elif SET_NAME == 'DEKOSKY':
        labels = ['Naive 1', 'Naive 2'] + [str(i+1) for i in range(CNT_TARGET)]
    else:
        labels = ['Naive 1', 'Naive 2'] + [str(i + 1) for i in range(CNT_TARGET)]
    plt.xticks(x, labels)
    plt.yticks(y, labels)

    ax.xaxis.tick_top()
    
    a = []
    for i in range(CNT_DB):
        a.append((i+1) * N_DB)

    for i in range(CNT_TARGET - 1):
        a.append((i + 1) * N_TARGET + CNT_DB * N_DB)

    for idx, item in enumerate(a):
        if idx == CNT_DB-1:
            ax.axhline(item, linestyle='-', color='black', linewidth=3)
            ax.axvline(item, linestyle='-', color='black', linewidth=3)
        else:
            ax.axhline(item, linestyle='-', color='black', linewidth=1)
            ax.axvline(item, linestyle='-', color='black', linewidth=1)

    fig.savefig('./results/'+SET_NAME +'_'+ name+'.png')

######################  Section 3.1 Sequence and feature similarity analysis (Heat map) ##########################

#################################################################################################################
#  function ReadBLOSUM:
#  Read in the BLOSUM 62 substitution matrix
# 
#  Output: 1. BLOSUM, a dictionary of pairwise permutation
#################################################################################################################
def ReadBLOSUM():
    with open(BLOSUM62_DIRECT, "r") as fi:
        data = fi.readlines()
    for i in range(len(data)):
        data[i] = data[i].strip().split(',')

    names = data[0]
    BLOSUM = {}
    for i in range(len(names)):
        for j in range(len(names)):
            BLOSUM[names[i] + names[j]] = data[i + 1][j]
    return BLOSUM

#################################################################################################################
#  function CalBLOSUM:
#  Calculate the sequence similarity for each sequence.
# 
#  Input:  SeqName_DS, Amino, Num, BLOSUM, chain
#  Output: 1. dist
#################################################################################################################
def CalBLOSUM(SeqName_DS, Amino, Num, BLOSUM, chain):
    dist = np.zeros((len(SeqName_DS), len(SeqName_DS)))
    for i, s1 in enumerate(SeqName_DS):
        seq1 = {}
        for k in range(len(Amino[chain][s1])):
            seq1[Num[chain][s1][k]] = Amino[chain][s1][k]
        for j, s2 in enumerate(SeqName_DS):
            seq2 = {}
            for k in range(len(Amino[chain][s2])):
                seq2[Num[chain][s2][k]] = Amino[chain][s2][k]
            cnt = 0
            for key in seq1:
                if key in seq2 and (seq1[key] + seq2[key]) in BLOSUM:
                    cnt += int(BLOSUM[seq1[key] + seq2[key]])
                else:
                    cnt += -4
            for key in seq2:
                if key not in seq1:
                    cnt += -4
            dist[i][j] = cnt
    return dist

#################################################################################################################
#  function CalBLOSUMVAR:
#  Calculate the sequence similarity for each sequence only on non-constant region.
#
#  Input:  SeqName_DS, Amino, Num, BLOSUM, chain
#  Output: 1. dist
#################################################################################################################
def CalBLOSUMVAR(SeqName_DS, Amino, Num, BLOSUM, chain):
    dist = np.zeros((len(SeqName_DS), len(SeqName_DS)))
    # should be the same, since being normalized afterwards

    for i, s1 in enumerate(SeqName_DS):
        seq1 = {}
        for k in range(len(Amino[chain][s1])):
            seq1[Num[chain][s1][k]] = Amino[chain][s1][k]
        for j, s2 in enumerate(SeqName_DS):
            seq2 = {}
            for k in range(len(Amino[chain][s2])):
                seq2[Num[chain][s2][k]] = Amino[chain][s2][k]
            cnt = 0
            for key in seq1:
                if key in seq2 and (seq1[key] + seq2[key]) in BLOSUM:
                    cnt += int(BLOSUM[seq1[key] + seq2[key]])
                else:
                    cnt += -4
            for key in seq2:
                if key not in seq1:
                    cnt += -4
            dist[i][j] = cnt
    return dist

#################################################################################################################
#  function HeatmapHL:
#  Calculate the heavy and light chain heatmap over multiple iteration, draw the first heatmap
# 
#  Input:  SeqName_IDS, Amino, Num
#  Output: 1. H_Idist
#          2. L_Idist
#################################################################################################################
def HeatmapHL(size, iterate, SeqName_IDS, Amino, Num):
    iterate = 1
    BLOSUM = ReadBLOSUM()
    H_Idist = []
    L_Idist = []
    for i in range(iterate):
        H_Idist.append(CalBLOSUM(SeqName_IDS[i], Amino, Num, BLOSUM, 'H'))
        H_Idist.append(CalBLOSUMVAR(SeqName_IDS[i], Amino, Num, BLOSUM, 'H'))
        if IF_ONLY_HEAVY:
            continue
        L_Idist.append(CalBLOSUM(SeqName_IDS[i], Amino, Num, BLOSUM, 'L'))
        L_Idist.append(CalBLOSUMVAR(SeqName_IDS[i], Amino, Num, BLOSUM, 'L'))
    return H_Idist, L_Idist

#################################################################################################################
#  function HeatmapFeature:
#  Calculate the feature heatmap over multiple iteration, draw the first heatmap
# 
#  Input:  X_IDS, AllFeatureNames, MotifFeatureNames
#  Output: 1. Idist
#################################################################################################################
def HeatmapFeature(size, iterate, X_IDS, AllFeatureNames, MotifFeatureNames):
    iterate = 1
    motifStart = len(AllFeatureNames) - len(MotifFeatureNames)
    Idist = [np.zeros((len(X_IDS[0]), len(X_IDS[0]))) for i in range(iterate)]
    for m in range(iterate):
        for i in range(len(X_IDS[m])):
            a = X_IDS[m][i]
            for j in range(i + 1):
                b = X_IDS[m][j]

                AandB = 0
                AorB = 0
                extr = [0 for x in range(len(X_IDS[m][j]))]
                for l in range(motifStart, len(X_IDS[m][i])):
                    if a[l] == 1 and b[l] == 1:
                        AandB += 1
                    if a[l] == 1 or b[l] == 1:
                        AorB += 1
                if AorB == 0:
                    jaccard = 0
                else:
                    jaccard = AandB / (1.0 * AorB)

                for k in range(0, motifStart):
                    if a[k] == 1 and b[k] == 1:
                        extr[k] = 1
                    else:
                        extr[k] = 0

                extr = np.array(extr)
                # jaccar score for motif, use except motif sum and motif jaccard score
                Idist[m][i][j] = np.sum(extr) + jaccard
                Idist[m][j][i] = Idist[m][i][j]
    # if SET_NAME=='MMP-cluster':
    #     Idist_new = []
    #     for j in range(len(Idist[0])):
    #         if AllFeatureNames[j].startswith('Germ_HV') or AllFeatureNames[j].startswith('Canonical_H2') or AllFeatureNames[
    #             j].startswith('Canonical_L2') \
    #                 or AllFeatureNames[j].startswith('Canonical_L3') or AllFeatureNames[j].startswith('Canonical_H1'):
    #             continue
    #         Idist_new.append(Idist[0][:, j])
    #     Idist_new = np.array(Idist_new)
    #     Idist_new = Idist_new.T
    #     Idist = Idist_new
    return Idist

###############################  Section 3.3 Similarity analysis (Statistical test) #############################

#################################################################################################################
#  function RankTestBlock:
#  Use Mann-Whitney test to check if hypothsis on the within set holds
# 
#  Input:  dist
#  Output: 1. p_value
#################################################################################################################
def RankTestBlock(size, dist):
    stop = int(len(dist)/2)
    block1 = np.reshape(dist[:stop, :stop], [-1])
    block4 = np.reshape(dist[stop:, stop:], [-1])
    mean1 = np.mean(block1)
    mean4 = np.mean(block4)
    std1 = np.std(block1)
    std4 = np.std(block4)
    effect_size1 = (mean1 - mean4)/std1
    effect_size4 = (mean1 - mean4) / std4
    p_value  = sta.ranksums(block1, block4) #, alternative='less')
    return p_value

#################################################################################################################
#  function RankTestBlock:
#  Use Mann-Whitney test to check if hypothsis on the correlation between heatmaps holds
# 
#  Input:  dist1, dist2
#  Output: 1. p_value
#################################################################################################################
def RankTestHeatMap(dist1, dist2):
    map1 = np.reshape(dist1, [-1])
    map2 = np.reshape(dist2, [-1])
    mean1 = np.mean(map1)
    mean2 = np.mean(map2)
    std1 = np.std(map1)
    std2 = np.std(map2)
    effect_size1 = (mean1 - mean2) / std1
    effect_size2 = (mean1 - mean2) / std2
    p_value = sta.ranksums(map1, map2)#, alternative='less')
    return p_value

#################################################################################################################
#  function MultiRankTest:
#  Check if the statistical test for two hypothesis holds over multiple iterations
# 
#  Input:  F_Idist, H_Idist, L_Idist
#################################################################################################################
def MultiRankTest(size, iterate, F_Idist, H_Idist, L_Idist):
    iterate = 1
    p_value_F = [[] for i in range(iterate)]
    p_value_H = [[] for i in range(iterate)]
    p_value_L = [[] for i in range(iterate)]
    p_value_Diff = [[] for i in range(iterate)]
    for i in range(iterate):
        # Wilcxon sum rank test, with effect size
        p_value_F[i] = RankTestBlock(size, F_Idist[i])
        p_value_H[i] = RankTestBlock(size, H_Idist[i])
        if not IF_ONLY_HEAVY:
            p_value_L[i] = RankTestBlock(size, L_Idist[i])
            p_value_Diff[i] = RankTestHeatMap(F_Idist[i] - H_Idist[i], F_Idist[i] - L_Idist[i])
    print(p_value_F[0], p_value_H[0])
    if np.max(p_value_F)<0.05 and np.max(p_value_H)<0.05 and (IF_ONLY_HEAVY or np.max(p_value_L)<0.05): #and np.max(p_value_Diff)<0.05:
        print("Statistical tests (Reference against Targeting) succeed.")
    else:
        print("Statistical tests results (Reference against Targeting):")
        print('Extracted features:', p_value_F[0])
        print('Heavy chain sequence:', p_value_H[0])
        if not IF_ONLY_HEAVY:
            print('Light chain sequence:', p_value_L[0])
            print('Difference between (Feature, Heavy) and (Feature, Light):', p_value_Diff)

#######################################  Section 3.4 Salient feature-value analysis  ############################                                     #


#################################################################################################################
#  function Fisher:
#  Calculate p-value with FET
# 
#  Input:  X_DS, Y_DS, AllFeatureNames
#  Output: 1. contingency_table
#          2. pvalue
#          3. X_DS
#          4. Y_DS
#################################################################################################################
def Fisher(X_DS, Y_DS, AllFeatureNames):
    contingency_table=[[] for i in range(len(AllFeatureNames))]
    N_feature = len(AllFeatureNames)
    pvalue= [0 for i in range(N_feature)]
    for i, name in enumerate(AllFeatureNames):
        a, b, c, d = 0, 0, 0, 0
        for j in range(len(Y_DS)):
            if Y_DS[j] == 0:
                if X_DS[j][i]==1:
                    a += 1
                else:
                    c += 1
            else:
                if X_DS[j][i]==1:
                    b += 1
                else:
                    d += 1
        contingency_table[i] = [[a, b], [c, d]]
    for i in range(N_feature):
        oddsratio, pv = sta.fisher_exact(contingency_table[i], "less")  # greater sig in DB # less sig in patent, "two-sided"
        pvalue[i] = pv
    return contingency_table, pvalue, X_DS, Y_DS

#################################################################################################################
#  function Importance:
#  Calculate importance score through feature selection
# 
#  Input:  X_DS, Y_DS, AllFeatureNames
#  Output: 1. importances
#          2. X_DS
#          3. Y_DS
#################################################################################################################
def Importance(X_DS, Y_DS, AllFeatureNames):
    # X_DS, Y_DS, SeqName_DS = DuplicateSelectFeature(DatasetName, DatasetSize, AllFeatureVectors, size)
    clf_featureSelect = ExtraTreesClassifier()
    clf_featureSelect = clf_featureSelect.fit(X_DS, Y_DS)
    importances = clf_featureSelect.feature_importances_
    # print(len(X_DS[0]), len(AllFeatureNames))
    X_DS = np.array(X_DS)
    # Y_DS = np.array(Y_DS)
    # print(len(X_DS))
    a = AllFeatureNames.index('Germ_HJ_IGHJ4*02')
    b = AllFeatureNames.index('Motif_5_YY')
    # b = AllFeatureNames.index('PI_3.5-3.9375')
    ###################################################################################################################################################
    # b = AllFeatureNames.index('Canonical_H3_3')
    # b = AllFeatureNames.index('Germ_HJ_IGHJ6*01')
    sum_ref = 0
    for j in range(int(len(X_DS)/2)):
        if X_DS[j,a]==1 and X_DS[j,b]==0:
            sum_ref+=1
    sum_tar = 0
    for j in range(int(len(X_DS)/2),len(X_DS)):
        if X_DS[j,a]==1 and X_DS[j,b]==0:
            sum_tar +=1

    # print(AllFeatureNames[a], AllFeatureNames[b], 'reference: ',sum_ref, 'targeting: ',sum_tar)
    return importances, X_DS, Y_DS, sum_ref, sum_tar

#################################################################################################################
#  function RankFisherFS:
#  Sort feature values according to FET and feature selection statistics
# 
#  Input:  Fpvalue, importances
#  Output: 1. RankFpvalue
#          2. RankImportance
#################################################################################################################
def RankFisherFS(Fpvalue, importances):
    RankFpvalue =[-1 for i in range(len(Fpvalue))]
    s_Fpvalue = sorted(range(len(Fpvalue)), key=lambda k: Fpvalue[k])
    for rank, idx in enumerate(s_Fpvalue):
        RankFpvalue[idx] = rank+1 # real rank start from 1

    RankImportance =[-1 for i in range(len(importances))]
    s_Importance = sorted(range(len(importances)), key=lambda k: importances[k], reverse = True)
    for rank, idx in enumerate(s_Importance):
        RankImportance[idx] = rank+1 # real rank start from 1
    return RankFpvalue, RankImportance

#################################################################################################################
#  function WriteFisherFS:
#  Write FET and feature selection results to csv files
# 
#  Input:  Fpvalue, importances, Fpvalue_std, importances_std, RankFpvalue, RankImportance, AllFeatureNames
#################################################################################################################
def WriteFisherFS(Fpvalue, importances, Fpvalue_std, importances_std, RankFpvalue, RankImportance, AllFeatureNames, AllFeatureVectors, DatasetSize):
    fo = open('./results/'+SET_NAME+'_RankFisherAndFS.csv', 'w')
    fo.write('Feature, Feature Value,')
    cnt_db = int(sum(DatasetSize[:CNT_DB]))
    cnt_mmp = int(sum(DatasetSize[CNT_DB:]))

    fo.write('Fisher Test p-value, Feature Selection (thereshold = ' + format(np.mean(importances), '.4f') + '),')
    fo.write('Rank of Statistic Significancy, Rank of Feature Selection, ')
    fo.write('Frequency in Reference , Frequency in Targeting \n')
    AgreeFeature = []
    for i in range(len(AllFeatureNames)):
        if AllFeatureNames[i].split('_')[0] == 'Germ' or AllFeatureNames[i].split('_')[0] == 'Canonical':
            fo.write(AllFeatureNames[i].split('_')[0] + ' ' + AllFeatureNames[i].split('_')[1])
            fo.write(','+AllFeatureNames[i].split('_')[2]+',')
        elif AllFeatureNames[i].split('_')[0] == 'PI':
            fo.write(AllFeatureNames[i].split('_')[0]+','+AllFeatureNames[i].split('_')[1]+',')
        elif AllFeatureNames[i].split('_')[0] == 'Motif':
            fo.write(AllFeatureNames[i].split('_')[0]+',')
            fo.write(AllFeatureNames[i].split('_')[1] + '_' + AllFeatureNames[i].split('_')[2]+',')
        fo.write(str(Fpvalue[i])+',')
        fo.write(str(importances[i])+',')

        if Fpvalue[i]<0.05:
            fo.write(str(RankFpvalue[i]))
        fo.write(',')
        if importances[i]>np.mean(importances):
            fo.write(str(RankImportance[i]))
        fo.write(',')
        fo.write(str('{:.2f}'.format(sum(AllFeatureVectors[:cnt_db, i])/cnt_db * 100)) + '%,')
        fo.write(str('{:.2f}'.format(sum(AllFeatureVectors[cnt_db:, i])/cnt_mmp * 100)) + '%,')

        fo.write('\n')
        if Fpvalue[i]<0.05 and importances[i]>np.mean(importances):
            AgreeFeature.append(i)
    print(AllFeatureVectors.shape, cnt_db, cnt_mmp)
    fo.close()

#################################################################################################################
#  function MultiFisherFS:
#  Average p-values for FET and importance scores for feature select over multiple iterations
# 
#  Input:  DatasetName, DatasetSize, AllFeatureVectors
#################################################################################################################
def MultiFisherFS(iterate, X_IDS, Y_IDS, DatasetName, DatasetSize, AllFeatureVectors, AllFeatureNames):
    Fpvalue =           [[] for i in range(iterate)]
    importances =       [[] for i in range(iterate)]
    RankFpvalue =       [[] for i in range(iterate)]
    RankImportance=     [[] for i in range(iterate)]

    ref_list = [0 for i in range(iterate)]
    tar_list = [0 for i in range(iterate)]
    for i in range(iterate):
        _, Fpvalue[i], _, _ = Fisher(X_IDS[i], Y_IDS[i], AllFeatureNames)
        # importances[i], _, _, ref_list[i], tar_list[i]= Importance(X_IDS[i], Y_IDS[i], AllFeatureNames)
        RankFpvalue[i], _= RankFisherFS(Fpvalue[i], importances[i])

    X_IDS_all = []
    Y_IDS_all = []
    for i in range(iterate):
        X_IDS_all+=X_IDS[i]
        Y_IDS_all+=Y_IDS[i]
    X_IDS_all = np.array(X_IDS_all)
    Y_IDS_all = np.array(Y_IDS_all)

    importances_all, _, _, _, _ = Importance(X_IDS_all, Y_IDS_all, AllFeatureNames)
    RankImportance_all = [-1 for i in range(len(importances_all))]
    s_Importance = sorted(range(len(importances_all)), key=lambda k: importances_all[k], reverse=True)
    for rank, idx in enumerate(s_Importance):
        RankImportance_all[idx] = rank + 1  # real rank start from 1

    Fpvalue_avg = np.mean(Fpvalue, axis = 0)
    # importances_avg = np.mean(importances, axis = 0)

    Fpvalue_std = np.std(Fpvalue, axis=0)
    # importances_std = np.std(importances, axis=0)


    # print('tar', '{:.2f}'.format(100*np.mean(tar_list)*2/len(X_IDS[0]))+'% ','ref','{:.2f}'.format(100*np.mean(ref_list)*2/len(X_IDS[0]))+'% ')
    ####### avgR
    RankFpvalue_avgR = np.mean(RankFpvalue, axis = 0)
    # RankImportance_avgR = np.mean(RankImportance, axis = 0)
    WriteFisherFS(Fpvalue_avg, importances_all,Fpvalue_std,Fpvalue_std, RankFpvalue_avgR, RankImportance_all, AllFeatureNames, AllFeatureVectors, DatasetSize)


#######################################  Section 3.4 Classification on segments  ################################
#################################################################################################################
#  function calculate_auc:
#  Calculate mean AUC over ten-fold cross validation for three algorithms, SVM, random forest, AdaBoost
# 
#  Input:  X, Y
#  Output: 1. auc(mean_fpr, mean_tpr_svm)
#          2. auc(mean_fpr, mean_tpr_rf) 
#          3. auc(mean_fpr, mean_tpr_ada)
#################################################################################################################
def calculate_auc(X, Y):
    clf_svm = svm.SVC(kernel='linear', probability=True, random_state=0)
    clf_randomforest = RandomForestClassifier()  # max_depth=5, n_estimators=10, max_features=1
    clf_adaboost = AdaBoostClassifier()

    X = np.array(X)
    Y = np.array(Y)
    indices = [i for i in range(len(Y))]
    random.shuffle(indices)

    mean_fpr = np.linspace(0, 1, 100)
    tpr_svms = []
    tpr_rfs = []
    tpr_adas = []

    for i in range(10):
        test_i = indices[int(i * len(Y) / 10):int((i + 1) * len(Y) / 10)]
        train_i = indices[:int(i * len(Y) / 10)] + indices[int((i + 1) * len(Y) / 10):]
        X_train, X_test, Y_train, Y_test = X[train_i], X[test_i], Y[train_i], Y[test_i]

        clf_svm = clf_svm.fit(X_train, Y_train)
        clf_randomforest = clf_randomforest.fit(X_train, Y_train)
        clf_adaboost = clf_adaboost.fit(X_train, Y_train)

        fpr_svm, tpr_svm, _ = roc_curve(Y_test, clf_svm.predict_proba(X_test)[:, 1], pos_label=1)
        tpr_svms.append(interp(mean_fpr, fpr_svm, tpr_svm))
        tpr_svms[-1][0] = 0.0

        fpr_rf, tpr_rf, _ = roc_curve(Y_test, clf_randomforest.predict_proba(X_test)[:, 1], pos_label=1)
        tpr_rfs.append(interp(mean_fpr, fpr_rf, tpr_rf))
        tpr_rfs[-1][0] = 0.0

        fpr_ada, tpr_ada, _ = roc_curve(Y_test, clf_adaboost.predict_proba(X_test)[:, 1], pos_label=1)
        tpr_adas.append(interp(mean_fpr, fpr_ada, tpr_ada))
        tpr_adas[-1][0] = 0.0

    mean_tpr_svm = np.mean(tpr_svms, axis=0)
    mean_tpr_svm[-1] = 1.0
    mean_tpr_rf = np.mean(tpr_rfs, axis=0)
    mean_tpr_rf[-1] = 1.0
    mean_tpr_ada = np.mean(tpr_adas, axis=0)
    mean_tpr_ada[-1] = 1.0
    return auc(mean_fpr, mean_tpr_svm), auc(mean_fpr, mean_tpr_rf), auc(mean_fpr, mean_tpr_ada)

#################################################################################################################
#  function MultiAuc:
#  Average AUC for three classification with all features over multiple iterations
# 
#  Input:  X_IDS, Y_IDS
#################################################################################################################
def MultiAuc(iterate, X_IDS, Y_IDS):
    auc_1 = [[] for i in range(iterate)]
    auc_2 = [[] for i in range(iterate)]
    auc_3 = [[] for i in range(iterate)]

    for i in range(iterate):
        auc_1[i], auc_2[i],auc_3[i] = calculate_auc(X_IDS[i], Y_IDS[i])
    print("Average AUC with all features: ")  
    print("SVM\t\t", np.mean(auc_1, axis = 0))
    print("Random forest\t",np.mean(auc_2, axis=0))
    print("AdaBoost\t",np.mean(auc_3, axis=0))
    
#################################################################################################################
#  function Classify:
#  Classify the reference and targeting set with three algorithms, SVM, random forest, AdaBoost 
# 
#  Input:  X, Y, roc_name
#################################################################################################################
def Classify(X, Y, roc_name):
    clf_svm = svm.SVC(kernel='linear', probability=True, random_state=0)
    clf_randomforest = RandomForestClassifier()  # max_depth=5, n_estimators=10, max_features=1
    clf_adaboost = AdaBoostClassifier()

    X = np.array(X)
    Y = np.array(Y)
    indices = [i for i in range(len(Y))]
    random.shuffle(indices)

    mean_fpr = np.linspace(0, 1, 100)
    tpr_svms = []
    tpr_rfs = []
    tpr_adas = []

    plt.figure(figsize=(10, 7))
    lw = 2
    for i in range(10):
        test_i = indices[int(i * len(Y) / 10):int((i + 1) * len(Y) / 10)]
        train_i = indices[:int(i * len(Y) / 10)] + indices[int((i + 1) * len(Y) / 10):]
        X_train, X_test, Y_train, Y_test = X[train_i], X[test_i], Y[train_i], Y[test_i]

        clf_svm = clf_svm.fit(X_train, Y_train)
        clf_randomforest = clf_randomforest.fit(X_train, Y_train)
        clf_adaboost = clf_adaboost.fit(X_train, Y_train)

        fpr_svm, tpr_svm, _ = roc_curve(Y_test, clf_svm.predict_proba(X_test)[:, 1], pos_label=1)
        tpr_svms.append(interp(mean_fpr, fpr_svm, tpr_svm))
        tpr_svms[-1][0] = 0.0

        fpr_rf, tpr_rf, _ = roc_curve(Y_test, clf_randomforest.predict_proba(X_test)[:, 1], pos_label=1)
        tpr_rfs.append(interp(mean_fpr, fpr_rf, tpr_rf))
        tpr_rfs[-1][0] = 0.0

        fpr_ada, tpr_ada, _ = roc_curve(Y_test, clf_adaboost.predict_proba(X_test)[:, 1], pos_label=1)
        tpr_adas.append(interp(mean_fpr, fpr_ada, tpr_ada))
        tpr_adas[-1][0] = 0.0

    mean_tpr_svm = np.mean(tpr_svms, axis=0)
    mean_tpr_svm[-1] = 1.0
    mean_tpr_rf = np.mean(tpr_rfs, axis=0)
    mean_tpr_rf[-1] = 1.0
    mean_tpr_ada = np.mean(tpr_adas, axis=0)
    mean_tpr_ada[-1] = 1.0

    plt.plot(mean_fpr, mean_tpr_svm, color='darkorange',
             lw=lw, alpha=1, label='SVM (AUC = %0.4f)' % auc(mean_fpr, mean_tpr_svm))
    plt.plot(mean_fpr, mean_tpr_rf, color='green',
             lw=lw, label='Random Forest (AUC = %0.4f)' % auc(mean_fpr, mean_tpr_rf))
    plt.plot(mean_fpr, mean_tpr_ada, color='darkred',
             lw=lw, label='AdaBoost (AUC = %0.4f)' % auc(mean_fpr, mean_tpr_ada))

    plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(roc_name)
    plt.legend(loc="lower right")
    plt.savefig('./results/'+SET_NAME +'_'+ roc_name + "_ROC.png")

#################################################################################################################
#  function ROCDrawing:
#  Draw ROC and report AUC for classification
# 
#  Input:  X_S, Y_S
#################################################################################################################
def ROCDrawing(X_S, Y_S, GermFeatureNames, CanonFeatureNames, PIFeatureNames, MotifFeatureNames, AllFeatureNames):
    # MMP features over 0.8 jaccard coefficient
    if SET_NAME == 'MMP-cluster':
        correlate_feature = ['Germ_HV_IGHV3-23*01', 'Canonical_L2_0', 'Canonical_L3_0', 'Canonical_H1_1', 'Canonical_H2_6']
    # elif SET_NAME == 'IGHV':
    # # # IGHV features over 0.8 jaccard coefficient
    #     correlate_feature = ['Germ_HV_IGHV3-23*01', 'Canonical_H1_1', 'Canonical_H2_6']
    else:
        correlate_feature = []
    X_S = np.array(X_S)

    X_S_new = []
    for j in range(len(X_S[0])):
        if AllFeatureNames[j].startswith('Germ_HV') or AllFeatureNames[j].startswith('Canonical_H2')or AllFeatureNames[j].startswith('Canonical_L2') \
                or AllFeatureNames[j].startswith('Canonical_L3') or AllFeatureNames[j].startswith('Canonical_H1'):
            continue
        X_S_new.append(X_S[:, j])
    X_S_new = np.array(X_S_new)
    X_S_new = X_S_new.T
    Classify(X_S_new, Y_S, 'All Features Included (Exclude Correlated)')


    Germ_E = len(GermFeatureNames)

    Canon_E = Germ_E + len(CanonFeatureNames)
    PI_E = Canon_E + len(PIFeatureNames)


    # Exclude the features correlated
    X_S_new = []
    for j in range(Germ_E):
        if AllFeatureNames[j].startswith('Germ_HV') or AllFeatureNames[j].startswith('Canonical_H2') or AllFeatureNames[
            j].startswith('Canonical_L2') \
                or AllFeatureNames[j].startswith('Canonical_L3') or AllFeatureNames[j].startswith('Canonical_H1'):
            continue
        X_S_new.append(X_S[:,j])
    X_S_new = np.array(X_S_new)
    X_S_new = X_S_new.T
    Classify(X_S_new, Y_S, 'Only Germline Features (Exclude Correlated)')

    X_S_new = []
    for j in range(Germ_E, Canon_E):
        if AllFeatureNames[j].startswith('Germ_HV') or AllFeatureNames[j].startswith('Canonical_H2') or AllFeatureNames[
            j].startswith('Canonical_L2') \
                or AllFeatureNames[j].startswith('Canonical_L3') or AllFeatureNames[j].startswith('Canonical_H1'):
            continue
        X_S_new.append(X_S[:, j])
    X_S_new = np.array(X_S_new)
    X_S_new = X_S_new.T
    Classify(X_S_new, Y_S, 'Only CDR Canonical Structure Features (Exclude Correlated)')

    X_S_new = []
    for j in range(Germ_E, X_S.shape[1]):
        if AllFeatureNames[j].startswith('Germ_HV') or AllFeatureNames[j].startswith('Canonical_H2') or AllFeatureNames[
            j].startswith('Canonical_L2') \
                or AllFeatureNames[j].startswith('Canonical_L3') or AllFeatureNames[j].startswith('Canonical_H1'):
            continue
        X_S_new.append(X_S[:, j])
    X_S_new = np.array(X_S_new)
    X_S_new = X_S_new.T
    Classify(X_S_new, Y_S, 'Except Germline Features (Exclude Correlated)')

    X_S_new = []
    for j in range(Germ_E):
        if AllFeatureNames[j].startswith('Germ_HV') or AllFeatureNames[j].startswith('Canonical_H2') or AllFeatureNames[
            j].startswith('Canonical_L2') \
                or AllFeatureNames[j].startswith('Canonical_L3') or AllFeatureNames[j].startswith('Canonical_H1'):
            continue
        X_S_new.append(X_S[:, j])
    for j in range(Canon_E, X_S.shape[1]):
        if AllFeatureNames[j].startswith('Germ_HV') or AllFeatureNames[j].startswith('Canonical_H2') or AllFeatureNames[
            j].startswith('Canonical_L2') \
                or AllFeatureNames[j].startswith('Canonical_L3') or AllFeatureNames[j].startswith('Canonical_H1'):
            continue
        X_S_new.append(X_S[:, j])
    X_S_new = np.array(X_S_new)
    X_S_new = X_S_new.T
    Classify(np.concatenate((X_S[:, :Germ_E], X_S[:, Canon_E:]), axis=1), Y_S,
             'Except CDR Canonical Structure Features (Exclude Correlated)')

    X_S_new = []
    for j in range(Canon_E):
        if AllFeatureNames[j].startswith('Germ_HV') or AllFeatureNames[j].startswith('Canonical_H2') or AllFeatureNames[
            j].startswith('Canonical_L2') \
                or AllFeatureNames[j].startswith('Canonical_L3') or AllFeatureNames[j].startswith('Canonical_H1'):
            continue
        X_S_new.append(X_S[:, j])
    for j in range(PI_E, X_S.shape[1]):
        if AllFeatureNames[j].startswith('Germ_HV') or AllFeatureNames[j].startswith('Canonical_H2') or AllFeatureNames[
            j].startswith('Canonical_L2') \
                or AllFeatureNames[j].startswith('Canonical_L3') or AllFeatureNames[j].startswith('Canonical_H1'):
            continue
        X_S_new.append(X_S[:, j])
    X_S_new = np.array(X_S_new)
    X_S_new = X_S_new.T
    Classify(np.concatenate((X_S[:, :Canon_E], X_S[:, PI_E:]), axis=1), Y_S, 'Except pI Features (Exclude Correlated)')

    X_S_new = []
    for j in range(PI_E):
        if AllFeatureNames[j].startswith('Germ_HV') or AllFeatureNames[j].startswith('Canonical_H2') or AllFeatureNames[
            j].startswith('Canonical_L2') \
                or AllFeatureNames[j].startswith('Canonical_L3') or AllFeatureNames[j].startswith('Canonical_H1'):
            continue
        X_S_new.append(X_S[:, j])
    X_S_new = np.array(X_S_new)
    X_S_new = X_S_new.T
    Classify(X_S_new, Y_S, 'Except Frequent Positional Motif Features (Exclude Correlated)')

    # Classify(X_S[:,:Germ_E], Y_S, 'Only Germline Features')
    # Classify(X_S[:,Germ_E:Canon_E], Y_S, 'Only CDR Canonical Structure Features')
    # Classify(X_S[:,Canon_E:PI_E], Y_S, 'Only pI Features')
    # Classify(X_S[:,PI_E:], Y_S, 'Only Frequent Positional Motif Features')
    #
    # Classify(X_S[:,Germ_E:], Y_S, 'Except Germline Features')
    # Classify(np.concatenate((X_S[:,:Germ_E],X_S[:,Canon_E:]),axis=1) , Y_S, 'Except CDR Canonical Structure Features')
    # Classify(np.concatenate((X_S[:,:Canon_E],X_S[:,PI_E:]),axis=1), Y_S, 'Except pI Features')
    # Classify(X_S[:,:PI_E], Y_S, 'Except Frequent Positional Motif Features')

def JaccardCoefficientAnalysis(AllFeatureVectors, AllFeatureNames, DatasetSize):
    if SET_NAME=='MMP-cluster' :
        PDB_size = DatasetSize[0] + DatasetSize[1]
    elif SET_NAME=='IGHV':
        PDB_size = DatasetSize[0]

    jac_sim_PDB = np.eye(len(AllFeatureNames))
    for i in range(len(AllFeatureNames)):
        for j in range(i + 1, len(AllFeatureNames)):
            if AllFeatureNames[i].startswith('Motif') or AllFeatureNames[j].startswith('Motif'):
                continue
            a = AllFeatureVectors[:PDB_size, i]
            b = AllFeatureVectors[:PDB_size, j]
            aandb = 0
            aorb = 0
            for k in range(len(a)):
                if a[k] == b[k] and a[k] == 1:
                    aandb += 1
                if a[k] == 1 or b[k] == 1:
                    aorb += 1
            if aorb == 0:
                jac_tmp = 0
            else:
                jac_tmp = float(aandb) / aorb
            # if AllFeatureNames[i] in interest_feature and AllFeatureNames[j] in interest_feature:
            #     print(AllFeatureNames[i], AllFeatureNames[j], jac_tmp)
            jac_sim_PDB[i][j] = jac_tmp
            jac_sim_PDB[j][i] = jac_tmp

    jac_sim_MMP = np.eye(len(AllFeatureNames))
    for i in range(len(AllFeatureNames)):
        for j in range(i + 1, len(AllFeatureNames)):
            if AllFeatureNames[i].startswith('Motif') or AllFeatureNames[j].startswith('Motif'):
                continue
            a = AllFeatureVectors[PDB_size:, i]
            b = AllFeatureVectors[PDB_size:, j]

            aandb = 0
            aorb = 0
            for k in range(len(a)):
                if a[k] == b[k] and a[k] == 1:
                    aandb += 1
                if a[k] == 1 or b[k] == 1:
                    aorb += 1
            if aorb == 0:
                jac_tmp = 0
            else:
                jac_tmp = float(aandb) / aorb
            # if AllFeatureNames[i] in interest_feature and AllFeatureNames[j] in interest_feature:
            #     print(AllFeatureNames[i], AllFeatureNames[j], jac_tmp)

            jac_sim_MMP[i][j] = jac_tmp
            jac_sim_MMP[j][i] = jac_tmp

    with open('./results/' + SET_NAME + '_Jaccard Feature Coefficient.csv', 'w') as fi:
        fi.write(
            'Feature value 1, Feature value 2, Jaccard coefficient for reference set, Jaccard coefficient for MMP-targeting set\n')
        for i in range(len(AllFeatureNames)):
            for j in range(i + 1, len(AllFeatureNames)):
                if AllFeatureNames[i].startswith('Motif') or AllFeatureNames[j].startswith('Motif'):
                    continue
                fi.write(AllFeatureNames[i] + ',' + AllFeatureNames[j] + ',' + str(jac_sim_PDB[i][j]) + ',' + str(
                    jac_sim_MMP[i][j]) + '\n')
