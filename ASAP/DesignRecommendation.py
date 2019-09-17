import numpy as np
import pydotplus
from sklearn import tree

SET_NAME = 'MMP-cluster'
IF_ONLY_HEAVY = False
CNT_DB = 2
CNT_TARGET = 1
REFERENCE_PATH_TESTCASE = './testCase/MMP-cluster/reference-PDB/'
TARGETING_PATH_TESTCASE = './testCase/MMP-cluster/targeting-MMP/'
TARGET_DESIRE_SIZE = 166 #44 #MMP-cluster



#################################################################################################################
#  function SanityFeature:
#  Omit non-recommending features, such as motif features and type 0 canonical structures, as feature value for decision tree
# 
#  Input:  AgreeFeature, AllFeatureNames
#  Output: 1. SanityAgreeFeature, [] a list of index according to the AllFeatureNames that remain to put in decision tree
#################################################################################################################

def SanityFeature(AgreeFeature, AllFeatureNames):
    SanityAgreeFeature=[]
    for idx in AgreeFeature:
        if not(AllFeatureNames[idx].split('_')[0] == 'Motif') and not(AllFeatureNames[idx].split('_')[0] == 'Canonical' and AllFeatureNames[idx].split('_')[2] == '0'):
            SanityAgreeFeature.append(idx)
    return SanityAgreeFeature

#################################################################################################################
#  function MultiDecisionTree:
#  Decision tree drawn with combined data across multiple iteration
# 
#  Input:   X_DS, Y_DS, FeatureN, type
#################################################################################################################
def MultiDecisionTree(iterate, X_IDS, Y_IDS, AllFeatureNames, type):
    Y = np.concatenate(Y_IDS, axis=0)
    AgreeFeature =[i for i in range(len(AllFeatureNames)) ]
    SanityAgreeFeature = SanityFeature(AgreeFeature, AllFeatureNames)

    SanityAgreeFeatureName = []
    for idx in SanityAgreeFeature:
        SanityAgreeFeatureName.append(AllFeatureNames[idx])
    
    Sig_X_DS =[[] for i in range(iterate)]
    for i in range(iterate):
        X_IDS[i]=np.array(X_IDS[i])
        Sig_X_DS[i] = X_IDS[i][:,SanityAgreeFeature]
        
    X =np.concatenate(Sig_X_DS, axis=0)

    minLeafSize = int(0.025 *len(Y))
    clf = tree.DecisionTreeClassifier(min_samples_leaf = minLeafSize)
    clf = clf.fit(np.ones((len(Y),len(X[0])))-X, Y) #flip the X for decision tree to meet the true false

    dot_data = tree.export_graphviz(clf, out_file=None, filled=True,feature_names=SanityAgreeFeatureName, class_names=['Reference', 'Targeting'], rounded=True)
    pydotplus.graph_from_dot_data(dot_data).write_png("./results/"+ SET_NAME + "_DTree"+ type +".png")

