import Bio.SeqUtils.ProtParam
import os
import numpy as np

SET_NAME = 'MMP-cluster'
IF_ONLY_HEAVY = False
CNT_DB = 2
CNT_TARGET = 1
REFERENCE_PATH_TESTCASE = './testCase/MMP-cluster/reference-PDB/'
TARGETING_PATH_TESTCASE = './testCase/MMP-cluster/targeting-MMP/'
TARGET_DESIRE_SIZE = 166 #44 #MMP-cluster




# Chothia numbering definition for CDR regions
CHOTHIA_CDR = {'L': {'1': [24, 34], '2': [50, 56], '3': [89, 97]}, 'H':{'1': [26, 32], '2': [52, 56], '3': [95, 102]}}

#################################################################################################################
#  function ReadAminoAndNum:
#  Read in the Chothia number reference and targeting files. Store the numbering and putative germline.
#
#  Input: targeting_direct, reference_direct
#  Output:1. dictionary of Amino, {'L': {}, 'H': {}}
#         2. dictionary of Num  , {'L': {}, 'H': {}}
#         3. dictionary of Germ , {'L': {'V': {}, 'J':{}}, 'H': {'V': {}, 'J':{}}}
#         4. list of DatasetName, [dh, dm, p1,....] 
#         5. list of DatasetSize, [ , , ,...]
#################################################################################################################

def ReadAminoNumGerm(targeting_direct, reference_direct):
    Amino = {'L': {}, 'H': {}}
    Num ={'L': {}, 'H': {}}
    Germ =  {'L': {'V': {}, 'J':{}}, 'H': {'V': {}, 'J':{}}}
    DatasetName = []
    DatasetSize = []
    
    targeting_filenames = sorted(os.listdir(targeting_direct))
    reference_filenames = sorted(os.listdir(reference_direct))

    for i, name in enumerate(reference_filenames + targeting_filenames):
        if not name.endswith('.txt'):
            continue
        if i < len(reference_filenames):
            direct = reference_direct
        else:
            direct = targeting_direct
        with open(direct + name, 'r') as fi:
            data = fi.readlines()
        DatasetName.append(name.split('_')[0])
        cnt_pattern = 0
        cnt_seq =  0
        tmp_num = []
        tmp_seq = []
        tmp_germ_V = ' '
        tmp_germ_J = ' '
        buff = ''
        for j in range(len(data)):
            # if chain begin
            if data[j][0] =='L' or data[j][0] =='H':
                L_H = data[j][0]
                tmp_seq.append(data[j].split()[-1])
                if len(data[j].split()) == 3:
                    tmp_num.append(data[j].split()[-2])
                else:
                    tmp_num.append(data[j].split()[1] + data[j].split()[-2])

            # second time of #|, line of germline
            if data[j][0]=='#' and data[j][1] == '|':
                cnt_pattern += 1
                if (cnt_pattern % 4) == 0:
                    tmp_germ_V = data[j].split("|")[2]
                    tmp_germ_J = data[j].split("|")[4]


            # time of \\, ending a sequence, need \\ to present \
            if data[j][0] == '/':
                if IF_ONLY_HEAVY:
                    seq_name = name.split('_')[0] + '_' + str(cnt_seq)
                else:
                    seq_name = name.split('_')[0] + '_' + str(int(cnt_seq / 2))
                cnt_seq += 1
                Amino[L_H][seq_name] = tmp_seq
                Num[L_H][seq_name] =tmp_num
                Germ[L_H]['V'][seq_name] = tmp_germ_V
                Germ[L_H]['J'][seq_name] = tmp_germ_J
                # if not tmp_germ_V.startswith('IGHV3-23'):
                #     print(data[j - 8])
                #     print(seq_name)
                #     print(tmp_germ_V, tmp_germ_J)
                tmp_num = []
                tmp_seq = []
                tmp_germ_V = ' '
                tmp_germ_J = ' '

        if IF_ONLY_HEAVY:
            DatasetSize.append(cnt_seq)
        else:
            DatasetSize.append(int(cnt_seq / 2))
    return Amino, Num, Germ, DatasetName, DatasetSize


#################################################################################################################
#  function GetOneHotGerm:
#  Transform the stored putative germline into one-hot encoded features.
#
#  Input:  Germ, DatasetSize, DatasetName
#  Output: 1. array of OneHotGerm, [[seq1 onehot], [seq2 onehot], [seq3 onehot], ...]
#          2. list of GermFeatureNames according to one hot, [LV_IGLV1*1, LV_IGLV1*2,....
#                                                             LJ_XXXX, 
#                                                             HV_XXXX,
#                                                             HJ_XXXX ...]
#################################################################################################################

def GetOneHotGerm(Germ, DatasetSize, DatasetName):
    OneHotGerm = []
    GermFeatureNames = []
    # for every feature type
    for H_L in Germ:
        if IF_ONLY_HEAVY:
            if H_L=='L':
                continue
        for V_J in Germ[H_L]:
            # every feature name in that type
            candidate = list(sorted(set(Germ[H_L][V_J].values())))
            for can in candidate:
                GermFeatureNames.append('Germ_' +H_L+ V_J+'_'+can)

    # for every dataset
    for i, name in enumerate(DatasetName):
        tmp = [[] for j in range(int(DatasetSize[i]))]
        # for every seq in that dataset
        for j in range(int(DatasetSize[i])):
            seq_name = name + '_' + str(j)

            for k in range(len(GermFeatureNames)):
                H_L = GermFeatureNames[k].split('_')[1][0]
                V_J = GermFeatureNames[k].split('_')[1][1]
                if Germ[H_L][V_J][seq_name] == GermFeatureNames[k].split('_')[2]:
                    tmp[j].append(1)
                else:
                    tmp[j].append(0)
        OneHotGerm += tmp

    return OneHotGerm, GermFeatureNames


#################################################################################################################
#  function ReadCanonTemp:
#  Read in the template file (default PIGS) and store it.
#  
#  Output: 1. dictionary of CanonTemp, {'L': {'1': {'1':[]}, '2': {'1':[]}, '3': {'1':[]}}, 'H': {'1': {'1':[]}, '2': {'1':[]}, '3': {'1':[]}}}
#################################################################################################################
def ReadCanonTemp(canonical_direct):
    CanonTemp = {'L': {'1': {'1':[]}, '2': {'1':[]}, '3': {'1':[]}}, 'H': {'1': {'1':[]}, '2': {'1':[]}, '3': {'1':[]}}}
    with open(canonical_direct, 'r') as fi:
        data = fi.readlines()
    for i in range(len(data)):
        if data[i].split()[1] not in CanonTemp[data[i][0]][data[i][1]]:
            CanonTemp[data[i][0]][data[i][1]][data[i].split()[1]] = []
        CanonTemp[data[i][0]][data[i][1]][data[i].split()[1]].append(data[i].split()[2:])
    return CanonTemp

#################################################################################################################
#  function GetCanon:
#  Assign each sequence witht the predicted type of canonical structure according to the template.
#  
#  Input:   Amino, Num
#  Output:  1. dictionary of CanonTemp, {'L': {'1': {'1':[]}, '2': {'1':[]}, '3': {'1':[]}}, 'H': {'1': {'1':[]}, '2': {'1':[]}, '3': {'1':[]}}}
#              optional: PIGS / Chothia
#################################################################################################################

def GetCanon(canonical_direct, Amino, Num):
    CanonTemp = ReadCanonTemp(canonical_direct)
    Canon = {'L': {'1': {}, '2': {}, '3': {}}, 'H': {'1': {}, '2': {}, '3': {}}}
    # for every sequence
    for seq_name in Num['H']:

        for L_H in Canon:
            if IF_ONLY_HEAVY:
                if L_H == 'L':
                    continue

            for j in Canon[L_H]:
                cnt_len = 0

                for k in Num[L_H][seq_name]:
                    if k[-1]>='A'and k[-1]<='Z':
                        num_i = int(k[:-1])
                    else:
                        num_i = int(k)
                    if num_i >= CHOTHIA_CDR[L_H][j][0] and num_i <= CHOTHIA_CDR[L_H][j][1]:
                        cnt_len += 1
                length = cnt_len
                # for every type number on specific CDR region
                for k in CanonTemp[L_H][j]:
                    ############## same type have diff version of template
                    for m in range(len(CanonTemp[L_H][j][k])):
                        # if have matched CDR length, then give zero type
                        if CanonTemp[L_H][j][k][m][0] == str(length):
                            # check if length is the only restriction
                            if len(CanonTemp[L_H][j][k][m]) == 1:
                                Canon[L_H][j][seq_name] = k
                            # check for each position with in specific motif
                            else:
                                restriction = CanonTemp[L_H][j][k][m][1:]
                                for l in range(0,len(restriction),2):

                                    pos = CanonTemp[L_H][j][k][m][l+1]

                                    # index of the number
                                    if pos not in Num[L_H][seq_name]:
                                        break
                                    else:
                                        id = int(Num[L_H][seq_name].index(pos))
                                        s=CanonTemp[L_H][j][k][m][l + 2]

                                        if Amino[L_H][seq_name][id] not in CanonTemp[L_H][j][k][m][l+2]:
                                            break
                                        Canon[L_H][j][seq_name] = k
                # if no match canonical structure found, then append 0
                if seq_name not in Canon[L_H][j]:
                    Canon[L_H][j][seq_name] = '0'
    return Canon

#################################################################################################################
#  function GetOneHotCanon:
#  Similar to GetOneHotGerm, transform the stored canonical structure into one-hot encoded features.
#
#  Input:  Amino, Num, DatasetSize, DatasetName
#  Output: 1. array of OneHotCanon, [[seq1 onehot], [seq2 onehot], [seq3 onehot], ...]
#          2. list of CanonFeatureNames according to one hot, [Canon_L1_1, Canon_L1_2,....
#                                                              Canon_L2_1, 
#                                                              Canon_L3_1,
#                                                              Canon_H1_1, 
#                                                              Canon_H2_1,
#                                                              Canon_H3_1,...]
#################################################################################################################

def GetOneHotCanon(canonical_direct, Amino, Num, DatasetSize, DatasetName):
    Canon = GetCanon(canonical_direct, Amino, Num)
    OneHotCanon = []
    CanonFeatureNames = []
    # for every feature type

    for H_L in Canon:
        if IF_ONLY_HEAVY:
            if H_L=='L':
                continue
        # O_T_T stands for 1_2_3
        for O_T_T in Canon[H_L]:
            # every feature name in that type
            candidate = list(sorted(set(Canon[H_L][O_T_T].values())))
            for can in candidate:
                CanonFeatureNames.append('Canonical_' +H_L+ O_T_T+'_'+can)
                
    # for every dataset
    for i, name in enumerate(DatasetName):
        tmp = [[] for j in range(int(DatasetSize[i]))]
        # for every seq in that dataset
        for j in range(int(DatasetSize[i])):
            seq_name = name + '_' + str(j)
            for k in range(len(CanonFeatureNames)):
                H_L = CanonFeatureNames[k].split('_')[1][0]
                O_T_T = CanonFeatureNames[k].split('_')[1][1]
                if Canon[H_L][O_T_T][seq_name] == CanonFeatureNames[k].split('_')[2]:
                    tmp[j].append(1)
                else:
                    tmp[j].append(0)
        OneHotCanon += tmp
        
    return OneHotCanon, CanonFeatureNames

#################################################################################################################
#  function GetCDRH3:
#  Take the CDR-H3 of each seqeunce.
# 
#  Input:   Amino, Num
#  Output: 1. dictionary of CDRH3, {}
#################################################################################################################

def GetCDRH3(Amino, Num):
    CDRH3={}
    for seq_name in Amino['H']:
        CDRH3[seq_name]=''
        for i in range(len(Num['H'][seq_name])):
            number = Num['H'][seq_name][i]
            if number[-1] >= 'A' and number[-1] <= 'Z':
                num_i = int(number[:-1])
            else:
                num_i = int(number)
            if num_i >= CHOTHIA_CDR['H']['3'][0] and num_i <= CHOTHIA_CDR['H']['3'][1]:
                CDRH3[seq_name] += Amino['H'][seq_name][i]
    return CDRH3

#################################################################################################################
#  function GetCDRH3PI:
#  Calculate the pI value for each sequence
# 
#  Input:   CDRH3
#  Output: 1. dictionary of PI, {}
#################################################################################################################

def GetCDRH3PI(CDRH3):
    void = ['KYPLAVSGIIT', '-------V', 'GVVTAAIDGMDV','DLYSGYRSYGLDV', 'GGTSYYGTDV','EEGDIPGTTCMDV']
    PI_CDRH3={}
    for seq_name in CDRH3:
        prot = Bio.SeqUtils.ProtParam.ProteinAnalysis(CDRH3[seq_name])
        try:
            PI_CDRH3[seq_name] = prot.isoelectric_point()
        except:
            PI_CDRH3[seq_name] = -1

    return PI_CDRH3


#################################################################################################################
#  function GetPIBin:
#  Halve the bin of pI following the binning method using sequence's pI information.
# 
#  Input:   PI_CDRH3
#  Output: 1. a list of PITheresholds, []
#################################################################################################################

def GetPIBin(PI_CDRH3):
    PITheresholds = [0.0, 7.0, 14.0]
    tenPercent = 0.1*len(PI_CDRH3)
    PITolerance = 0.3
    cnt = 0
    while cnt > tenPercent or len(PITheresholds) == 3:
        # count how many sequence over threshold
        for i in range(1, len(PITheresholds)):
            cnt = 0
            if (PITheresholds[i] - PITheresholds[i-1])< (2 * PITolerance):
                continue
            # go over the dict
            for seq in PI_CDRH3:
               if PI_CDRH3[seq]> PITheresholds[i-1] and PI_CDRH3[seq]<PITheresholds[i]:
                        cnt +=1

            #check if overflow tenpercent
            if cnt > tenPercent:
                PITheresholds.append((PITheresholds[i-1] + PITheresholds[i])/2.0)
                PITheresholds = sorted(PITheresholds)
                break
    return PITheresholds

#################################################################################################################
#  function GetOneHotPI:
#  Transform the pI values into one-hot encoded pI bin features.
# 
#  Input:   CDRH3, DatasetSize, DatasetName
#  Output: 1. array of OneHotPI, [[seq1 onehot], 
#                                 [seq2 onehot], 
#                                 [seq3 onehot],
#                                 ...]
#          2. list of PIFeatureNames according to one hot, [PI_bin1, PI_bin2, PI_bin3...]
#################################################################################################################

def GetOneHotPI(CDRH3, DatasetSize, DatasetName):

    PI_CDRH3 = GetCDRH3PI(CDRH3)

    PITheresholds = GetPIBin(PI_CDRH3)

    PIFeatureNames = []
    OneHotPI = []
    for i in range(1, len(PITheresholds)):
        PIFeatureNames.append('PI_'+str(PITheresholds[i-1])+'-'+str(PITheresholds[i]))

    # for every dataset
    for i, name in enumerate(DatasetName):
        tmp = [[0 for k in range(len(PIFeatureNames))] for j in range(int(DatasetSize[i]))]
        # for every seq in that dataset
        for j in range(int(DatasetSize[i])):
            seq_name = name + '_' + str(j)
            for k in range(1, len(PITheresholds)):
                if PI_CDRH3[seq_name] >= float(PITheresholds[k-1]) and PI_CDRH3[seq_name] <= float(PITheresholds[k]):
                    tmp[j][k-1] = 1
                    break
        OneHotPI += tmp
    return OneHotPI, PIFeatureNames

#################################################################################################################
#  function GetPositionalMotifFreq:
#  Count the frequency of each possible frequent possitional motif for each dataset.
# 
#  Input:   CDRH3
#  Output: 1. dictionary of MotifFreq, {'r1':{}, 'r2':{},'t1':{}, 't2':{}, 't3':{}, 't4':{}, 't5':{}, 't6':{}, 't7':{}, 't8':{}}
#################################################################################################################

def GetPositionalMotifFreq(CDRH3):
    MotifFreq ={'r1':{}, 'r2':{},'t1':{}, 't2':{}, 't3':{}, 't4':{}, 't5':{}, 't6':{}, 't7':{}, 't8':{}}
    MotifDict = {}
    for seq_name in CDRH3:
        MotifDict[seq_name] = []
        f_name = seq_name.split('_')[0]
        # length of motif
        for i in range(2, 10):
            if i > len(CDRH3[seq_name]):
                continue
            else:
                for j in range(len(CDRH3[seq_name])-i):
                    PostionalMotif = str(j) +'_'+CDRH3[seq_name][j:j+i]

                    MotifDict[seq_name].append(PostionalMotif)
                    if PostionalMotif in MotifFreq[f_name]:
                        MotifFreq[f_name][PostionalMotif] += 1
                    else:
                        MotifFreq[f_name][PostionalMotif] = 1
    return MotifFreq, MotifDict

#################################################################################################################
#  function GetImpMotif (Version 1.0):
#  Take only the most 2 frequent motif in each data set,  top 2 * 10 set * 9 length = 180 
# 
#  Input:   MotifFreq
#  Output: 1. list of ImpMotif, [motif1, motif2, ...]
#################################################################################################################

def GetImpMotif(MotifFreq):
    ImpMotif = []
    Top2 = 2
    for f_name in MotifFreq:
        motif_dic = MotifFreq[f_name]
        for i in range(2, 11):
            tmp = {}
            for motif in motif_dic:

                if motif.split('_')[0] == str(i):
                    tmp[motif]= motif_dic[motif]
            sorted_tmp = sorted(tmp.items(),key= lambda k: k[1],reverse= True)
            for j in range(Top2):
                if len(sorted_tmp)> j:
                    ImpMotif.append(sorted_tmp[j][0])
    ImpMotif = list(sorted(set(ImpMotif)))
    return ImpMotif

#################################################################################################################
#  function GetCDRH3Motif:
#  Assign present frequent motif for each sequence
# 
#  Input:   ImpMotif, CDRH3
#  Output: 1. dictionary of Motif_CDRH3, {}
#################################################################################################################

def GetCDRH3Motif(ImpMotif, CDRH3, MotifDict):
    Motif_CDRH3={}
    for seq_name in CDRH3:
        # seq_len = len(CDRH3[seq_name])
        Motif_CDRH3[seq_name]=[0 for z in range(len(ImpMotif))]
        for i in range(len(ImpMotif)):
            if ImpMotif[i] in MotifDict[seq_name]:
                Motif_CDRH3[seq_name][i] = 1
    return Motif_CDRH3

#################################################################################################################
#  function MultiHotMotif:
#  Transfer motif information for each sequence to multi-hot encoded features.
# 
#  Input:   CDRH3, DatasetSize, DatasetName
#  Output: 1. array of MultiHotMotif, [[seq1 multihot], [seq2 multihot], [seq3 multihot],...]
#          2. list of MotifFeatureNames according to multi hot, [Motif1, Motif2, ...]
#################################################################################################################

def MultiHotMotif(CDRH3, DatasetSize, DatasetName):
    MotifFreq, MotifDict = GetPositionalMotifFreq(CDRH3)

    ImpMotif = GetImpMotif(MotifFreq)

    Motif_CDRH3 = GetCDRH3Motif(ImpMotif, CDRH3, MotifDict)

    MotifFeatureNames = []
    for motif in ImpMotif:
        MotifFeatureNames.append("Motif_"+ motif)

    MultiHotMotif =[]
    for i, name in enumerate(DatasetName):
        tmp = [[] for j in range(int(DatasetSize[i]))]
        # for every seq in that dataset
        for j in range(int(DatasetSize[i])):
            seq_name = name + '_' + str(j)
            tmp[j]= Motif_CDRH3[seq_name]
        MultiHotMotif+=tmp
    return MultiHotMotif, MotifFeatureNames

#################################################################################################################
#  function GetFeatureVectors:
#  Combine germline, canonical structure, pI, motif features to feature vectors
# 
#  Input:   OneHotGerm, GermFeatureNames, OneHotCanon, CanonFeatureNames, OneHotPI, PIFeatureNames, MultiHotMotif, MotifFeatureNames
#  Output: 1. AllFeatureVectors for every sequence, [[seq1 LV, LJ, HV, HJ, L1, L2, L3, L1, L2, L3, pI, motif1, motif2, motifi...],
#                                                    [seq2 LV, LJ, HV, HJ, L1, L2, L3, L1, L2, L3, pI, motif1, motif2, motifi...],
#                                                    ...]
# 
#          2. AllFeatureNames        [LV, LJ, HV, HJ, L1, L2, L3, L1, L2, L3, pI, motif1, motif2, motifi...]
#################################################################################################################

def GetFeatureVectors(OneHotGerm, GermFeatureNames,
                      OneHotCanon, CanonFeatureNames,
                      OneHotPI, PIFeatureNames,
                      MultiHotMotif, MotifFeatureNames):
    AllFeatureNames= GermFeatureNames + CanonFeatureNames + PIFeatureNames + MotifFeatureNames
    AllFeatureVectors =[[] for i in range(len(OneHotGerm))]
    # num of seq
    for i in range(len(OneHotGerm)):
        AllFeatureVectors[i] += OneHotGerm[i]
        AllFeatureVectors[i] += OneHotCanon[i]
        AllFeatureVectors[i] += OneHotPI[i]
        AllFeatureVectors[i] += MultiHotMotif[i]


    AllFeatureVectors = np.array(AllFeatureVectors)
    ExcludeIGHVVectors = AllFeatureVectors
    ExcludeFeatureNames = AllFeatureNames
    if SET_NAME == 'IGHV':
        name_index = []
        ExcludeFeatureNames = []
        for i, name in enumerate(AllFeatureNames):
            if not name.startswith('Germ_HV_IGHV3-23'):
                name_index.append(i)
                ExcludeFeatureNames.append(AllFeatureNames[i])

        ExcludeIGHVVectors = AllFeatureVectors[:, name_index]

    return AllFeatureVectors, AllFeatureNames, ExcludeIGHVVectors, ExcludeFeatureNames

if __name__=='__main__':
    targeting_direct = '../testCase-MMP/data/IGHV/'
    reference_direct = '../testCase-MMP/data/IGHV/'
    Amino, Num, Germ, DatasetName, DatasetSize = ReadAminoNumGerm(targeting_direct, reference_direct)

