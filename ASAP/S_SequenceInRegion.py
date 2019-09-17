import Bio.SeqUtils.ProtParam
import os
import ASAP.FeatureExtraction as extract
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# Chothia numbering definition for CDR regions
CHOTHIA_CDR = {'L': {'1': [24, 34], '2': [50, 56], '3': [89, 97]}, 'H':{'1': [26, 32], '2': [52, 56], '3': [95, 102]}}
canonical_direct = '../data/pigs_canonical.txt'

SET_NAME = 'IGHV'
IF_ONLY_HEAVY = True
CNT_DB = 1
CNT_TARGET = 1
REFERENCE_PATH_TESTCASE = '../testCase/IGHV/reference-IGHV/'
TARGETING_PATH_TESTCASE = '../testCase/IGHV/targeting-MMP-IGHV/'
TARGET_DESIRE_SIZE = 134 #44  #IGHV

targeting_direct = TARGETING_PATH_TESTCASE
reference_direct = REFERENCE_PATH_TESTCASE

Amino, Num, Germ, DatasetName, DatasetSize = extract.ReadAminoNumGerm(targeting_direct, reference_direct)

seq_id = []
for i, name in enumerate(DatasetName):
    # if i<2:
    #     continue
    tmp= [[] for j in range(int(DatasetSize[i]))]
    # for every seq in that dataset
    for j in range(int(DatasetSize[i])):
        seq_name = name + '_' + str(j)
        seq_id.append(seq_name)

# raw sequence
def sequence_raw():
    def getSequenceHL(sname):
        SH = ''.join(Amino['H'][sname])
        SL = ''
        if not IF_ONLY_HEAVY:
            SL = ''.join(Amino['L'][sname])
            return SL, SH
        else:
            return [SH]

    with open('../results/'+SET_NAME +'_Sequence.csv','w') as fi:
        fi.write('sequence name, ')
        if not IF_ONLY_HEAVY:
            fi.write('light chain, ')
        fi.write('heavy chain\n')
        for sname in seq_id:
            fi.write(sname + ',' + ','.join(getSequenceHL(sname))+ '\n')

# sequence with numbering
def sequence_num():
    def getSequenceHL_num(sname):
        NH = ','.join(Num['H'][sname])
        SH = ','.join(Amino['H'][sname])
        NL = ','.join(Num['L'][sname])
        SL = ','.join(Amino['L'][sname])
        return NH, SH, NL, SL

    with open('./Sequence_numbered.csv','w') as fi:
        for sname in seq_id:
            NH, SH, NL, SL = getSequenceHL_num(sname)
            fi.write(sname + ' light num,' + NL + '\n')
            fi.write(sname + ' light seq,' + SL + '\n')
            fi.write(sname + ' heavy num,' + NH + '\n')
            fi.write(sname + ' heavy seq,' + SH + '\n')

# sequence with region
def sequence_region():
    def getSequenceHL_region(sname):
        NH = Num['H'][sname]

        HFW1, HCDR1, HFW2, HCDR2, HFW3, HCDR3, HFW4 = '', '', '', '', '', '', ''

        for i, number in enumerate(NH):
            if number[-1] >= 'A' and number[-1] <= 'Z':
                num_i = int(number[:-1])
            else:
                num_i = int(number)
            if num_i < CHOTHIA_CDR['H']['1'][0]:
                HFW1 += Amino['H'][sname][i]
            elif num_i <= CHOTHIA_CDR['H']['1'][1]:
                HCDR1+= Amino['H'][sname][i]
            elif num_i < CHOTHIA_CDR['H']['2'][0]:
                HFW2 += Amino['H'][sname][i]
            elif num_i <= CHOTHIA_CDR['H']['2'][1]:
                HCDR2 += Amino['H'][sname][i]
            elif num_i < CHOTHIA_CDR['H']['3'][0]:
                HFW3 += Amino['H'][sname][i]
            elif num_i <= CHOTHIA_CDR['H']['3'][1]:
                HCDR3 += Amino['H'][sname][i]
            else:
                HFW4 += Amino['H'][sname][i]
        if IF_ONLY_HEAVY:
            return ''.join(HFW1), ''.join(HCDR1), ''.join(HFW2), ''.join(HCDR2), ''.join(HFW3), ''.join(HCDR3), ''.join(
                HFW4)
        else:
            NL = Num['L'][sname]
            LFW1, LCDR1, LFW2, LCDR2, LFW3, LCDR3, LFW4 = '', '', '', '', '', '', ''
            for i, number in enumerate(NL):
                if number[-1] >= 'A' and number[-1] <= 'Z':
                    num_i = int(number[:-1])
                else:
                    num_i = int(number)
                if num_i < CHOTHIA_CDR['L']['1'][0]:
                    LFW1 += Amino['L'][sname][i]
                elif num_i <= CHOTHIA_CDR['L']['1'][1]:
                    LCDR1 += Amino['L'][sname][i]
                elif num_i < CHOTHIA_CDR['L']['2'][0]:
                    LFW2 += Amino['L'][sname][i]
                elif num_i <= CHOTHIA_CDR['L']['2'][1]:
                    LCDR2 += Amino['L'][sname][i]
                elif num_i < CHOTHIA_CDR['L']['3'][0]:
                    LFW3 += Amino['L'][sname][i]
                elif num_i <= CHOTHIA_CDR['L']['3'][1]:
                    LCDR3 += Amino['L'][sname][i]
                else:
                    LFW4 += Amino['L'][sname][i]
            return ''.join(LFW1), ''.join(LCDR1), ''.join(LFW2), ''.join(LCDR2), ''.join(LFW3), ''.join(LCDR3), ''.join(LFW4),\
                   ''.join(HFW1), ''.join(HCDR1), ''.join(HFW2), ''.join(HCDR2), ''.join(HFW3), ''.join(HCDR3), ''.join(HFW4)

    with open('../results/'+SET_NAME +'_Sequence_region.csv','w') as fi:
        if IF_ONLY_HEAVY:
            fi.write(
                'sequence id, heavy chain FW1, heavy chain CDR1, heavy chain FW2, heavy chain CDR2, heavy chain FW3, heavy chain CDR3, heavy chain FW4\n')

        else:
            fi.write('sequence id, light chain FW1, light chain CDR1, light chain FW2, light chain CDR2, light chain FW3, light chain CDR3, light chain FW4, '+
                                'heavy chain FW1, heavy chain CDR1, heavy chain FW2, heavy chain CDR2, heavy chain FW3, heavy chain CDR3, heavy chain FW4\n')
        for sname in seq_id:
            fi.write(sname + ',' + ','.join(getSequenceHL_region(sname)) + '\n')


def feature_distribution():
    from collections import Counter
    write_out = [[] for i in range(len(seq_id))]
    for fi in range(1,12):
        feat = []
        for item in write_out:
            feat.append(item[fi])

        feat_count = Counter(feat)
        sorted_count = sorted(feat_count.items(), key=lambda kv: kv[1], reverse=True)
        if fi==11:
            feat_type = sorted_count[0][0].split('_')[0]
        else:
            feat_type = sorted_count[0][0].split('_')[0] + sorted_count[0][0].split('_')[1]
        with open('./Features_distribution_'+feat_type+'.csv','w') as fi:
            for i in range(len(sorted_count)):
                fi.write(sorted_count[i][0]+','+str(sorted_count[i][1])+'\n')

def feature():
    write_out = [[] for i in range(len(seq_id))]
    for i in range(len(seq_id)):
        write_out[i].append(seq_id[i])
        for idx, f in enumerate(AllFeatureVectors[i]):
            if f == 1:
                write_out[i].append(AllFeatureNames[idx])

    with open('../results/'+SET_NAME +'_Features.csv', 'w') as fi:

        fi.write('sequence id, ')
        if not IF_ONLY_HEAVY:
            fi.write('light chain V region, light chain J region, ')
        fi.write('heavy chain V region, heavy chain J region, ')
        if not IF_ONLY_HEAVY:
            fi.write('Canonical L1, Canonical L2, Canonical L3, ')
        fi.write('Canonical H1, Canonical H2, Canonical H3, ' )
        fi.write('PI, frequent positional motif\n')
        for i in range(len(write_out)):
            fi.write(','.join(write_out[i]) + '\n')


def correlation_feature():

    ###### plot correlation matrix
    data = pd.DataFrame(AllFeatureVectors, columns=AllFeatureNames)
    # print(AllFeatureVectors.shape)
    corr = data.corr()
    import numpy as np
    corr = np.array(corr)
    with open('../results/Pearson_feature_correlation.csv', 'w') as fi:
        fi.write('Feature value 1, Feature value 2, Pearson coefficient\n')
        for i in range(len(AllFeatureNames)):
            for j in range(i+1, len(AllFeatureNames)):
                # if str(corr[i][j])=='nan':
                #     print('nan', AllFeatureNames[i], AllFeatureNames[j])
                fi.write(AllFeatureNames[i]+ ','+AllFeatureNames[j]+','+ str(corr[i][j])+'\n')



    # data.to_csv(r'../results/Feature_test.csv', header=True)

    # fig = plt.figure(figsize=(100, 70))
    # ax = fig.add_subplot(111)
    # cax = ax.matshow(corr, cmap='seismic', vmin=-1, vmax =1)
    # fig.colorbar(cax)
    # ticks = np.arange(0, len(data.columns),1)
    # ax.set_xticks(ticks)
    # plt.xticks(rotation=90)
    # ax.set_yticks(ticks)
    # ax.set_xticklabels(data.columns)
    # ax.set_yticklabels(data.columns)
    # plt.savefig('../results/feature_correlation.png')
    # corr = pd.DataFrame(corr, index=AllFeatureNames, columns=AllFeatureNames)
    ###### display pairwise correlation value
    # au_corr = corr.where(np.triu(np.ones(corr.shape), k=1).astype(np.bool))
    # au_corr = au_corr.stack().sort_values(ascending=False)
    # au_corr = corr.unstack()
    # au_corr.columns = [' 1', 'Feature 2', 'Pearson Correlation Value']
    # au_corr = pd.DataFrame(au_corr.values, columns = ['Feature 1, Feature 2, Pearson Correlation Value'])
    # au_corr.to_csv(r'../results/Pearson_feature_correlation.csv', header=True)
    # print(len(au_corr))

    # print(AllFeatureVectors[:, AllFeatureNames.index('Germ_LJ_IGKJ3*01')])
    # print(AllFeatureVectors[:, AllFeatureNames.index('Canonical_L2_0')])

# def JaccardCoefficientAnalysis():
#     df = pd.DataFrame(AllFeatureVectors, columns=AllFeatureNames)
#
#     interest_feature=['Germ_HV_IGHV3-23*01', 'Canonical_H2_6', 'Germ_HJ_IGHJ4*02', 'Germ_HJ_IGHJ6*01', 'Germ_LV_IGKV1D-39*01',
#                       'Canonical_H2_5', 'Germ_HJ_IGHJ4*01']
#     jac_sim = np.eye(len(AllFeatureNames))
#     for i in range(len(AllFeatureNames)):
#         for j in range(i+1, len(AllFeatureNames)):
#             if AllFeatureNames[i].startswith('Motif') or AllFeatureNames[j].startswith('Motif'):
#                 continue
#             a = AllFeatureVectors[:, i]
#             b = AllFeatureVectors[:, j]
#             aandb =0
#             aorb = 0
#             for k in range(len(a)):
#                 if a[k]==b[k] and a[k]==1:
#                     aandb +=1
#                 if a[k]==1 or b[k]==1:
#                     aorb +=1
#             if aorb==0:
#                 jac_tmp=0
#             else:
#                 jac_tmp = float(aandb)/aorb
#             if AllFeatureNames[i] in interest_feature and AllFeatureNames[j] in interest_feature:
#                 print(AllFeatureNames[i], AllFeatureNames[j], jac_tmp)
#
#             jac_sim[i][j]=jac_tmp
#             jac_sim[j][i]=jac_tmp
#
#
#     with open('../results/Jaccard_feature_coefficient.csv', 'w') as fi:
#         fi.write('Feature value 1, Feature value 2, Jaccard coefficient\n')
#         for i in range(len(AllFeatureNames)):
#             for j in range(i+1, len(AllFeatureNames)):
#                 if AllFeatureNames[i].startswith('Motif') or AllFeatureNames[j].startswith('Motif'):
#                     continue
#                 fi.write(AllFeatureNames[i]+ ','+AllFeatureNames[j]+','+ str(jac_sim[i][j])+'\n')
#
#
#     fig = plt.figure(figsize=(100, 70))
#     ax = fig.add_subplot(111)
#     cax = ax.matshow(jac_sim, cmap='Blues', vmin=0, vmax =1)
#     fig.colorbar(cax)
#     ticks = np.arange(0, len(df.columns),1)
#     ax.set_xticks(ticks)
#     plt.xticks(rotation=90)
#     ax.set_yticks(ticks)
#     ax.set_xticklabels(df.columns)
#     ax.set_yticklabels(df.columns)
#     plt.savefig('../results/feature_coefficient.png')
#
#     # print(AllFeatureVectors[:,AllFeatureNames.index('Germ_LJ_IGKJ3*01')])
#     # print(AllFeatureVectors[:,AllFeatureNames.index('Canonical_L2_0*01')])
#     # where(np.triu(np.ones(jac_sim.shape), k=1).astype(np.bool))
#     # au_jac = jac_sim.where(np.triu(np.ones(jac_sim.shape), k=0).astype(np.bool))
#     # au_jac = au_jac.stack().sort_values(ascending=False)
#     # au_jac = jac_sim.unstack()
#     # print(len(au_jac))
#     # au_jac.to_csv(r'../results/Jaccard_feature_coefficient.csv', header=True)

def JaccardCoefficientAnalysis():

    PDB_size = DatasetSize[0]

    jac_sim_PDB = np.eye(len(AllFeatureNames))
    for i in range(len(AllFeatureNames)):
        for j in range(i+1, len(AllFeatureNames)):
            if AllFeatureNames[i].startswith('Motif') or AllFeatureNames[j].startswith('Motif'):
                continue
            a = AllFeatureVectors[:PDB_size, i]
            b = AllFeatureVectors[:PDB_size, j]
            aandb =0
            aorb = 0
            for k in range(len(a)):
                if a[k]==b[k] and a[k]==1:
                    aandb +=1
                if a[k]==1 or b[k]==1:
                    aorb +=1
            if aorb==0:
                jac_tmp=0
            else:
                jac_tmp = float(aandb)/aorb

            # if AllFeatureNames[i] == 'Germ_HV_IGHV3-23*01' and AllFeatureNames[j] =='Canonical_H2_6':
            #     print(a, b, jac_tmp)
            # if AllFeatureNames[i] in interest_feature and AllFeatureNames[j] in interest_feature:
            #     print(AllFeatureNames[i], AllFeatureNames[j], jac_tmp)
            jac_sim_PDB[i][j]=jac_tmp
            jac_sim_PDB[j][i]=jac_tmp

    jac_sim_MMP = np.eye(len(AllFeatureNames))
    for i in range(len(AllFeatureNames)):
        for j in range(i+1, len(AllFeatureNames)):
            if AllFeatureNames[i].startswith('Motif') or AllFeatureNames[j].startswith('Motif'):
                continue
            a = AllFeatureVectors[PDB_size:, i]
            b = AllFeatureVectors[PDB_size:, j]

            aandb =0
            aorb = 0
            for k in range(len(a)):
                if a[k]==b[k] and a[k]==1:
                    aandb +=1
                if a[k]==1 or b[k]==1:
                    aorb +=1
            if aorb==0:
                jac_tmp=0
            else:
                jac_tmp = float(aandb)/aorb
            # if AllFeatureNames[i] in interest_feature and AllFeatureNames[j] in interest_feature:
            #     print(AllFeatureNames[i], AllFeatureNames[j], jac_tmp)

            jac_sim_MMP[i][j]=jac_tmp
            jac_sim_MMP[j][i]=jac_tmp


    with open('../results/'+SET_NAME+'_Jaccard Feature Coefficient.csv', 'w') as fi:
        fi.write('Feature value 1, Feature value 2, Jaccard coefficient for reference set, Jaccard coefficient for MMP-targeting set\n')
        for i in range(len(AllFeatureNames)):
            for j in range(i+1, len(AllFeatureNames)):
                if AllFeatureNames[i].startswith('Motif') or AllFeatureNames[j].startswith('Motif'):
                    continue
                fi.write(AllFeatureNames[i]+ ','+AllFeatureNames[j]+','+ str(jac_sim_PDB[i][j])+','+ str(jac_sim_MMP[i][j])+'\n')
if __name__=='__main__':
    sequence_raw()
    sequence_region()
    OneHotGerm, GermFeatureNames = extract.GetOneHotGerm(Germ, DatasetSize, DatasetName)
    OneHotCanon, CanonFeatureNames = extract.GetOneHotCanon(canonical_direct, Amino, Num, DatasetSize, DatasetName)
    CDRH3 = extract.GetCDRH3(Amino, Num)
    OneHotPI, PIFeatureNames = extract.GetOneHotPI(CDRH3, DatasetSize, DatasetName)
    MultiHotMotif, MotifFeatureNames = extract.MultiHotMotif(CDRH3, DatasetSize, DatasetName)
    AllFeatureVectors, AllFeatureNames, _, _ = extract.GetFeatureVectors(OneHotGerm, GermFeatureNames, OneHotCanon, CanonFeatureNames, OneHotPI, PIFeatureNames, MultiHotMotif, MotifFeatureNames)

    feature()
    # correlation_feature()
    JaccardCoefficientAnalysis()







