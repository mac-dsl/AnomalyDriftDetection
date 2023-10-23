import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import stumpy

from util.TSB_AD.metrics import metricor
import matplotlib.patches as mpatches 

from scipy.signal import argrelextrema
from scipy.cluster.hierarchy import dendrogram, linkage, fcluster, distance

from statsmodels.tsa.stattools import acf

############################################################################
## Code from TSB-UAD
## https://github.com/TheDatumOrg/TSB-UAD
## slidingWindows.py
## "TSB-UAD: An End-to-End Benchmark Suite for Univariate Time-Series Anomaly Detection"
## John Paparrizos, Yuhao Kang, Paul Boniol, Ruey Tsay, Themis Palpanas, and Michael Franklin.
## Proceedings of the VLDB Endowment (PVLDB 2022) Journal, Volume 15, pages 1697–1711

def find_length(data):
    if len(data.shape)>1:
        return 0
    data = data[:min(20000, len(data))]
    
    base = 3
    auto_corr = acf(data, nlags=400, fft=True)[base:]
    
    local_max = argrelextrema(auto_corr, np.greater)[0]
    try:
        max_local_max = np.argmax([auto_corr[lcm] for lcm in local_max])
        if local_max[max_local_max]<3: #or local_max[max_local_max]>300:
            return 125
        return local_max[max_local_max]+base
    except:
        # return 125
        return 300
###########################################################################

class drift_window():
    def __init__(self, size, window, th):
        self.size = size
        self.seq = []
        self.idx_cl = []
        self.score = []
        self.curr = 0
        self.th = th
        self.win = window

    def enqueue(self, seq, idx, score, init=False):
        if init:
            self.seq = seq
            self.idx_cl = np.zeros(len(seq))
            self.score = np.zeros(len(seq))
            self.curr = len(seq)
        else:
            self.seq = np.append(self.seq, seq[-1])
            self.idx_cl = np.append(self.idx_cl, idx)
            self.score = np.append(self.score, score)
            self.curr +=1

        if self.curr > self.size:
            self.seq = np.delete(self.seq, 0)
            self.idx_cl = np.delete(self.idx_cl, 0)
            self.score = np.delete(self.score, 0)
            self.curr = self.size

    def is_Ex_Drift(self, cl):
        if self.curr < self.size: return -1
        if list(self.idx_cl).count(cl) > round(self.size*self.th): return cl
        else: return -1

    def is_New_Pattern(self, thres_cls):
        seq_to_check = [] ## extract the subsequences (# of length l)
        to_idx = []
        if self.curr < self.size: return [], []

        ## TODO: Need to identify the similarity within the L 
        ## cluster가 같아야 함 (curr cluster) & threshold 보다 커야 함 (anomalies) & 자기들끼리 비슷해야 함 (subsequence)
        cl_idx_list = [x for x in range(self.size) if self.score[x] >= thres_cls[int(self.idx_cl[x])]]
        start, end = cl_idx_list[0], cl_idx_list[-1]
        
        if end-start < round(self.size*self.th): return [], []

        if cl_idx_list[-1] - cl_idx_list[0]+1 >= len(cl_idx_list): 
            for i in range(len(cl_idx_list)-1):
                if cl_idx_list[i+1] - cl_idx_list[i] > self.win: start = cl_idx_list[i+1]

        ## Continuous list를 생성해야 함
        if end - start < round(self.size*self.th): return [], []

        seq_to_check = self.seq[start:end]
        to_idx = [start, end]
        return seq_to_check, to_idx
    
###########################################################################
def compute_min_score(ts, pattern_length, nms, scores_nms, normalize):
    # Compute score
    all_join = []
    for index_name in range(len(nms)):            
        
        join = stumpy.stump(ts,pattern_length,nms[index_name],ignore_trivial = False, normalize=normalize, p=1)[:,0]
 	   #join,_ = mp.join(nm_name + '/' + str(index_name),ts_name,len(nms[index_name]),len(ts), self.pattern_length)
        join = np.array(join)
        all_join.append(join)

    join = [0]*len(all_join[0]) # all_join 으로부터 join을 계산. all_join은 각 cluster의 mean subsequnece와 전체 time series의 join값
    for sub_join,scores_sub_join in zip(all_join,scores_nms):
        join = [float(j) + float(sub_j)*float(scores_sub_join) for j,sub_j in zip(list(join),list(sub_join))]
    join = np.array(join)

    # join_n = running_mean(join, pattern_length)
    # join_n = np.array([join_n[0]]*(pattern_length//2) + list(join_n) + [join_n[-1]]*(pattern_length//2))
    join_n = join
    
    return join_n

###########################################################################
def chk_other_cluster(seq, cl_NMs, cl_Ws, window_size,  normalize):
    d_cl = []
    score_cl = []
    for cl in range(len(cl_NMs)):
        t_score = compute_min_score(seq, window_size, cl_NMs[cl], cl_Ws[cl], normalize)
        d_cl.append(t_score)
        # score_cl.append(t_score[:window_size])

    return np.min(d_cl), np.argmin(d_cl)

###########################################################################
def compute_mp(seq, l, nms, normalize):
    # Compute score
    all_join = []
    for index_name in range(len(nms)):            
        join = stumpy.stump(seq,l,nms[index_name],ignore_trivial = False, normalize=normalize, p=1)[:,0]
        all_join.append(np.min(join))

    return np.min(all_join), all_join

###########################################################################
def chk_add_nm(L, thres_cls, normalize, init=False):
    ## Check how many 'anomalies' in the L
    to_check, to_idx = L.is_New_Pattern(thres_cls)
    if len(to_check) ==0 : return []

    ## compute distances between subseq in to_check
    candidate = to_check[-L.win:]
    temp_candidate = np.concatenate([candidate, candidate])

    ## TODO: check the unit-length of including candidate (1-by-1 or window_size)
    while len(to_check) - len(candidate) >= L.win:
        test = to_check[len(to_check)-len(candidate)-L.win:len(to_check)-len(candidate)]
        dist = stumpy.stump(test, L.win, temp_candidate, ignore_trivial=False, normalize=normalize, p=1)[0][0]
        if init:
            check_score = stumpy.stump(L.seq, L.win, temp_candidate, ignore_trivial=False, normalize=normalize, p=1)[:,0]
            # print('TA:', temp_candidate)
            # print(check_score)
            temp_th = np.mean(check_score) + 3*np.std(check_score)
            # print('Temp Init. TH:', temp_th)
            thres_cls[0] = temp_th

        if dist < thres_cls[int(L.idx_cl[-1])]: ## curr. threshold
            candidate = np.concatenate([test, candidate])
            temp_candidate = candidate
        else:
            break
    
    if len(candidate) >= 3*L.win:
        return candidate[-3*L.win:]
    else:
        return []

#
#    sim_seq = []
#    for i in range(len(to_check)-L.win):        
#        base = to_check[-L.win-i:-i]
#        base = np.concatenate([base, base])
#        remain = to_check[:-L.win-i]               
#        
#
#        mp_ds = compute_mp(base, L.win, remain, normalize=normalize)[1]
#                
#        if init ==True:
#            thres = np.mean(mp_ds) + 3*np.std(mp_ds)
#            print('Temp TH:', thres)
#            sim = [x for x in range(len(mp_ds)) if mp_ds[x] < thres]
#        else:
#            sim = [x for x in range(len(mp_ds)) if mp_ds[x] < np.min(thres_cls)]
#            print('Inter distances:', mp_ds, np.min(thres_cls), normalize)
#        
#        sim_L = [to_idx[y] for y in sim] + [len(to_idx)-1]
#        # print('SIM:', sim, to_idx, sim_L)
#
#        if (len(sim_L) > round(L.size*0.5)) and (len(sim_L) >=3):
#            for j in range(len(sim_L)-3,-1,-1):
#                if sim_L[j+2] - sim_L[j] ==2:
#                    return_seq = np.concatenate([L.seq[sim_L[j]], L.seq[sim_L[j+1]], L.seq[sim_L[j+2]]])
#                    print('Index for new NM:',sim_L[j], len(sim_L))
#                    return return_seq, sim_L
#
#            sim_seq.append(np.mean([to_check[x] for x in range(len(to_check)) if x in [i]+ sim], axis=0))
#                       
#    return [], [] #to_check, to_idx #sim_seq
############################################################################
def plotFigKL(data, label, label_cd, score, a_score, ths, slidingWindow, fileName, modelName, plotRange=None, y_pred=None, ADAD=True):
    grader = metricor()
    
    R_AUC, R_AP, R_fpr, R_tpr, R_prec = grader.RangeAUC(labels=label, score=score, window=slidingWindow, plot_ROC=True, ADAD=ADAD) #
    
    L, fpr, tpr= grader.metric_new(label, a_score, plot_ROC=True, ADAD=ADAD, thres=np.mean(ths))
    precision, recall, AP = grader.metric_PR(label, score)
    
    range_anomaly = grader.range_convers_new(label)
    # print(range_anomaly)
    
    # max_length = min(len(score),len(data), 20000)
    max_length = len(score)

    if plotRange==None:
        plotRange = [0,max_length]
    
    fig3 = plt.figure(figsize=(20, 10), constrained_layout=True)
    gs = fig3.add_gridspec(3, 4)
    
    
    # f3_ax1 = fig3.add_subplot(gs[0, :-1])
    f3_ax1 = fig3.add_subplot(gs[0, :])
    plt.tick_params(labelbottom=False)
   
    plt.plot(data[:max_length],'k')
    # added by jj
    if np.any(y_pred):
        plt.plot(y_pred[:max_length], 'c')
    for r in range_anomaly:
        if r[0]==r[1]:
            plt.plot(r[0],data[r[0]],'r.')
        else:
            plt.plot(range(r[0],r[1]+1),data[range(r[0],r[1]+1)],'r')
    # plt.xlim([0,max_length])
    plt.xlim(plotRange)
    
        
    # L = [auc, precision, recall, f, Rrecall, ExistenceReward, 
    #       OverlapReward, Rprecision, Rf, precision_at_k]
    # f3_ax2 = fig3.add_subplot(gs[1, :-1])
    f3_ax2 = fig3.add_subplot(gs[1, :])
    # plt.tick_params(labelbottom=False)
    L1 = [ '%.2f' % elem for elem in L]
    plt.plot(ths[:max_length]/np.max(a_score), 'r:', label='Thresholds')
    plt.plot(a_score[:max_length]/np.max(a_score), 'k', label='score')
    # plt.plot(score[:max_length], 'b', label='Anomalies')
    plt.fill_between(np.arange(score.shape[0]), score, color='blue', alpha=0.3, label='Anomalies')
    plt.fill_between(np.arange(label_cd.shape[0]), label_cd, color='yellow', alpha=0.3, label='Drift')
    plt.legend(loc='best')

    # delete threshold here
    # plt.hlines(np.mean(score)+3*np.std(score),0,max_length,linestyles='--',color='red')
    
    plt.ylabel('score')
    # plt.xlim([0,max_length])
    plt.xlim(plotRange)
    
    
    #plot the data
    # f3_ax3 = fig3.add_subplot(gs[2, :-1])
    f3_ax3 = fig3.add_subplot(gs[2, :])
    # index = ( label + 2*(score > (np.mean(score)+3*np.std(score))))
    index = ( label + 2*(score > 0))
    cf = lambda x: 'k' if x==0 else ('r' if x == 1 else ('g' if x == 2 else 'b') )
    cf = np.vectorize(cf)
    
    color = cf(index[:max_length])
    black_patch = mpatches.Patch(color = 'black', label = 'TN')
    red_patch = mpatches.Patch(color = 'red', label = 'FN')
    green_patch = mpatches.Patch(color = 'green', label = 'FP')
    blue_patch = mpatches.Patch(color = 'blue', label = 'TP')
    plt.scatter(np.arange(max_length), data[:max_length], c=color, marker='.')
    plt.legend(handles = [black_patch, red_patch, green_patch, blue_patch], loc= 'best')
    # plt.xlim([0,max_length])
    plt.xlim(plotRange)
    
    
    # f3_ax4 = fig3.add_subplot(gs[0, -1])
    # # plt.plot(fpr, tpr)
    # plt.plot(R_fpr,R_tpr)
    # # plt.title('R_AUC='+str(round(R_AUC,3)))
    # plt.xlabel('FPR')
    # plt.ylabel('TPR')
    # # plt.legend(['ROC','Range-ROC'])
    
    # f3_ax5 = fig3.add_subplot(gs[1, -1])
    # plt.plot(recall, precision)
    # plt.plot(R_tpr[:-1],R_prec)   # I add (1,1) to (TPR, FPR) at the end !!!
    # plt.xlabel('Recall')
    # plt.ylabel('Precision')
    # plt.legend(['PR','Range-PR'])
    
    plt.suptitle(fileName + '    window='+str(slidingWindow) +'   '+ modelName
    +'\nAUC='+L1[0]+'     R_AUC='+str(round(R_AUC,2))+'     Precision='+L1[1]+ '     Recall='+L1[2]+'     F='+L1[3]
    + '     ExistenceReward='+L1[5]+'   OverlapReward='+L1[6]
    +'\nAP='+str(round(AP,2))+'     R_AP='+str(round(R_AP,2))+'     Precision@k='+L1[9]+'     Rprecision='+L1[7] + '     Rrecall='+L1[4] +'    Rf='+L1[8]
    )
    # L = printResult(data, label, score, slidingWindow, fileName=name, modelName=modelName)
    column = ['auc', 'precision', 'recall', 'f', 'Rrecall', 'ExistenceReward', 'OverlapReward', 'Rprecision', 'Rf', 'precision_at_k']
    L[0] = R_AUC
    df = pd.DataFrame(columns=column)
    df = df.append(pd.Series(L, index=df.columns), ignore_index=True)
    print(df)
    return df


def running_mean(x,N):
	return (np.cumsum(np.insert(x,0,0))[N:] - np.cumsum(np.insert(x,0,0))[:-N])/N
# Revised by JP
def compute_score(ts, pattern_length, nms, scores_nms, normalize):
    # Compute score
    all_join = []
    for index_name in range(len(nms)):            
        join = stumpy.stump(ts,pattern_length,nms[index_name],ignore_trivial = False, normalize=normalize, p=1)[:,0]
 	   #join,_ = mp.join(nm_name + '/' + str(index_name),ts_name,len(nms[index_name]),len(ts), self.pattern_length)
        join = np.array(join)
        all_join.append(join)

    join = [0]*len(all_join[0]) # all_join 으로부터 join을 계산. all_join은 각 cluster의 mean subsequnece와 전체 time series의 join값
    for sub_join,scores_sub_join in zip(all_join,scores_nms):
        join = [float(j) + float(sub_j)*float(scores_sub_join) for j,sub_j in zip(list(join),list(sub_join))]
    join = np.array(join)
    join_n = running_mean(join,pattern_length)
        # join_n = join
    
    #reshifting the score time series
    join_n = np.array([join_n[0]]*(pattern_length//2) + list(join_n) + [join_n[-1]]*(pattern_length//2))
    if len(join_n) > len(ts):
        join_n = join_n[:len(ts)]

    # return join_n, all_join
    return join_n/pattern_length, all_join

def plotFig(data, label, score, slidingWindow, fileName, modelName, plotRange=None, y_pred=None, th=None):
    grader = metricor()
    
    if np.sum(label) != 0:
        R_AUC, R_AP, R_fpr, R_tpr, R_prec = grader.RangeAUC(labels=label, score=score, window=slidingWindow, plot_ROC=True) #
    
        L, fpr, tpr= grader.metric_new(label, score, plot_ROC=True, thres= th)
        precision, recall, AP = grader.metric_PR(label, score)
    
    range_anomaly = grader.range_convers_new(label)

    # max_length = min(len(score),len(data), 20000)
    max_length = len(score)

    if plotRange==None:
        plotRange = [0,max_length]
    
    fig3 = plt.figure(figsize=(20, 10), constrained_layout=True)
    gs = fig3.add_gridspec(3, 4)
    
    # f3_ax1 = fig3.add_subplot(gs[0, :-1])
    f3_ax1 = fig3.add_subplot(gs[0, :])
    plt.tick_params(labelbottom=False)
   
    plt.plot(data[:max_length],'k')
    if np.any(y_pred):
        plt.plot(y_pred[:max_length], 'c')
    for r in range_anomaly:
        if r[0]==r[1]:
            plt.plot(r[0],data[r[0]],'r.')
        else:
            plt.plot(range(r[0],r[1]+1),data[range(r[0],r[1]+1)],'r')
    # plt.xlim([0,max_length])
    plt.xlim(plotRange)
    
        
    # L = [auc, precision, recall, f, Rrecall, ExistenceReward, 
    #       OverlapReward, Rprecision, Rf, precision_at_k]
    # f3_ax2 = fig3.add_subplot(gs[1, :-1])
    f3_ax2 = fig3.add_subplot(gs[1, :])
    # plt.tick_params(labelbottom=False)
    if np.sum(label) != 0:
        L1 = [ '%.2f' % elem for elem in L]
    plt.plot(score[:max_length])
    if th is None:
        plt.hlines(np.mean(score)+3*np.std(score),0,max_length,linestyles='--',color='red')
    else:
        plt.hlines(th,0,max_length,linestyles='--',color='red')
    plt.ylabel('score')
    # plt.xlim([0,max_length])
    plt.xlim(plotRange)
    
    
    #plot the data
    # f3_ax3 = fig3.add_subplot(gs[2, :-1])
    f3_ax3 = fig3.add_subplot(gs[2, :])
    if th is None:
        index = ( label + 2*(score > (np.mean(score)+3*np.std(score))))
    else:
        index = (label + 2*(score > th))
    cf = lambda x: 'k' if x==0 else ('r' if x == 1 else ('g' if x == 2 else 'b') )
    cf = np.vectorize(cf)
    
    color = cf(index[:max_length])
    black_patch = mpatches.Patch(color = 'black', label = 'TN')
    red_patch = mpatches.Patch(color = 'red', label = 'FN')
    green_patch = mpatches.Patch(color = 'green', label = 'FP')
    blue_patch = mpatches.Patch(color = 'blue', label = 'TP')
    plt.scatter(np.arange(max_length), data[:max_length], c=color, marker='.')
    plt.legend(handles = [black_patch, red_patch, green_patch, blue_patch], loc= 'best')
    # plt.xlim([0,max_length])
    plt.xlim(plotRange)
    
    
    # f3_ax4 = fig3.add_subplot(gs[0, -1])
    # # plt.plot(fpr, tpr)
    # plt.plot(R_fpr,R_tpr)
   ##  plt.title('R_AUC='+str(round(R_AUC,3)))
    # plt.xlabel('FPR')
    # plt.ylabel('TPR')
   ##  plt.legend(['ROC','Range-ROC'])
    
    # f3_ax5 = fig3.add_subplot(gs[1, -1])
    # plt.plot(recall, precision)
    # plt.plot(R_tpr[:-1],R_prec)   # I add (1,1) to (TPR, FPR) at the end !!!
    # plt.xlabel('Recall')
    # plt.ylabel('Precision')
    # plt.legend(['PR','Range-PR'])
    if np.sum(label) != 0:
        plt.suptitle(fileName + '    window='+str(slidingWindow) +'   '+ modelName
        +'\nAUC='+L1[0]+'     R_AUC='+str(round(R_AUC,2))+'     Precision='+L1[1]+ '     Recall='+L1[2]+'     F='+L1[3]
        + '     ExistenceReward='+L1[5]+'   OverlapReward='+L1[6]
        +'\nAP='+str(round(AP,2))+'     R_AP='+str(round(R_AP,2))+'     Precision@k='+L1[9]+'     Rprecision='+L1[7] + '     Rrecall='+L1[4] +'    Rf='+L1[8]
        )
    # L = printResult(data, label, score, slidingWindow, fileName=name, modelName=modelName)
    column = ['auc', 'precision', 'recall', 'f', 'Rrecall', 'ExistenceReward', 'OverlapReward', 'Rprecision', 'Rf', 'precision_at_k']

    df = pd.DataFrame(columns=column)
    L[0] = R_AUC
    df = df.append(pd.Series(L, index=df.columns), ignore_index=True)
    print(df)
    return df