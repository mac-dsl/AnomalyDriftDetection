import numpy as np
from util.util_overlap import *
from tqdm.notebook import tqdm

class ADDD():

    def __init__(self,pattern_length, size_L=5, normalize=True):
        self.size_L = size_L
        self.pattern_length = pattern_length
        self.cl_NMs =[]
        self.cl_Ws = []
        self.thres_cls = []
        self.seq_idx=pattern_length*size_L
        self.normalize=normalize
        self.L = drift_window(size=size_L*pattern_length, window=pattern_length, th=0.5)
        ## TODO

    def fit(self, X, y=None, online=True, training=False, delta=0, overlapping=1, correct=False, training_len=None):
        ## if overlapping is not 1, revise for loop 
        self.overlapping = overlapping
        self.s_score = []
        self.s2_score = []
        self.ths =[]
        self.ts = X


        if online:
            if training:
                ## initialize with NormA
                self._set_norma(training_len)
            else:
                self._init(len(X))
                # print('Init. NM:', self.cl_NMs)

            for self.seq_idx in tqdm(range(self.pattern_length*self.size_L, len(X)-self.pattern_length,1)):
                test_seq = self.ts[self.seq_idx:self.seq_idx+self.pattern_length]
                t_score = self._get_anomaly_score(test_seq)
                t2_score = 1 if t_score > self.thres_cls[self.curr_cluster] else 0

                ## Anomaly
                if t2_score:
                    d_cl, d_cl_idx = self._get_dist_NMs(test_seq)
                    ## Similar to one of the other cluster
                    if d_cl < self.thres_cls[d_cl_idx]:
                        self.L.enqueue(self.test_sample, d_cl_idx, d_cl)
                        if d_cl_idx != self.curr_cluster:
                            ## Examine re-occurred drift
                            if self.L.is_Ex_Drift(d_cl_idx) == d_cl_idx:
                                print('[MP] Re: Drift at', self.seq_idx, 'From', self.curr_cluster, 'to', d_cl_idx, 'Cls:', self.L.idx_cl)
                                self.curr_cluster = d_cl_idx

                    ## Check a new drift
                    else:
                        self.L.enqueue(test_seq, self.curr_cluster, t_score)
                        to_chk = chk_add_nm(self.L, self.thres_cls, self.normalize)
                        if len(to_chk) > 0:
                            ## add cluster
                            self.cl_NMs.append(to_chk)
                            self.cl_Ws.append(np.sum(self.cl_Ws[self.curr_cluster]))
                            self.thres_cls.append(self.thres_cls[self.curr_cluster])
                            print('[MP] Add NM at', self.seq_idx, 'Distance:', d_cl, d_cl_idx)  
                            self.curr_cluster = len(self.cl_NMs)-1

                ## Normal
                else:
                    self.L.enqueue(test_seq, self.curr_cluster, t_score)
                
                self.ths = np.concatenate([self.ths, [self.thres_cls[self.curr_cluster]]])
                self.s_score = np.concatenate([self.s_score, t_score])
                self.s2_score = np.concatenate([self.s2_score, [t2_score]])

            ## Results
            self.s_score = np.array(list(self.s_score[((self.pattern_length)//2):]) + [self.s_score[-1]]*((self.pattern_length)//2))
            self.s2_score = np.array(list(self.s2_score[((self.pattern_length)//2):]) + [self.s2_score[-1]]*((self.pattern_length)//2))
        
        

    def _get_anomaly_score(self, seq):
        # To revise the scoring function with subsampling
        join = stumpy.stump(seq, self.pattern_length, np.array(self.cl_NMs[self.curr_cluster]).reshape(-1), ignore_trivial=False, normalize=self.normalize, p=1)[:,0]
        join = join/self.pattern_length
        return np.array(join)
    
    def _get_dist_NMs(self, seq):
        d_cl = []
        for cl in self.cl_NMs:
            d_cl.append(stumpy.stump(seq, self.pattern_length, np.array(cl).reshape(-1), ignore_trivial=False, normalize =self.normalize, p=1)[:,0]/self.pattern_length)
        return np.min(d_cl), np.argmin(d_cl)
    
    def _init(self, size_ts):

        self.L.enqueue(self.ts[:self.pattern_length*self.size_L], 0, score=100, init=True)
        for self.seq_idx in range(self.pattern_length*self.size_L, size_ts-self.pattern_length,1):
            test_sample = self.ts[self.seq_idx:self.seq_idx+self.pattern_length]
            self.L.enqueue(test_sample, 0, 0)
            to_chk = chk_add_nm(self.L, [0], self.normalize, init=True)
            if len(to_chk) >0:
                self.cl_NMs.append([to_chk])
                self.cl_Ws.append([1])
                temp = compute_min_score(self.L.seq, self.pattern_length, self.cl_NMs[0], self.cl_Ws[0], self.normalize)/self.pattern_length
                m_t = np.mean(temp[np.nonzero(temp)])
                std_t = np.std(temp[np.nonzero(temp)])
                max_t = np.max(temp[np.nonzero(temp)])
                if m_t + 3*std_t < max_t:
                    self.thres_cls.append(m_t+max_t)
                else:
                    self.thres_cls.append(m_t+3*std_t)
                self.curr_cluster = 0
                self.s_score = np.zeros(self.seq_idx+self.pattern_length)
                self.s2_score = np.zeros(self.seq_idx+self.pattern_length)
                self.ths = np.ones(self.seq_idx+self.pattern_length)*self.thres_cls[0]
                break
            else:
                continue




        

def addd_cl_overlap_cold(data, label, label_cd, window_size, normalize, size_L = 5, correct=False):
    # window_size, cl_NMs, cl_Ws, thres_cls, curr_cluster
    if window_size < 5:
        window_size = find_length(data)
    
    # L = Window(size=size_L)
    L = drift_window(size=size_L*window_size, window = window_size, th=0.5)    
    cl_NMs, cl_Ws, thres_cls = [], [], []

    # prev_seq = data[:window_size]
    s_score, s2_score, ths = [], [], []
    state = False
    L.enqueue(data[:window_size*size_L], 0, score =100, init=True)
    seq_idx=window_size*size_L

    # while seq_idx <= len(data) - window_size:
    for seq_idx in tqdm(range(window_size*size_L, len(data)-window_size, 1)):
        ## test subseq.
        test_sample = data[seq_idx:seq_idx+window_size]
#
        # Cold start: no cluster yet
        if len(cl_NMs) ==0:            
            L.enqueue(test_sample, 0, 0)
            to_chk = chk_add_nm(L, [0], normalize, init=True)
            # print(to_idx)
            if len(to_chk) > 0:
                # print(seq_idx)
                cl_NMs.append([to_chk])
                cl_Ws.append([1])
                # print('CL:', cl_NMs)
                temp = compute_min_score(L.seq, window_size, cl_NMs[0], cl_Ws[0], normalize)
                # print('NM_score',[np.mean(temp[np.nonzero(temp)]) + 6*np.std(temp[np.nonzero(temp)])], np.mean(temp+6*np.std(temp)))
                m_t = np.mean(temp[np.nonzero(temp)])
                std_t = np.std(temp[np.nonzero(temp)])
                max_t = np.max(temp[np.nonzero(temp)])
                print('mean, std, max:', m_t, std_t, max_t)
                if m_t + 3*std_t < max_t:
                    thres_cls.append((m_t+max_t))
                else:
                    # thres_cls.append(m_t+3*std_t)
                    thres_cls.append((m_t+max_t))

                init_thres = thres_cls[0]
                if init_thres > 2*m_t:
                    init_thres = 2*m_t
                    thres_cls[0] = init_thres
                print('init:', init_thres, seq_idx)
                curr_cluster = 0 #len(cl_NMs)-1


                ## revise L here
                # L.clear_queue()
                s_score = np.zeros(seq_idx+window_size)
                s2_score = np.zeros(seq_idx+window_size)
                ths = np.ones(seq_idx+window_size)*thres_cls[0]
                # prev_seq = test_sample
                continue
            else:
                # seq_idx += 1
                continue

        ## compute Anomaly Score
        ## In case of overlapped window, score (t_score and t2_score) is a single value for incoming data point at t        
        t_score = compute_min_score(test_sample, window_size, cl_NMs[curr_cluster], cl_Ws[curr_cluster], normalize=normalize)
        if t_score > thres_cls[curr_cluster]: t2_score = 1
        else: t2_score = 0

        # print('T2:',t2_score, t_score)
        ## TODO: dividing multiple sub-window to examine point(or partial) anomalies

        ## Anomaly detection in the cluster
        if t2_score == 1:
            
            ## Check re-occurring cluster first
            d_cl, d_cl_idx = chk_other_cluster(test_sample, cl_NMs, cl_Ws, window_size, normalize)
            
            ## Close to one of the other cluster, 
            if d_cl < thres_cls[d_cl_idx]:
                L.enqueue(test_sample, d_cl_idx, d_cl)
                if d_cl_idx != curr_cluster:                                        
                    if L.is_Ex_Drift(d_cl_idx) == d_cl_idx: 
                        ## Re-occurring drift (KL)
                        print('Change of Cluster:', d_cl_idx, 'at', seq_idx)
                        print('[MP] Re: Drift at', seq_idx, 'From', curr_cluster, 'to', d_cl_idx, 'Cls:', L.idx_cl)                        
                        curr_cluster = d_cl_idx
                    # else:
                        # continue
                    ################## TODO
            else:
                ## Check add a new cluster or not                
                L.enqueue(test_sample, curr_cluster, t_score)    
                to_chk = chk_add_nm(L, thres_cls, normalize)
                if len(to_chk) > 0:
                    ## Add cluster here
                    cl_NMs.append([to_chk])
                    cl_Ws.append([np.sum(cl_Ws[curr_cluster])])
                    thres_cls.append(thres_cls[curr_cluster])  ### TODO: how can we initialize the threshold for new cluster. 
                    print('[MP] Add NM at', seq_idx, 'Distance:', d_cl, d_cl_idx)  
                    print('Curr:', curr_cluster, 'New:', len(cl_NMs)-1, len(cl_NMs[-1]))              
                    # plt.figure(figsize=(20,5))
                    # for nm in cl_NMs[curr_cluster]:
                        # plt.plot(nm)
                    # plt.plot(data[seq_idx - (L.curr-1)*window_size:seq_idx+ window_size], 'k',label='Seq. in L')
                    # plt.plot(to_chk, 'r', label='New')
                    # plt.legend()
                    # plt.show()
                    # L.clear_queue()          
                    curr_cluster = len(cl_NMs)-1



                # else:
                    # if state==False:
                    # thres_cls[curr_cluster] = thres_cls[curr_cluster] + t_score/window_size
                # state = True

        else: ## Normal
            # state=False
            L.enqueue(test_sample, curr_cluster, t_score)
            # print(len(cl_NMs[curr_cluster][0]), cl_NMs[curr_cluster])
            
            # cl_NMs[curr_cluster][0] = L.seq[-len(cl_NMs[curr_cluster][0]):]

            # if (cnt_normal == L.size) and (thres_cls[curr_cluster] > init_thres):
                # thres_cls[curr_cluster] = thres_cls[curr_cluster]-th_normal/L.size
            # print(thres_cls[curr_cluster])
            
        prev_seq = test_sample
        ths = np.concatenate([ths, [thres_cls[curr_cluster]]])
        # print('S1:', s_score, t_score)
        # print('S2:', s2_score, t2_score)
        s_score = np.concatenate([s_score, t_score])
        s2_score = np.concatenate([s2_score, [t2_score]])
        # seq_idx = seq_idx + 1   ## overlapping window    

    # s_score = running_mean(s_score, window_size)
    # s_score = np.array([s_score[0]]*(window_size//2) + list(s_score) + [s_score[-1]]*((window_size)//2))
    print('SCORE:', len(s_score), s_score)
    s_score = np.array(list(s_score[((window_size)//2):]) + [s_score[-1]]*((window_size)//2))
    s2_score = np.array(list(s2_score[((window_size)//2):]) + [s2_score[-1]]*((window_size)//2))
    # print('LEN: pattern:', window_size, 'data:', len(data), 'score:', len(s2_score), 'each:', len(t_score))    

    df_result = plotFigKL(data[:len(s2_score)], label[:len(s2_score)], label_cd[:len(s2_score)], s2_score, s_score, ths, window_size, 'ECG', 'AD-DD', plotRange=None, y_pred=None, ADAD=False)    
    df_result2 = plotFigKL(data[:len(s2_score)], label[:len(s2_score)], label_cd[:len(s2_score)], s2_score, s_score, ths, window_size, 'ECG', 'AD-DD', plotRange=[30000,35000], y_pred=None, ADAD=False)    
    return df_result, cl_NMs, s_score, s2_score
