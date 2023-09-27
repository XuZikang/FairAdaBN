import pandas as pd
import torch
import numpy as np
from rich.console import Console
from pycm import ConfusionMatrix
import ipdb
from basemodels import cusResNet152
from collections import OrderedDict
import os

class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

class Fairness_Metrics():
    def __init__(self, result_frame):
        self.frame = result_frame
        self.groups = list(self.frame.groupby('sensitive'))

        self.p_group = self.groups[0][1]
        self.up_group = self.groups[1][1]
        
        self.cm = ConfusionMatrix(actual_vector=self.frame.y_true.to_numpy(), predict_vector=self.frame.y_pred.to_numpy())
        self.cm_p = ConfusionMatrix(actual_vector=self.p_group.y_true.to_numpy(), predict_vector=self.p_group.y_pred.to_numpy())
        self.cm_up = ConfusionMatrix(actual_vector=self.up_group.y_true.to_numpy(), predict_vector=self.up_group.y_pred.to_numpy())
        
        self.avg_performance()
        
        self.avg_scores = self.get_average()

    def accuracy(self):
        return self.cm.weighted_average('ACC')
    
    def precision(self):
        return self.cm.weighted_average('PPV')

    def recall(self):
        return self.cm.weighted_average('TPR')        

    def f1(self):
        return self.cm.weighted_average('F1')
    
    def statistical_parity(self, pos_label=0):
        '''
            |Pr(Y_hat = pos_label | a = 0) - Pr(Y_hat = pos_label | a = 1)|
        '''

        return np.abs((sum(self.p_group.y_pred == pos_label) - sum(self.up_group.y_pred == pos_label)) / len(self.p_group.y_pred))

    def equal_opportunity_0(self, pos_label=0):
        '''
            |TNR_0 - TNR_1|
        '''
        p_tnr = self.cm_p.TNR[pos_label]
        up_tnr = self.cm_up.TNR[pos_label]

        return np.abs(p_tnr - up_tnr)

    def equal_opportunity_1(self, pos_label=0):
        '''
            |TPR_0 - TPR_1|
        '''
        p_tpr = self.cm_p.TPR[pos_label]
        up_tpr = self.cm_up.TPR[pos_label]
        
        return np.abs(p_tpr - up_tpr)

    def equal_odds(self, pos_label=0):
        '''
           |TPR_0 - TPR_1 + FPR_1 - FPR_0| = |TPR_0 - TPR_1 + TNR_0 - TNR_1|
        '''
        p_tnr = self.cm_p.TNR[pos_label]
        up_tnr = self.cm_up.TNR[pos_label]
        p_tpr = self.cm_p.TPR[pos_label]
        up_tpr = self.cm_up.TPR[pos_label]
        
        return np.abs(p_tpr - up_tpr + p_tnr - up_tnr)
    
    def get_average(self):
        scores = {
            'accuracy': self.accuracy(),
            'precision': self.precision(),
            'recall': self.recall(),
            'f1': self.f1(),
            'statistical_parity': np.mean([self.statistical_parity(pos_label) for pos_label in range(7)]),
            'equal_opp_0': np.mean([self.equal_opportunity_0(pos_label) for pos_label in range(7)]),
            'equal_opp_1': np.mean([self.equal_opportunity_1(pos_label) for pos_label in range(7)]),
            'equal_odds': np.mean([self.equal_odds(pos_label) for pos_label in range(7)])
        }
        
        return scores
    
    def avg_performance(self):
        print('overall_PPV:\t{:.4f}'.format(self.cm.weighted_average('PPV', none_omit=True)))
        print('dark_PPV:\t{:.4f}'.format(self.cm_up.weighted_average('PPV', none_omit=True)))
        print('light_PPV:\t{:.4f}'.format(self.cm_p.weighted_average('PPV', none_omit=True)))
        
        print('overall_TPR:\t{:.4f}'.format(self.cm.weighted_average('TPR', none_omit=True)))
        print('dark_TPR:\t{:.4f}'.format(self.cm_up.weighted_average('TPR', none_omit=True)))
        print('light_TPR:\t{:.4f}'.format(self.cm_p.weighted_average('TPR', none_omit=True)))
        
        print('overall_F1:\t{:.4f}'.format(self.cm.weighted_average('F1', none_omit=True)))
        print('dark_F1:\t{:.4f}'.format(self.cm_up.weighted_average('F1', none_omit=True)))
        print('light_F1:\t{:.4f}'.format(self.cm_p.weighted_average('F1', none_omit=True)))
        
        # print('TPR:', self.cm.weighted_average('TPR'))
        # print('F1:', self.cm.weighted_average('F1'))
        
    
def save_best_model(model, idx, seed):
    torch.save(model, 'weights/rand_seed={}_exp_{}_best.pkl'.format(seed, idx))
    
def save_final_model(model, idx, seed):
    torch.save(model, 'weights/rand_seed={}_exp_{}_final.pkl'.format(seed, idx))
    
def save_logs(t_acc, t_loss, v_acc, v_loss, idx, seed):
    np.save('logs/rand_seed={}_exp_{}_train_acc.npy'.format(seed, idx), np.array(t_acc))
    np.save('logs/rand_seed={}_exp_{}_train_loss.npy'.format(seed, idx), np.array(t_loss))
    np.save('logs/rand_seed={}_exp_{}_valid_acc.npy'.format(seed, idx), np.array(v_acc))
    np.save('logs/rand_seed={}_exp_{}_valid_loss.npy'.format(seed, idx), np.array(v_loss))

def save_result(result, exp_id, method, seed):
    result.to_csv('preds/rand_seed={}_exp_{}_{}_prediction.csv'.format(seed, exp_id, method))
    
def save_fairness_score(fairness_score, seed, exp_id, method):
    fairness_score.to_csv('fair_scores/rand_seed={}_exp_{}_{}_fairness_scores.csv'.format(seed, exp_id, method))

def load_weights(net, weights='imagenet', pth='./pretrained_weights/exp_0_final.pkl'):
    if weights == 'imagenet':
        state_dict = cusResNet152(n_classes=9, pretrained=True).state_dict()
    else:
        assert os.path.exists(pth)
        state_dict = torch.load(pth).state_dict()
    
    # pairing keys
    net_keys = []
    for name, _ in net.state_dict().items():
        if not ('ins' in name or 'running' in name or 'batches' in name or 'bn' in name or 'shortcut.1' in name or 'conv1.1' in name):
            net_keys.append(name)
    pretrain_net_keys = []
    for name, _ in state_dict.items():
        if not ('bn' in name or 'running' in name or 'batches' in name or 'downsample.1' in name or 'shortcut' in name):
            pretrain_net_keys.append(name)
            
    renaming_dict = {}
    for i in range(len(pretrain_net_keys)):
        renaming_dict[pretrain_net_keys[i]] = net_keys[i]
    
    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        if not k in renaming_dict.keys():
            print('skip:', k)
            continue
        name = renaming_dict[k]
        new_state_dict[name] = v
    
    net.load_state_dict(new_state_dict, strict=False)