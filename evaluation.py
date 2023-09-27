import argparse

import ipdb
import torch
import torch.utils.data as data
from rich.console import Console
from torchvision import transforms

from Fitz17k import Fitz17kTest
from utils import *
from tqdm import tqdm

def test(net, p, up):
    # inits
    result_df = pd.DataFrame(columns=['y_pred', 'y_true', 'sensitive'])
    score_df = pd.DataFrame(columns=['statistical_parity', 'equal_opp_0', 'equal_opp_1', 'equal_odds'])
    
    # test
    net.eval()
    # test privileged group
    for _, image, label, sensitive in tqdm(p):
        image, label = image.cuda(), label.cuda()
        
        with torch.no_grad():
            output, _ = net(image, task_idx=0)
            preds = output.argmax(dim=1)
                
        for i in range(label.shape[0]):
            sample_df = pd.DataFrame(
                {
                    'y_pred': preds[i].item(),
                    'y_true': int(label[i][0].item()),
                    'sensitive': sensitive[i].item()
            }, index=[0])
            result_df = pd.concat([result_df, sample_df], ignore_index=True)
   
    # test unprivileged group
    for _, image, label, sensitive in tqdm(up):
        image, label = image.cuda(), label.cuda()
        
        with torch.no_grad():
            output, _ = net(image, task_idx=1)
            preds = output.argmax(dim=1)
                
        for i in range(label.shape[0]):
            sample_df = pd.DataFrame(
                {
                    'y_pred': preds[i].item(),
                    'y_true': int(label[i][0].item()),
                    'sensitive': sensitive[i].item()
            }, index=[0])
            result_df = pd.concat([result_df, sample_df], ignore_index=True)
            
    fair_metrics = Fairness_Metrics(result_df)
    fair_scores = pd.DataFrame(fair_metrics.avg_scores, index=[0])
    
    console.log(fair_scores)
        
    return result_df, fair_scores



if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Test settings')
    parser.add_argument('--exp_id', default=0)
    parser.add_argument('--batch_size', type=int, default=256)
    parser.add_argument('--method', choices=['final', 'best'])
    parser.add_argument('--rand_seed',type=str)
        
    args = parser.parse_args()
    
    # logs
    console = Console()
    console.log('baseline model with backbone resnet152')
    console.log(args)
    
    # paths
    ann_test = pd.read_csv('./dataset/Fitzpatrick-17k/processed/rand_seed={}/split/test.csv'.format(args.rand_seed), index_col=0)
    ann_test.reset_index(inplace=True)

    test_pkl = './dataset/Fitzpatrick-17k/processed/rand_seed={}/pkls/test_images.pkl'.format(args.rand_seed)

    # standard transform
    transform = transforms.Compose([
        transforms.ToTensor(),
    ])
    
    # create test dataset
    test_p_set = Fitz17kTest(dataframe=ann_test, path_to_pickles=test_pkl, sens_name='skintone', sens_classes=2, transform=transform, sens_attr='light')
    test_up_set = Fitz17kTest(dataframe=ann_test, path_to_pickles=test_pkl, sens_name='skintone', sens_classes=2, transform=transform, sens_attr='dark')
    
    test_p_loader = data.DataLoader(test_p_set,
                                     batch_size=args.batch_size,
                                     shuffle=False,
                                     num_workers=4,
                                     pin_memory=True)
    test_up_loader = data.DataLoader(test_up_set,
                                     batch_size=args.batch_size,
                                     shuffle=False,
                                     num_workers=4,
                                     pin_memory=True)
    # GPU setting
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # models
    net = torch.load('weights/rand_seed={}_exp_{}_{}.pkl'.format(args.rand_seed, args.exp_id, args.method))

    # test
    result_df, fairness_score = test(net, test_p_loader, test_up_loader)
    save_result(result_df, args.exp_id, args.method, args.rand_seed)
    save_fairness_score(fairness_score, args.rand_seed, args.exp_id, args.method)
    