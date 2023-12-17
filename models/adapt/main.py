import os
import torch
import torch.nn.functional as F
from torch.utils.data import random_split
from torch_geometric.data import DataLoader
# from OTC_data import BitcoinOTC
from inj_cora_dataset import InjCoraDataset
import util
from networks import Net
import numpy as np
from sklearn.metrics import roc_auc_score

from sklearn.metrics import recall_score


import time

import warnings
warnings.filterwarnings('ignore')


#parameter initialization
parser = util.parser
args = parser.parse_args()
torch.manual_seed(args.seed)
print(args)

#device selection
if torch.cuda.is_available():
    torch.cuda.manual_seed(args.seed)
    args.device = 'cuda:6'
else:
    args.device = 'cpu'

#dataset split
def data_builder(args):

    # dataset = InjCoraDataset(root='./data/inj_flickr/train_10')
    dataset = InjCoraDataset(root='./data/inj_cora/train_30')
    # dataset = InjCoraDataset(root='./data/inj_amazon/train_30')
    args.num_classes = 2
    args.num_features = 1433

    num_training = int(len(dataset)*0.05)
    num_val = int(len(dataset)*0.5)
    num_test = len(dataset) - (num_training+num_val)
    training_set,validation_set,test_set = random_split(dataset,[num_training,num_val,num_test])

    train_loader = DataLoader(training_set, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(validation_set,batch_size=args.batch_size,shuffle=False)
    test_loader = DataLoader(test_set,batch_size=args.batch_size,shuffle=False)


    return train_loader, val_loader, test_loader
   
#test function
def test(model,loader):
    model.eval()
    correct = 0.
    loss = 0.
    anomaly_label_graph = []
    anomaly_pred_graph = []
    anomaly_pred_label = []


    for data in loader:
        data = data.to(args.device)
        out, prop = model(data)
        pred = out.max(dim=1)[1]

        prop = prop[:,1]
        correct += pred.eq(data.ys).sum().item()
        loss += F.nll_loss(out,data.ys,reduction='sum').item()

        anomaly_label_graph.append(data.ys)
        anomaly_pred_graph.append(prop)
        anomaly_pred_label.append(pred)


    y_label_list = torch.cat(anomaly_label_graph, 0).view(-1).tolist()
    score_list = torch.cat(anomaly_pred_graph, 0).view(-1).tolist()
    pred_label_list = torch.cat(anomaly_pred_label, 0).view(-1).tolist()
    y_label = np.array(y_label_list)
    scores = np.array(score_list)
    pred_label = np.array(pred_label_list)
    auc = roc_auc_score(y_label, scores)

    # Presicion@100
    temp = list(zip(score_list, y_label_list))
    sorted_K = sorted(temp, key=lambda x: x[0], reverse=True)
    result = zip(*sorted_K)
    sorted_label = [list(x) for x in result]

    precision100 = np.mean(sorted_label[1][:100])
    recall = recall_score(y_label, pred_label, pos_label=1, average='binary', sample_weight=None)

    print('auc', auc,'precision100',precision100,"recall",recall)
    return correct / len(loader.dataset),loss / len(loader.dataset)


#training configuration
train_loader, val_loader, test_loader = data_builder(args)
model = Net(args).to(args.device)
optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)


#training steps
patience = 0
min_loss = args.min_loss
start_time = time.time()
for epoch in range(args.epochs):
    model.train()
    print("Epoch{}".format(epoch))
    for i, data in enumerate(train_loader):
        # noise injection
        # t, _ = schedule_sampler.sample(batch.shape[0], dist_util.dev())
        # batch = diffusion.q_sample(batch,t)

        data = data.to(args.device)
        out,_ = model(data)
        loss = F.nll_loss(out, data.ys)

        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

    val_acc,val_loss = test(model,val_loader)
    print("Validation loss:{}\taccuracy:{}".format(val_loss,val_acc))

    if val_loss < min_loss:
        torch.save(model.state_dict(),'latest_cora5.pth')
        print("Model saved at epoch{}".format(epoch))
        min_loss = val_loss
        patience = 0
    else:
        patience += 1
    if patience > args.patience:
        end_time = time.time()
        elapsed_time = end_time - start_time
        print(f"time：{elapsed_time}seconds")
        break

#test step
model = Net(args).to(args.device)
model.load_state_dict(torch.load('latest_cora5.pth'))
test_acc,test_loss = test(model,test_loader)
print("Test accuarcy:{}".format(test_acc))




