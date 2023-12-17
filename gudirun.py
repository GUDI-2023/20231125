import tqdm
import torch
import argparse
import warnings
from models.metric import *
from models.utils import load_data,load_pretrain_data
from models.detector.gaegsa import encoderf
from models.detector.encoders import encoders
from models.adapt.networks import Net
from models.adapt import util
from torch_geometric.data import DataLoader
from models.adapt.inj_cora_dataset import InjCoraDataset
import torch.nn.functional as F
import numpy as np



def minmaxscaler(data):
    min = torch.min(data)
    max = torch.max(data)
    return (data - min)/(max-min)


def main(args):
    auc = []
    rec = []

    for _ in tqdm.tqdm(range(num_trial)):

        pretrain_data = load_pretrain_data()

        # pretrain structure reconstruction networks
        model1 = encoders()
        for data in pretrain_data:
            model1.fit(data)

        # pretrain feature reconstruction networks
        model2 = encoderf()
        for data in pretrain_data:
            model2.fit(data)

        # load specific dataset
        data = load_data('inj_cora')
        k_all = sum(data.y)
        # load sampled subgraph with 30 nodes
        dataset = InjCoraDataset(root='./pygod/pretrain/data/inj_cora/eval_30')

        # Inference with guided diffusion
        parser = util.parser
        args = parser.parse_args()
        # load test dataset
        args.num_classes = 2
        args.num_features = 1433
        if torch.cuda.is_available():
            torch.cuda.manual_seed(args.seed)
            args.device = 'cuda:0'

        loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False)
        # call the subgraph classifier, which is trained separately in ./models/adapt/main.py
        classifier = Net(args).to(args.device)
        classifier.load_state_dict(torch.load('models/pretrain/latest_cora.pth'))
        classifier.eval()
        correct = 0.
        loss = 0.
        anomaly_label_graph = []
        anomaly_pred_graph = []

        # calculate the prediction by subgraph classifier
        for data_pre in loader:
            data_pre = data_pre.to(args.device)
            s_recover = model1.forward_di(data_pre)
            f_recover = model2.forward_di(data_pre)

            for t in range(0,3):
                out, prop= classifier(model1.toden(s_recover, f_recover))
                pred = out.max(dim=1)[1]
                prop_s = prop_s[:, 1]
                prop_f = prop_f[:, 1]
                correct += pred.eq(data_pre.y).sum().item()
                loss += F.nll_loss(out, data_pre.y, reduction='sum').item()

                s_recover = model1.conditional_di(s_recover, prop_s)
                f_recover = model2.conditional_di(f_recover, prop_f)

            ss = model1.loss_func(data_pre, s_recover)
            sf = model2.loss_func(data_pre.x, f_recover)
            anomaly_label_graph.append(data_pre.label)
            anomaly_pred_graph.append(pred)

        y_label_list = torch.cat(anomaly_label_graph, 0).view(-1).tolist()
        pred_list = torch.cat(anomaly_pred_graph, 0).view(-1).tolist()


        pred_t = np.array(pred_list)
        data.y = data.y.bool().int()
        fs = torch.zeros_like(ss)
        # 0-1 scale
        ss = minmaxscaler(F.normalize(ss, p=2, dim=-1))
        sf = minmaxscaler(F.normalize(sf, p=2, dim=-1))

        for i, y in enumerate(data.y):
            assert y == y_label_list[i]
            fs[i] += args.lamba * pred_t[i] * ss[i] + (1 - args.lamba) * (1 - pred_t[i]) * sf[i]
        auc.append(eval_roc_auc(data.y, fs))
        rec.append(eval_recall_at_k(data.y, fs, k_all))

    print("AUC: {:.4f}±{:.4f} ({:.4f})\t"
          "Recall: {:.4f}±{:.4f} ({:.4f})\n"

          .format(np.mean(auc), np.std(auc), np.max(auc),
                  np.mean(rec), np.std(rec), np.max(rec)))



if __name__ == '__main__':

    # The article pertains to enterprise collaboration, and this part of the code is currently undergoing the patent application process.
    # Therefore, partially code is release just for paper review, but not workable.
    # Once the patent application is finalized, the code will be made publicly available in the first quarter of 2024.
    # The code is implemented on the benchmark work, pygod, refer to https://github.com/pygod-team/pygod/tree/main/benchmark
    parser = argparse.ArgumentParser()
    parser.add_argument("--gpu", type=int, default=1,
                        help="GPU Index. Default: -1, using CPU.")
    parser.add_argument("--dataset", type=str, default='inj_cora',
                        help="supported dataset: [inj_cora, inj_amazon,OTC,Weibo "
                             "inj_flickr]. Default: inj_cora.")
    args = parser.parse_args()

    # global setting
    num_trial = 5

    main(args)
