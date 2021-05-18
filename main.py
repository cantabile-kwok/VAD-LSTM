import spafe
import spafe.utils.preprocessing as preprocess
import spafe.features.mfcc as mfcc
import librosa
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
import os
from tqdm import tqdm
import json
import argparse
from vad_utils import parse_vad_label, prediction_to_vad_label
from vad_utils import read_label_from_file
from evaluate import compute_eer, get_metrics
import re
import random

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def vis_sample(model, lbl_dict, feat_path, fig_path, save=True):
    """
    Visualize a sample prediction and its label
    :param model: the VAD model
    :param lbl_dict: map of sample ID and its label
    :param feat_path: sample feature path
    :param fig_path: where to store the figure
    :param save: if save, then save the figure to fig_path
    """
    sample_path = os.listdir(feat_path)[np.random.randint(len(os.listdir(feat_path)))]
    sample_id = sample_path.split('.')[0]

    sample = np.load(os.path.join(feat_path, sample_path))
    sample_label = lbl_dict[sample_id]

    hiddens = model.init_hiddens(1)
    hiddens = (hiddens[0].to(device), hiddens[1].to(device))

    sample_pred = model(torch.from_numpy(sample).unsqueeze(1).float().to(device), hiddens)
    sample_pred = sample_pred.squeeze()
    sample_pred = sample_pred.detach().cpu().numpy()
    auc_nosmooth, eer_nosmooth, _, _ = get_metrics(sample_pred, sample_label)

    plt.figure(figsize=(8, 4))
    plt.plot(sample_label, c='r', label='sample label')
    plt.plot(sample_pred + 0.01, c='g', linestyle='-.',
             label='prediction\n auc,eer={:.4f}, {:.4f}'.format(auc_nosmooth, eer_nosmooth))
    plt.legend(fontsize=10)
    plt.xlabel('frame', fontsize=12)
    if save:
        plt.savefig(os.path.join(fig_path, '{}_pred.png'.format(sample_id)), dpi=120)
    plt.show()


class VADnet(nn.Module):
    def __init__(self, input_dim, hidden_size, num_layers):
        super(VADnet, self).__init__()
        self.hidden_size = hidden_size
        self.input_dim = input_dim
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_size=input_dim,
                            hidden_size=hidden_size, num_layers=num_layers, bias=True, batch_first=False,
                            bidirectional=True)
        self.fc = nn.Linear(in_features=2 * hidden_size,
                            out_features=1, bias=True)
        self.sigmoid = nn.Sigmoid()

    def init_hiddens(self, batch_size):
        # hidden state should be (num_layers*num_directions, batch_size, hidden_size)
        # returns a hidden state and a cell state
        return (torch.rand(size=(self.num_layers * 2, batch_size, self.hidden_size)),) * 2

    def forward(self, input_data, hiddens):
        '''
        input_data : (seq_len, batchsize, input_dim)
        '''
        outputs, hiddens = self.lstm(input_data, hiddens)
        # outputs: (seq_len, batch_size, num_directions* hidden_size)
        pred = self.fc(outputs)
        pred = self.sigmoid(pred)
        return pred


def train(vad_net, inp, target, criterion, optimizer):
    """
    :param inp: (seq_len, batch_size, feat_dim)
    :param vad_net: the model
    :param target: (seq_len, batch_size)
    :return: prediction (0-1) and loss (float)
    """
    # print(id(vad_net))
    if inp.ndim == 2:
        inp.unsqueeze_(1).float()
    if target.ndim == 1:
        target.unsqueeze_(1).float()
    inp.to(device)
    target.to(device)
    hiddens = vad_net.init_hiddens(batch_size=1)
    hiddens = (hiddens[0].to(device), hiddens[1].to(device))
    optimizer.zero_grad()
    pred = torch.squeeze(vad_net(inp, hiddens))
    loss = criterion(pred, target.squeeze_())
    loss.backward()
    optimizer.step()
    return loss.item()


def evaluate(model, feat_path, lbl_dict):
    """
    :param model: the model to be evaluated
    :param feat_path: the path where validation set features are restored
    :param lbl_dict: a dict, wave id to frame-wise label
    :return: auc, eer, tpr, fpr on the validation set
    """
    all_pred = []
    all_lbls = []
    for audio_file in tqdm(os.listdir(feat_path)):
        audio_id = audio_file.split('.')[0]
        frames = np.load(os.path.join(feat_path, audio_file))

        inp = torch.unsqueeze(torch.from_numpy(frames).float(), 1).to(device)

        hiddens = model.init_hiddens(batch_size=1)
        hiddens = (hiddens[0].to(device), hiddens[1].to(device))

        pred = torch.squeeze(model(inp, hiddens))
        all_pred += pred.detach().cpu().numpy().tolist()
        all_lbls += lbl_dict[audio_id]
        assert len(all_pred) == len(all_lbls)

    auc, eer, fpr, tpr = get_metrics(all_pred, all_lbls)
    return auc, eer, fpr, tpr


def calc_metrics(model, feat_path, lbl_dict, L, thres=0.5):
    """
    Calculate J_A,J_S,J_E,J_B, VACC according to {Evaluating VAD for Automatic Speech Recognition, ICSP2014, Sibo Tong et al.}
    :param L: The adjust length in that paper
    :param thres: threshold. Frames with predicion higher than thres will be determined to be speech.
    :return: ACC(J_A), J_S(SBA), J_E(EBA), J_B(BP), VACC
    """
    all_pred = []
    all_lbls = []
    for audio_file in tqdm(os.listdir(feat_path)):
        audio_id = audio_file.split('.')[0]
        frames = np.load(os.path.join(feat_path, audio_file))

        inp = torch.unsqueeze(torch.from_numpy(frames).float(), 1).to(device)

        hiddens = model.init_hiddens(batch_size=1)
        hiddens = (hiddens[0].to(device), hiddens[1].to(device))

        pred = torch.squeeze(model(inp, hiddens))
        all_pred += pred.detach().cpu().numpy().tolist()
        all_lbls += lbl_dict[audio_id]
        assert len(all_pred) == len(all_lbls)
    all_pred = np.array(all_pred)
    all_lbls = np.array(all_lbls)
    all_pred[all_pred >= thres] = 1
    all_pred[all_pred < thres] = 0
    acc = (all_pred + all_lbls) % 2
    acc = (len(all_pred) - acc.sum()) / len(all_pred)

    R, M = 0, 0  # R for speech segments in label, M for from VAD
    Js = []
    Je = []
    for i in tqdm(range(len(all_pred) - 1)):
        if all_pred[i] == 0 and all_pred[i + 1] == 1:
            M += 1
        if all_lbls[i] == 0 and all_lbls[i + 1] == 1:
            R += 1  # speech segment begins
            begin_seg_lbl = all_lbls[i + 1:i + L + 1]
            begin_seg_pred = all_pred[i + 1:i + L + 1]
            js = ((begin_seg_pred + begin_seg_lbl) % 2).sum()
            Js.append((L - js) / L)
        if all_lbls[i] == 1 and all_lbls[i + 1] == 0:
            end_lbl = all_lbls[i - L + 1:i + 1]
            end_pred = all_pred[i - L + 1:i + 1]
            je = ((end_lbl + end_pred) % 2).sum()
            Je.append((L - je) / L)
    Js = sum(Js) / R
    Je = sum(Je) / R
    Jb = R * (Js + Je) / (2 * M)
    vacc = 4 / (1 / acc + 1 / Js + 1 / Je + 1 / Jb)

    return acc, Js, Je, Jb, vacc


def predict(args, model, feat_path, target_path):
    """
    :param model: vad model
    :param feat_path: features path of test data
    :param target_path: where test_label_task2.txt is going to be stored
    """
    model.eval()
    with open(os.path.join(target_path, 'test_label_task2.txt'), 'w') as f:
        for testfile in tqdm(os.listdir(feat_path)):
            test_id = testfile.split('.')[0]
            frames = np.load(os.path.join(feat_path, testfile))
            inp = torch.unsqueeze(torch.from_numpy(frames).float(), 1).to(device)

            hiddens = model.init_hiddens(batch_size=1)
            hiddens = (hiddens[0].to(device), hiddens[1].to(device))

            pred = torch.squeeze(model(inp, hiddens))

            line = prediction_to_vad_label(pred, args.win_len, args.win_hop, 0.5)
            f.write(test_id + " " + line + '\n')


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='Run LSTM for VAD')
    parser.add_argument('--fs', default=16000, type=int)
    parser.add_argument('--win_len', default=0.032, type=float)
    parser.add_argument('--win_hop', default=0.008, type=float)
    parser.add_argument('--hidden_size', default=32, type=int)
    parser.add_argument('--num_layers', default=1, type=int)
    parser.add_argument('--lr', default=0.001, type=float)
    parser.add_argument('--num_epoch', default=2, type=int)
    parser.add_argument('--report_interval', default=50, type=int)
    parser.add_argument('--stage', default=2, type=int)
    parser.add_argument('--L', default=5, type=int)  # adjust length in VACC calculation

    args = parser.parse_args()

    root = os.getcwd()
    dev_path = os.path.join(root, "wavs", "dev")
    train_path = os.path.join(root, "wavs", "train")
    data_path = os.path.join(root, r"data")
    fig_path = os.path.join(root, r"fig")
    test_path = os.path.join(root, "wavs", "test")
    feat_path = os.path.join(root, r"feats")
    dev_feat_path = os.path.join(feat_path, r'dev')
    train_feat_path = os.path.join(feat_path, r'train')
    test_feat_path = os.path.join(feat_path, r'test')

    if not os.path.exists(feat_path):
        os.mkdir(feat_path)
    for path in [fig_path, dev_feat_path, train_feat_path, test_feat_path]:
        if not os.path.exists(path):
            os.mkdir(path)

    if args.stage <= 0:
        print('stage 0: data preparation and feature extraction')
        dev_lbl_dict = read_label_from_file(path=os.path.join(data_path, r"dev_label.txt"))
        for wav in tqdm(os.listdir(dev_path)):
            wav_id = wav.split('.')[0]
            wav_file = os.path.join(dev_path, wav)
            wav_array, fs = librosa.load(wav_file, sr=args.fs)
            wav_framed, frame_len = preprocess.framing(wav_array, fs=args.fs, win_len=args.win_len,
                                                       win_hop=args.win_hop)
            frame_num = wav_framed.shape[0]
            assert frame_num >= len(dev_lbl_dict[wav_id])
            dev_lbl_dict[wav_id] += [0] * (frame_num - len(dev_lbl_dict[wav_id]))

            frame_energy = (wav_framed ** 2).sum(1)[:, np.newaxis]
            assert frame_num == len(dev_lbl_dict[wav_id])
            frame_mfcc = mfcc.mfcc(wav_array, fs=args.fs, win_len=args.win_len, win_hop=args.win_hop)
            frame_feats = np.concatenate((frame_energy, frame_mfcc), axis=1)
            np.save(os.path.join(dev_feat_path, wav_id + '.npy'), frame_feats)
        # if args.make_dev_lbl_dict:
        json_str = json.dumps(dev_lbl_dict)
        with open(os.path.join(data_path, r"dev_lbl_dict.json"), 'w') as json_file:
            json_file.write(json_str)

        train_lbl_dict = read_label_from_file(path=os.path.join(data_path, r"train_label.txt"))
        for wav in tqdm(os.listdir(train_path)):
            wav_id = wav.split('.')[0]
            wav_file = os.path.join(train_path, wav)
            wav_array, fs = librosa.load(wav_file, sr=args.fs)
            wav_framed, frame_len = preprocess.framing(wav_array, fs=args.fs, win_len=args.win_len,
                                                       win_hop=args.win_hop)
            frame_num = wav_framed.shape[0]
            # if args.make_train_lbl_dict:
            assert frame_num >= len(train_lbl_dict[wav_id])
            train_lbl_dict[wav_id] += [0] * (frame_num - len(train_lbl_dict[wav_id]))  # 补0

            frame_energy = (wav_framed ** 2).sum(1)[:, np.newaxis]
            assert frame_num == len(train_lbl_dict[wav_id])
            frame_mfcc = mfcc.mfcc(wav_array, fs=args.fs, win_len=args.win_len, win_hop=args.win_hop)
            frame_feats = np.concatenate((frame_energy, frame_mfcc), axis=1)
            np.save(os.path.join(train_feat_path, wav_id + '.npy'), frame_feats)
        json_str = json.dumps(train_lbl_dict)
        with open(os.path.join(data_path, r"train_lbl_dict.json"), 'w') as json_file:
            json_file.write(json_str)

        for wav in tqdm(os.listdir(test_path)):
            wav_id = wav.split('.')[0]
            wav_file = os.path.join(test_path, wav)
            wav_array, fs = librosa.load(wav_file, sr=args.fs)
            wav_framed, frame_len = preprocess.framing(wav_array, fs=args.fs, win_len=args.win_len,
                                                       win_hop=args.win_hop)
            frame_num = wav_framed.shape[0]

            frame_energy = (wav_framed ** 2).sum(1)[:, np.newaxis]
            frame_mfcc = mfcc.mfcc(wav_array, fs=args.fs, win_len=args.win_len, win_hop=args.win_hop)
            frame_feats = np.concatenate((frame_energy, frame_mfcc), axis=1)
            np.save(os.path.join(test_feat_path, wav_id + '.npy'), frame_feats)

    with open(os.path.join(data_path, r"dev_lbl_dict.json"), 'r') as f:
        dev_lbl_dict = json.load(f)
    with open(os.path.join(data_path, r"train_lbl_dict.json"), 'r') as f:
        train_lbl_dict = json.load(f)

    input_dim = 14  # 1(energy)+13(mfcc)
    hidden_size = args.hidden_size
    num_layers = args.num_layers
    vad_net = VADnet(input_dim, hidden_size=hidden_size, num_layers=num_layers).to(device)

    criterion = nn.BCELoss()
    optimizer = torch.optim.Adam(vad_net.parameters(), lr=args.lr)
    report_interval = args.report_interval
    interval_loss = 0

    if args.stage <= 1:
        print('stage 1: model training')
        # here we first implement the case where batch_size = 1
        plt.figure()
        plt.xlabel('FPR', fontsize=13)
        plt.ylabel("TPR", fontsize=13)
        plt.title('RoC curve', fontsize=15)
        plt.plot(np.linspace(0, 1, 1000), np.linspace(1, 0, 1000), '.', linewidth=0.5, markersize=3, color='cyan')

        for epoch in range(args.num_epoch):
            vad_net.train()
            for i, audio_file in tqdm(
                    enumerate(random.sample(os.listdir(train_feat_path), len(os.listdir(train_feat_path))))):
                audio_id = audio_file.split('.')[0]
                audio = np.load(os.path.join(train_feat_path, audio_file))
                inp = torch.from_numpy(audio).float().to(device)
                target = torch.tensor(train_lbl_dict[audio_id]).float().to(device)
                loss = train(vad_net, inp, target, criterion, optimizer)
                if i % report_interval == 0:
                    print("epoch = ", epoch, "batch n = ", i, " average loss = ", interval_loss / report_interval)
                    interval_loss = 0
                else:
                    interval_loss += loss
            vad_net.eval()
            auc, eer, fpr, tpr = evaluate(vad_net, dev_feat_path, dev_lbl_dict)

            plt.plot(fpr, tpr, '-.', linewidth=3,
                     label="epoch {}\nAUC={:.4f},EER={:.4f}".format(epoch, auc, eer))

            print('======> epoch {}, (auc, eer)={:.5f},{:.5f}'.format(epoch, auc, eer))
            torch.save(vad_net.state_dict(), os.path.join(root, f"epoch_{epoch}.pth"))
        plt.legend(fontsize=14, loc="lower center")
        plt.xlim([0, 1])
        plt.ylim([0, 1])
        plt.savefig(os.path.join(fig_path, 'roc_curve.png'), dpi=100)

    if args.stage <= 2:
        print('stage 2: model evaluating')

        vad_net.load_state_dict(torch.load(os.path.join(root, f"epoch_{args.num_epoch-1}.pth")))

        vis_sample(vad_net, dev_lbl_dict, dev_feat_path, fig_path, save=True)
        vis_sample(vad_net, dev_lbl_dict, dev_feat_path, fig_path, save=True)
        acc, Js, Je, Jb, vacc = calc_metrics(vad_net, dev_feat_path, dev_lbl_dict, L=args.L, thres=0.5)
        print('>=== ACC = {:.4f}'.format(acc))
        print('>=== SBA = {:.4f}'.format(Js))
        print('>=== EBA = {:.4f}'.format(Je))
        print('>=== BP = {:.4f}'.format(Jb))
        print('>=== VADD = {:.4f}'.format(vacc))

    if args.stage <= 3:
        print('stage 3: model predicting')
        vis_sample(vad_net, dev_lbl_dict, dev_feat_path, fig_path, save=True)
        vis_sample(vad_net, dev_lbl_dict, dev_feat_path, fig_path, save=True)
        predict(args, vad_net, test_feat_path, data_path)

    print('DONE!')
