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
from vad.vad_utils import parse_vad_label, prediction_to_vad_label, read_label_from_file
from vad.evaluate import compute_eer, get_metrics
import re


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
    optimizer.zero_grad()
    pred = torch.squeeze(vad_net(inp, hiddens))
    loss = criterion(pred, target.squeeze_())
    loss.backward()
    optimizer.step()
    return pred.squeeze_(), loss.item()


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
    parser.add_argument('--stage', default=0, type=int)

    args = parser.parse_args()

    root = os.getcwd()
    dev_path = os.path.join(root, r"wavs\dev")
    train_path = os.path.join(root, r"wavs\train")
    data_path = os.path.join(root, r"data")
    fig_path = os.path.join(root, r"fig")
    test_path = os.path.join(root, r"wavs\test")
    feat_path = os.path.join(root, r"feats")
    dev_feat_path = os.path.join(feat_path, r'dev')
    train_feat_path = os.path.join(feat_path, r'train')
    test_feat_path = os.path.join(feat_path, r'test')

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    if not os.path.exists(feat_path):
        os.mkdir(feat_path)

    if args.stage <= 0:
        print('stage 0: data preparation and feature extraction')
        # if args.make_dev_lbl_dict or args.compute_dev_feats:
        dev_lbl_dict = read_label_from_file(path=os.path.join(data_path, r"dev_label.txt"))
        for wav in tqdm(os.listdir(dev_path)):
            wav_id = wav.split('.')[0]
            wav_file = os.path.join(dev_path, wav)
            wav_array, fs = librosa.load(wav_file, sr=args.fs)
            wav_framed, frame_len = preprocess.framing(wav_array, fs=args.fs, win_len=args.win_len,
                                                       win_hop=args.win_hop)
            frame_num = wav_framed.shape[0]
            # if args.make_dev_lbl_dict:
            assert frame_num >= len(dev_lbl_dict[wav_id])
            dev_lbl_dict[wav_id] += [0] * (frame_num - len(dev_lbl_dict[wav_id]))  # 补0

            # if args.compute_dev_feats:
            frame_energy = (wav_framed ** 2).sum(1)[:, np.newaxis]
            assert frame_num == len(dev_lbl_dict[wav_id])
            frame_mfcc = mfcc.mfcc(wav_array, fs=args.fs, win_len=args.win_len, win_hop=args.win_hop)
            frame_feats = np.concatenate((frame_energy, frame_mfcc), axis=1)
            np.save(os.path.join(dev_feat_path, wav_id + '.npy'), frame_feats)
        # if args.make_dev_lbl_dict:
        json_str = json.dumps(dev_lbl_dict)
        with open(os.path.join(data_path, r"dev_lbl_dict.json"), 'w') as json_file:
            json_file.write(json_str)

        # if args.make_train_lbl_dict or args.compute_train_feats:
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

            # if args.compute_train_feats:
            frame_energy = (wav_framed ** 2).sum(1)[:, np.newaxis]
            assert frame_num == len(train_lbl_dict[wav_id])
            frame_mfcc = mfcc.mfcc(wav_array, fs=args.fs, win_len=args.win_len, win_hop=args.win_hop)
            frame_feats = np.concatenate((frame_energy, frame_mfcc), axis=1)
            np.save(os.path.join(train_feat_path, wav_id + '.npy'), frame_feats)
        # if args.make_dev_lbl_dict:
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
            # if args.make_test_lbl_dict:

            # if args.compute_test_feats:
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
    report_interval = 50

    if args.stage <= 1:
        print('stage 1: model training')
        # here we first implement the case where batch_size = 1
        vad_net.train()
        for i, audio_file in tqdm(enumerate(os.listdir(train_feat_path))):
            audio_id = audio_file.split('.')[0]
            audio = np.load(os.path.join(train_feat_path, audio_file))
            inp = torch.from_numpy(audio).float().to(device)
            target = torch.tensor(train_lbl_dict[audio_id]).float().to(device)
            pred, loss = train(vad_net, inp, target, criterion, optimizer)
            # print(id(vad_net))
            if i % report_interval == 0:
                print("batch n = ", i, " loss = ", loss)
    if args.stage <= 2:
        torch.save(vad_net.state_dict(), os.path.join(root, r"epoch_1.pth"))
