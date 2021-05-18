import numpy as np
import wave
import os
import librosa
from scipy.io import wavfile
import concurrent.futures
from tqdm import tqdm
from vad.vad_utils import parse_vad_label, prediction_to_vad_label, read_label_from_file
from vad.evaluate import compute_eer, get_metrics
import matplotlib.pyplot as plt
import seaborn as sns
import json
import concurrent.futures
from functools import partial
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import train_test_split
from sklearn.tree import export_graphviz
from sklearn import tree
import scipy.signal as signal
import re

'''
def filt_low(audio, fs, cut_freq=50):
    # cut_freq = 50.0  # 截止频率Hz
    b, a = signal.butter(2, 2 * cut_freq / fs, "highpass", output="ba")
    res= signal.filtfilt(b, a, audio)
    assert np.isfinite(res).all()
    return res
'''


def ZCR(frames):
    """
    :param frames: the input framed audio (frame num*frame width)
    :return: an array of length=frame number,each element a float number of {zero crossing num}/{frame width}
    """
    nw = frames.shape[1]
    assert nw != 0, "nw==0"
    assert np.isfinite(frames).all(), ".."

    frames_shift = np.concatenate(
        (np.zeros(shape=(frames.shape[0], 1)), frames[:, 0:-1]), axis=1)
    tmp = frames * frames_shift
    assert np.isfinite(tmp).all(), 'not finite'
    tmp = np.sum(tmp < 0, axis=1)
    # print(tmp)
    assert np.isfinite(tmp).all(), 'not finite'
    return tmp / nw


def calc_energy_normalize(frames):
    """
    :param frames: the input framed audio (frame num*frame width)
    :return: an array of length=frame number,each element a float number of normalized energy
    """
    assert np.isfinite(frames).all(), ".."
    E = np.sum(frames ** 2, axis=1)
    E = (E - np.min(E)) / (np.max(E) - np.min(E))  # normalize energy
    return E


def smooth(pred, len_win: int):
    """
    :param pred: 样本预测结果
    :param len_win: 平滑窗长
    :return: 平滑后的预测结果
    """
    assert len_win % 2 == 1, "smooth length must be odd"
    N = len(pred)
    kernel = np.ones(len_win)
    kernel = kernel / np.sum(kernel)
    res = np.convolve(pred, kernel)[(len_win - 1) // 2: -(len_win - 1) // 2]
    # res[res >= 0.5] = 1
    # res[res < 0.5] = 0
    for i in range(len(res)):
        if res[i] >= 1 - 0.5 * min(min(i, N - i), len_win) / len_win:
            res[i] = 1
        else:
            res[i] = 0
    res_list = res.tolist()
    res_str = str(res_list)
    res_str = re.sub("0, 1, 0", "0, 0, 0", res_str)  # this is to insure we don't get 0,1,0 in pred label
    res = np.array((eval(res_str)))
    return res


def vis_energy_stat(dev_data_path, fig_path, save=True):
    E_min_list, E_max_list, E_avg_list = [], [], []
    for wave_file in os.listdir(dev_data_path):
        audio = np.load(os.path.join(dev_data_path, wave_file))
        E = np.sum(audio ** 2, axis=1)
        assert E.size == audio.shape[0], "wrong power calculation"
        E_min_list.append(np.min(E))
        E_max_list.append(np.max(E))
        E_avg_list.append(np.average(E))
    fig = plt.figure(figsize=(15, 4))
    sns.histplot(E_min_list, label='min frame energy', stat="probability", kde=True, color='g')
    plt.xlabel('min frame energy')
    plt.xlim([-0.005, 0.100])
    # sns.distplot(E_min_list,label='min frame power',kde=True,color='g')
    plt.twiny()
    sns.histplot(E_max_list, label='max frame energy', stat="probability", kde=True, color='r')
    sns.histplot(E_avg_list, label='average frame energy', stat="probability", kde=True, color='b')
    plt.xlim([-10, 200])
    plt.title('frame energy distribution of dev audio')
    plt.xlabel('max or average frame energy')
    fig.legend(bbox_to_anchor=(0, 0, 0.8, 0.5))
    if save:
        fig.savefig(fig_path + r"\dev_energy_dist.png", dpi=150, bbox_inches='tight')
    plt.show()


def vis_energy_silence_vs_speech(E_features, labels, fig_path, save=True):
    true_index = (labels == 1)
    false_index = (labels == 0)
    energy_target = E_features[true_index]
    energy_nontgt = E_features[false_index]

    fig = plt.figure(figsize=(9, 4))
    plt.subplot(121)
    sns.histplot(energy_target, bins=30, color='cyan', stat='probability')
    plt.xlabel('energy in speech')
    plt.subplot(122)
    sns.histplot(energy_nontgt, bins=30, color='r', stat='probability')
    plt.xlabel('energy in silence')

    # plt.hist(energy_target,bins=30)
    fig.suptitle('distribution of normalized energy in speech/silence')
    if save:
        fig.savefig(fig_path + r"\dev_energy_dist_speech_silence.png", dpi=150, bbox_inches='tight')
    plt.show()


def vis_zcr_silence_vs_speech(labels, zcr_features, fig_path, save=True):
    true_index = (labels == 1)
    false_index = (labels == 0)
    zcr_target = zcr_features[true_index]
    zcr_nontgt = zcr_features[false_index]

    fig = plt.figure(figsize=(9, 4))
    # plt.subplot(121)
    sns.histplot(zcr_target, bins=30, color='cyan', stat='probability', kde=True, label="Speech")
    plt.xlabel('zcr in speech')
    # plt.subplot(122)
    sns.histplot(zcr_nontgt, bins=30, color='red', stat='probability', alpha=0.5, kde=True, label='Silence')
    plt.xlabel('zero crossing rate', fontsize=12)

    # plt.hist(energy_target,bins=30)
    fig.suptitle('distribution of zero crossing rate in speech/silence')
    plt.legend()
    if save:
        fig.savefig(fig_path + r"\dev_zcr_dist_speech_silence.png", dpi=150, bbox_inches='tight')
    plt.show()


def vis_zcr_speech_low_high_energy(E_features, labels, zcr_features, fig_path, thres=0.02, save=True):
    true_index = (labels == 1)
    energy_target = E_features[true_index]
    zcr_target = zcr_features[true_index]

    high_ind = energy_target > thres
    low_ind = energy_target <= thres

    high_E_zcr = zcr_target[high_ind]
    low_E_zcr = zcr_target[low_ind]

    fig = plt.figure(figsize=(9, 5))

    sns.histplot(high_E_zcr, bins=30, color='cyan', stat='probability', kde=True, label=f"Energy>{thres}")
    sns.histplot(low_E_zcr, bins=30, color='red', stat='probability', alpha=0.5, kde=True, label=f'Energy<={thres}')

    plt.xlabel('zero crossing rate', fontsize=14)

    # plt.hist(energy_target,bins=30)
    fig.suptitle('distribution of zcr in high/low energy speech', fontsize=16)
    plt.legend(fontsize=12)
    if save:
        fig.savefig(fig_path + r"\dev_zcr_dist_energy.png", dpi=150, bbox_inches='tight')
    plt.show()


def vis_zcr_low_energy_speech_silence(E_features, labels, zcr_features, fig_path, thres=0.002, save=True):
    low_energy_ind = E_features <= thres
    lbl_low_eng = labels[low_energy_ind]
    zcr_low_eng = zcr_features[low_energy_ind]
    zcr_low_eng_target = zcr_low_eng[lbl_low_eng == 1]
    zcr_low_eng_nontgt = zcr_low_eng[lbl_low_eng == 0]
    fig = plt.figure(figsize=(16, 6))
    sns.histplot(zcr_low_eng_target, bins=30, kde=True, stat="probability", color='cyan',
                 label=f'speech zcr in low energy frames')
    sns.histplot(zcr_low_eng_nontgt, bins=30, kde=True, stat="probability", color='r', alpha=0.3,
                 label=f'silence zcr in low energy frames')
    plt.xlabel('zero crossing rate', fontsize=15)
    plt.xticks(fontsize=15)
    plt.ylabel('probability', fontsize=15)
    plt.yticks(fontsize=15)
    plt.title(f"distribution of zero crossing rate in low energy frames\n(standardized energy<={thres})", fontsize=20)
    plt.legend(fontsize=16)
    if save:
        fig.savefig(fig_path + r'\dist_zcr_low_energy.png', dpi=150)
    plt.show()


def vis_zcr_wave_sample(dev_lbl_dict, root, dev_data_path, fig_path, save=True):
    sample_wave = os.listdir(dev_data_path)[np.random.randint(len(os.listdir(dev_data_path)))]
    sample_zcr = ZCR(np.load(dev_data_path + '\\' + sample_wave))
    sample_id = sample_wave.split('.')[0]
    sample_waveform, _ = librosa.load(os.path.join(root, "wavs", "dev", sample_id + '.wav'))

    fig = plt.figure(figsize=(10, 6))
    plt.subplot(211)
    plt.plot(sample_zcr, label='zero crossing rate')
    plt.xlabel('frame', fontsize=14)
    # plt.subplot(312)
    plt.plot(dev_lbl_dict[sample_id], label='ground truth label')
    plt.legend(fontsize=12)
    plt.subplot(212)
    plt.plot(sample_waveform)
    plt.xlabel('time (sampled)', fontsize=14)
    plt.subplots_adjust(hspace=0.5)
    fig.suptitle(f"wave form and zcr plot of wave {sample_id}", fontsize=14)
    if save:
        fig.savefig(fig_path + r"\zcr_eg.png", dpi=100)
    plt.show()


def vis_tree(clf, fig_path, img_name):
    name = os.path.join(fig_path, img_name + ".dot")
    with open(name, 'w') as f:
        f = export_graphviz(clf, feature_names=['standardized energy', 'zero crossing rate'], out_file=f,
                            class_names=['silence', 'speech']
                            , rounded=True,
                            precision=6
                            )


def vis_sample(model, dev_lbl_dict, smooth_len, dev_data_path, fig_path, save=True):
    sample_path = os.listdir(dev_data_path)[np.random.randint(len(os.listdir(dev_data_path)))]
    sample_id = sample_path.split('.')[0]

    sample = np.load(os.path.join(dev_data_path, sample_path))
    sample_E = calc_energy_normalize(sample)
    sample_zcr = ZCR(sample)
    sample_features = np.concatenate((sample_E[:, np.newaxis], sample_zcr[:, np.newaxis]), axis=1)
    sample_label = dev_lbl_dict[sample_id]

    sample_pred = model.predict(sample_features)
    sample_pred_sm = smooth(sample_pred, smooth_len)
    auc_nosmooth, eer_nosmooth, _, _ = get_metrics(sample_pred, sample_label)
    auc_smooth, eer_smooth, _, _ = get_metrics(sample_pred_sm, sample_label)

    plt.figure(figsize=(8, 4))
    plt.plot(sample_label, c='r', label='sample label')
    plt.plot(sample_pred + 0.02, c='g', linestyle='-.',
             label='prediction with no smooth\n auc,eer={:.4f}, {:.4f}'.format(auc_nosmooth, eer_nosmooth))
    plt.plot(sample_pred_sm + 0.04, c='cyan', linestyle='-.',
             label='prediction after smooth\n auc,eer={:.4f}, {:.4f}'.format(auc_smooth, eer_smooth))
    plt.title(f'sample label and prediction (smooth len = {smooth_len})', fontsize=15)
    plt.legend(fontsize=10)
    plt.xlabel('frame', fontsize=12)
    if save:
        plt.savefig(fig_path + r'\\' + "sample_pred_smooth.png", dpi=120)
    plt.show()


def vis_smooth_len(labels, model, dev_data_path, fig_path, save=True):
    print("-------visualizing smooth length---------")
    frame_feature_dict = {}
    for frames in os.listdir(dev_data_path):
        audio_id = frames.split(".")[0]
        audio = np.load(os.path.join(dev_data_path, frames))
        E_frame = calc_energy_normalize(audio)
        zcr_frame = ZCR(audio)
        frame_features = np.concatenate((E_frame[:, np.newaxis], zcr_frame[:, np.newaxis]), axis=1)
        frame_feature_dict[audio_id] = frame_features

    auc_list = []
    eer_list = []
    len_list = np.linspace(3, 99, 49)
    for len_win in tqdm(len_list):
        smoothed_pred = []
        for frames in os.listdir(dev_data_path):
            audio_id = frames.split(".")[0]
            frame_pred = smooth(model.predict(frame_feature_dict[audio_id]), len_win=int(len_win)).tolist()
            smoothed_pred += frame_pred
        auc, eer, _, _ = get_metrics(smoothed_pred, labels)
        auc_list.append(auc)
        eer_list.append(eer)

    auc_max = max(auc_list)
    auc_max_ind = np.argmax(auc_list)
    auc_max_len = len_list[auc_max_ind]
    eer_min = min(eer_list)
    eer_min_ind = np.argmin(eer_list)
    eer_min_len = len_list[eer_min_ind]

    from matplotlib.pyplot import MultipleLocator
    x_major_locator = MultipleLocator(3)
    y_major_locator = MultipleLocator(0.005)
    fig, ax = plt.subplots(ncols=1, nrows=1, figsize=(9, 6))
    ax.plot(len_list, auc_list, label='AUC')
    ax.xaxis.set_major_locator(x_major_locator)
    ax.yaxis.set_major_locator(y_major_locator)
    ax.set_ylabel('AUC value')
    ax1 = ax.twinx()
    ax1.plot(len_list, eer_list, label='EER', c='g')
    ax1.set_ylabel('EER value')
    fig.legend(fontsize=14)
    ax.set_xlabel('smooth kernel length', fontsize=12)
    fig.suptitle(f"AUC max = {auc_max} at len = {auc_max_len}\n EER min = {eer_min} at len = {eer_min_len}",
                 fontsize=15)
    if save:
        plt.savefig(fig_path + '\\smooth_len_plot.png', dpi=100)
    fig.show()


def vis_eval(clf, all_features, labels, dev_data_path, fig_path, len_win, save=True):
    Ypred_all = clf.predict(all_features)
    auc, eer, fpr, tpr = get_metrics(Ypred_all, labels)
    plt.figure()
    plt.plot(fpr, tpr, '-.', linewidth=3,
             label="no smooth\n AUC={:.4f},EER={:.4f},Acc={:.4f}".format(auc, eer, clf.score(all_features, labels)))
    plt.xlabel('FPR', fontsize=13)
    plt.ylabel("TPR", fontsize=13)
    plt.title('RoC curve', fontsize=15)
    plt.plot(np.linspace(0, 1, 1000), np.linspace(1, 0, 1000), '.', linewidth=0.5, markersize=3, color='cyan')

    smoothed_pred = []
    for audio_name in tqdm(os.listdir(dev_data_path)):
        audio = np.load(os.path.join(dev_data_path, audio_name))
        audio_E = calc_energy_normalize(audio)
        audio_zcr = ZCR(audio)
        audio_feat = np.concatenate((audio_E[:, np.newaxis], audio_zcr[:, np.newaxis]), axis=1)
        audio_pred = clf.predict(audio_feat)
        # audio_pred = smooth(audio_pred, len_win=len_win)
        audio_pred = smooth(audio_pred, len_win=len_win)
        smoothed_pred += audio_pred.tolist()

    auc_sm, eer_sm, fpr_sm, tpr_sm = get_metrics(smoothed_pred, labels)
    plt.plot(fpr_sm, tpr_sm, '-.', linewidth=3,
             label="after smooth\n AUC={:.4f},EER={:.4f},Acc={:.4f}".format(auc_sm, eer_sm, acc(smoothed_pred, labels)))
    plt.legend(fontsize=14, loc="lower center")
    plt.xlim([0, 1])
    plt.ylim([0, 1])
    if save:
        plt.savefig(os.path.join(fig_path, r"roc_curve.png"), dpi=120)
    plt.show()


def acc(pred, labels):
    '''
    :param pred: 预测结果（0，1）
    :param labels: 真实标签
    :return: Accuracy of prediction, CORRECT/LENGTH
    '''
    n = len(pred)
    assert n == len(labels), 'pred and labels have different length'
    return (n - np.sum((pred + labels) % 2)) / n
