import concurrent.futures
from vad.func_vis_utils import *  # 所有可视化函数


def read_frame_store(filename, nw, inc, wave_path, target_path, winc=None):
    '''
    :param filename: wave file name end with .wav
    :param nw: width of frame (seconds)
    :param inc: shift of frame (seconds)
    :param winc: window function (optional)
    :return: None
    This function reads wave file, enframe and store it
    '''
    if not os.path.exists(target_path):
        os.mkdir(target_path)
    audio, fs = librosa.load(os.path.join(wave_path, filename), mono=True)
    # audio = filt_low(audio, fs, cut_freq=50)
    assert np.isfinite(audio).all()
    # print(filename)
    nw, inc = int(fs * nw), int(fs * inc)

    # 下面分帧的代码是参考CSDN博客的
    # from https://blog.csdn.net/luolinll1212/article/details/98940838
    signal_length = len(audio)  # 信号总长度
    if signal_length <= nw:  # 如果信号长度小于一帧的长度，则帧数定义为1
        nf = 1  # nf表示帧数量
    else:
        nf = int(np.ceil((1.0 * signal_length - nw + inc) / inc))  # 处理后，所有帧的数量，不要最后一帧
    pad_length = int((nf - 1) * inc + nw)  # 所有帧加起来总的平铺后的长度
    pad_signal = np.pad(audio, (0, pad_length - signal_length), 'constant')  # 0填充最后一帧
    indices = np.tile(np.arange(0, nw), (nf, 1)) + np.tile(np.arange(0, nf * inc, inc), (nw, 1)).T  # 每帧的索引
    indices = np.array(indices, dtype=np.int32)
    frames = pad_signal[indices]  # 得到帧信号, 用索引拿数据
    if winc is not None:
        win = np.tile(winc, (nf, 1))  # winc为窗函数
        frames = frames * win  # 返回加窗的信号
    frames = frames - np.average(frames)
    assert np.isfinite(frames).all()
    np.save(os.path.join(target_path, filename.split('.')[0] + '.npy'), frames)


def multipro_read_store(R, wavelist):
    with concurrent.futures.ProcessPoolExecutor() as executor:
        print('executing...')
        executor.map(R, wavelist)
    print("done!")


def pad_dict(dict_lbl, data_path, dev_data_path):
    for framed_data in os.listdir(dev_data_path):
        wave_id = framed_data.split('.')[0]
        nf = np.load(os.path.join(dev_data_path, framed_data)).shape[0]
        assert wave_id in dict_lbl.keys(), "wave ID not in dict"
        dict_lbl[wave_id] = np.pad(dict_lbl[wave_id], (0, max(nf - len(dict_lbl[wave_id]), 0))).tolist()
    return dict_lbl


def extract_energy_labels(dev_data_path, dev_lbl_dict):
    E_features = []
    labels = []
    print("extracting energy features and label")
    for wave_file in tqdm(os.listdir(dev_data_path)):
        audio = np.load(os.path.join(dev_data_path, wave_file))
        E = calc_energy_normalize(audio)
        E_features += E.tolist()
        wave_name = wave_file.split('.')[0]
        lbl = dev_lbl_dict[wave_name]
        labels += lbl

    E_features, labels = np.array(E_features), np.array(labels)
    return E_features, labels


def extract_zcr(dev_data_path):
    zcr_features = []
    print("extracting zcr features")
    for wave_file in tqdm(os.listdir(dev_data_path)):
        audio = np.load(os.path.join(dev_data_path, wave_file))
        zcr_features += ZCR(audio).tolist()
    zcr_features = np.array(zcr_features)
    return zcr_features


def custom_pred(sample_E, sample_zcr):
    thres_E = 0.00075
    thres_zcr = 0.5  # 高于它认为是语音
    pred = np.zeros_like(sample_E)
    for i in range(len(sample_E)):
        if sample_E[i] >= thres_E:
            pred[i] = 1
        elif sample_zcr[i] >= thres_zcr:
            pred[i] = 1
        else:
            continue
    return pred


if __name__ == "__main__":
    root = os.getcwd()
    dev_path = os.path.join(root, r"wavs\dev")
    dev_data_path = os.path.join(root, r"data\dev")
    data_path = os.path.join(dev_data_path, '..')
    fig_path = os.path.join(root, r"fig")
    test_path = os.path.join(root, r"wavs\test")
    test_data_path = os.path.join(root, r"data\test")

    for path in [dev_data_path, fig_path]:
        if not os.path.exists(path):
            os.mkdir(path)

    nw = 0.032
    inc = 0.008
    smooth_length = 25

    print('------begin reading dev audios----------')
    R = partial(read_frame_store, nw=nw, inc=inc, wave_path=dev_path, target_path=dev_data_path, winc=None)
    multipro_read_store(R, os.listdir(dev_path))
    # when have not split frames, do the above functions to ensure further processing

    if r"dev_label_dict.json" not in os.listdir(data_path):
        dict_lbl = read_label_from_file(os.path.join(data_path, r"dev_label.txt"))
        dev_lbl_dict = pad_dict(dict_lbl, data_path, dev_data_path)
        json_str = json.dumps(dev_lbl_dict)
        with open(os.path.join(data_path, r"dev_label_dict.json"), 'w') as json_file:
            json_file.write(json_str)
    else:
        with open(os.path.join(data_path, r"dev_label_dict.json"), 'r') as f:
            dev_lbl_dict = json.load(f)

    # Now dev_lbl_dict is a dictionary of **wave id** to **padded label** (0-1 string)

    E_features, labels = extract_energy_labels(dev_data_path, dev_lbl_dict)
    zcr_features = extract_zcr(dev_data_path)

    assert not np.isnan(E_features).any(), "NaN encountered"
    assert not np.isnan(zcr_features).any(), "NaN encountered"

    # do some visualizations below
    # vis_energy_stat(dev_data_path, fig_path, save=False)
    # vis_energy_silence_vs_speech(E_features, labels, fig_path, save=False)
    # vis_zcr_silence_vs_speech(labels, zcr_features, fig_path, save=False)
    # vis_zcr_speech_low_high_energy(E_features, labels, zcr_features, fig_path, thres=0.02, save=False)
    # vis_zcr_low_energy_speech_silence(E_features, labels, zcr_features, fig_path, thres=0.002, save=False)
    # vis_zcr_wave_sample(dev_lbl_dict, root, dev_data_path, fig_path, save=False)

    E_col = E_features[:, np.newaxis]
    zcr_col = zcr_features[:, np.newaxis]
    all_features = np.concatenate((E_col, zcr_col), axis=1)

    clf = DecisionTreeClassifier(max_depth=5, max_leaf_nodes=5,
                                 criterion='entropy'
                                 )
    clf = clf.fit(all_features, labels)
    score = clf.score(all_features, labels)
    print('model fit score = ', score, 'AUC,EER,ACC = ', get_metrics(clf.predict(all_features), labels),
          acc(clf.predict(all_features), labels))
    # vis_tree(clf, fig_path, img_name="entropy_5_5")

    # vis_sample(clf, dev_lbl_dict, 21, dev_data_path, fig_path, save=True)

    # vis_smooth_len(labels, clf, dev_data_path, fig_path, save=True)

    vis_eval(clf, all_features, labels, dev_data_path, fig_path, len_win=smooth_length, save=False)

    # Now it's time to read in test wave and do prediction, and output result as is required
    print('------------begin test prediction---------')

    R = partial(read_frame_store, nw=nw, inc=inc, wave_path=test_path, target_path=test_data_path, winc=None)
    multipro_read_store(R, os.listdir(test_path))

    with open(os.path.join(data_path, r"test_label_task1.txt"), 'w') as f:
        for test_file in tqdm(os.listdir(test_data_path)):
            test_audio = np.load(os.path.join(test_data_path, test_file))
            test_id = test_file.split('.')[0]
            E_test = calc_energy_normalize(test_audio)
            zcr_test = ZCR(test_audio)
            E_test = E_test[:, np.newaxis]
            zcr_test = zcr_test[:, np.newaxis]
            test_audio_features = np.concatenate((E_test, zcr_test), axis=1)
            test_label = clf.predict(test_audio_features)
            test_label = smooth(test_label, len_win=smooth_length)
            # -------begins test in 4.25 ----------
            test_str = str(test_label.tolist())
            assert "0, 1, 0" not in test_str
            # -----------end of test -------
            test_time_str = prediction_to_vad_label(test_label, frame_size=nw, frame_shift=inc, threshold=0.5)
            f.write(test_id + " " + test_time_str + '\n')
            # exit(0)

    print("--------done!---------")
