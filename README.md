# VAD Project Task2: VAD Based on BiLSTM and MFCC

*cantabile-kwok*

### Code Environment & Required Packages

Code tested on the following environments:

* Linux 3.10.0
* CUDA 10.1
* Python 3.6.13

Required Python Packages:

* PyTorch 1.4.0
* NumPy 1.19.2
* sklearn 0.24.2
* **spafe** 0.1.0
* librosa 0.8.0
* matplotlib 3.3.4

Note: **spafe** may be a niche package. Use `pip install spafe` to install it (requires SciPy and librosa). See https://spafe.readthedocs.io/en/latest/ for document of this package.

---

### Guidelines for Running the Code

Similar to ESPNet recipes, we have explicit **stages** for running the code.

* **stage 0**: Preparing data and extracting feature. After this stage, features are stored in `feats/`
* **stage 1**: Model training. After this stage, the models are stored as `epoch_{}.pth`. The RoC curve is stored in `fig/`
* **stage 2**: Model evaluating. 
* **stage 3**: Model predicting (generates `test_label_task2.txt`)

For example, to start from raw wave files, use

```
python main.py --stage 0
```

Note that his stage only has to be run once.

Other arguments include:

* `--fs`: sampling rate
* `--win_len, --win_hop` frame size and shift in seconds
* `--hidden_size, --num_layers`: LSTM hyperparameters
* `--lr`: learning rate
* `--num_epoch`: number of epochs
* `--report_interval`: number of batches of each report. E.g. if set to 50, then training loss will be printed to console every 50 batches.
* `--L`: adjust length in evaluating VACC when calculating SBA, EBA, according to *Evaluating VAD for Automatic Speech Recognition, ICSP2014, Sibo Tong et al.* 

Note: The number of epochs is set to 2 by default. We use the last epoch to evaluate and predict after training. If you want to specify this, modify line **357**.

---

### Project Structure

vad (*root*)

├──data

│ 	├──dev_label.txt, train_label.txt

│ 	├──*dev_label_dict.json* （存储dev中的音频和对应label的字典）

│ 	├──*train_label_dict.json* （同上）

│ 	└──***test_label_task2.txt*** (*预测结果*)

├──**feats**

│ 	├──dev

│ 	├──test

│ 	└──train

├──*fig* （可视化的结果，包括一些样本的预测值和真值对比、RoC曲线）

├──latex, wavs文件夹

├──\_\_init\_\_.py, evaluate.py, vad_utils.py

├──**main.py** (主函数)

├──epoch_0.pth, epoch_1.pth等（保存模型）

├──script (可sbatch提交之)

└──README.md，README.html

