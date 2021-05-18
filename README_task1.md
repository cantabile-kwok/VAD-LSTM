## VAD-Project 1  Task1

*519030910352 郭奕玮*

#### Guidelines for running the code

* Lang: Python (3.6+ would be enough)
* About **7G** disk space。

* Prerequisite 3-party Python packages：`tqdm,numpy,sklearn,librosa `，`matplotlib`（if you want to see the visualizations）
* Intall `graphviz` if you want to visualize the decision tree （not necessary）
* 运行方式：直接运行`main.py`
  * 这份代码首先利用`concurrent`模块，**并行**地从`dev`文件夹中读取音频并分帧，在`data\dev`中存为NumPy格式的`.npy`文件以便之后要用可以直接读取。（训练完成后对`test`文件夹也是同样的操作）
  * 然后从读取`dev`文件夹下音频的标签，并构造一个`wave id->label`的字典，存储为`data\dev_label_dict.json`。
  * 随后提取特征、训练模型。代码中注释了很多过程中用到的可视化，函数名均以`vis_`开头。所有的可视化函数都有一个名为`save`的参数，控制是否保存图片（默认为True）。图片保存路径为`fig\`。
  * 最后提取测试集样本、分帧、保存，并对每一条音频输出预测结果，保存为`data\test_label.txt`

* 提交的压缩包中**不含所有音频数据**，否则太大，需要麻烦助教把音频复制到`dev`和`test`文件夹中。

* 提交的压缩包中**并不含有**分好帧的`dev,test`数据（即`data\dev,data\test`）以及`data\dev_label_dict.json`文件，因此`main.py`中**未注释**读取音频和存储分帧结果的函数。**如果需要重复运行，则在第二次及以后可以注释`main.py` 中的Line 116-117, 168-169**，for warm start。

#### Project Structure

*运行代码之后*：（斜体表示程序会创建）

vad (*root*)

├──data

│     ├──*dev* (*dev中的音频分帧之后的NumPy文件*)

│     ├──*test* (*test中的音频分帧之后的NumPy文件*)

│ 	├──dev_label.txt, train_label.txt

│ 	├──*dev_label_dict.json* （存储dev中的音频和对应label的字典）

│ 	└──***test_label_task1.txt*** (*预测结果*)

├──*fig* （所有可视化的结果）

├──latex, wavs文件夹

├──\_\_init\_\_.py, evaluate.py, vad_utils.py

├──**main.py** (主函数)

├──**func_vis_utils.py** (包括所有可视化函数，以及主函数中调用的其他帮助函数)

├──**519030910352-郭奕玮.pdf **(Report)

└──README.md，README.html

#### Other Comments

* 更改了vad_utils.py 中的`prediction_to_vad_label`函数的精度，保存三位小数。
* evaluate.py中，增加了`get_metrics`函数的返回值，让其返回了`fpr,tpr`。

#### Reference

* 仅在分帧的部分参考了https://blog.csdn.net/luolinll1212/article/details/98940838