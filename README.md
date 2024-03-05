# STEMLP: A Spatial-Temporal Embedding Multi-Layer Perceptron for Traffic Flow Forecasting

Code for our paper: "STEMLP: A Spatial-Temporal Embedding Multi-Layer Perceptron for Traffic Flow Forecasting".

Our code is built on [BasicTS](https://github.com/zezhishao/BasicTS), an open-source standard time series forecasting benchmark.

We recommend you use [BasicTS](https://github.com/zezhishao/BasicTS) to find more baselines and more detailed comparisons.

> The linear framework-based traffic flow forecasting model enhances the spatial-temporal contextualization representation of the data through temporal embedding and spatial embedding, leading to a comparable or even more predictive performance than Spatial-Temporal Graph Neural Networks (STGNNs). Meanwhile, it is feasible to significantly reduce model complexity and computational overhead. However, the spatial-temporal embedding methods in such models do not provide enough consideration on the differences in time periodicity across datasets, as well as the static road network topology and dynamic global spatial relationships. Thus, these models generally suffer from poor accuracy and dynamic adaptivity of spatial-temporal contextualization representations. To address the above problems and challenges, this paper proposes a Spatial-Temporal Embedding Multi-Layer Perceptron (STEMLP) model for traffic flow forecasting. Firstly, we utilize the Fourier transform to decompose the traffic flow data for obtaining the frequency component composition of the periodic signals hidden in the time domain. This process can improve the accuracy and dynamic adaptability of temporal embedding expression compared to the specific periodicity patterns such as daily, weekly, and monthly, which are used in existing methods. Secondly, constructing spatial embedding based on the normalized Laplace eigenvector matrices of the predefined graph and adaptive graph can better model spatial relationships between road network nodes. Finally, we use mixed joint Multi-Layer Perceptron modules to integrate the spatial and temporal embedding, achieving a better learned spaital-temporal patterns for efficient and accurate traffic flow forecasting. Experimental results on four real-world datasets show that the method proposed in this paper outperforms the benchmark model in terms of prediction accuracy and computational efficiency.

##Table of Contents

```text
basicts   --> The BasicTS, which provides standard pipelines for training MTS forecasting models. Don't worry if you don't know it, because it doesn't prevent you from understanding STEMLP's code.

datasets  --> Raw datasets and preprocessed data

figures   --> Some figures used in README.

scripts   --> Data preprocessing scripts.

basicts/archs/arch_zoo/stemlp_arch/      --> The implementation of STEMLP.

examples/STEMLP/STEMLP_${DATASET_NAME}.py    --> Training configs.
```

Replace `${DATASET_NAME}` with one of `PEMS04`, `PEMS07`, `PEMS08` and `TFA`.

## 💿Requirements

The code is built based on Python 3.9, PyTorch 1.10.0, and [EasyTorch](https://github.com/cnstark/easytorch).
You can install PyTorch following the instruction in [PyTorch](https://pytorch.org/get-started/locally/). For example:

```bash
pip install torch==1.10.0+cu111 torchvision==0.11.0+cu111 torchaudio==0.10.0 -f https://download.pytorch.org/whl/torch_stable.html
```

After ensuring that PyTorch is installed correctly, you can install other dependencies via:

```bash
pip install -r requirements.txt
```

## 📦 Data Preparation

### **Download Data**

You can download all the raw datasets at [Google Drive](https://drive.google.com/drive/folders/1IaN5YdTHuvR17m_huE7iSIRM4Nk71903?usp=sharing) and unzip them to `datasets/raw_data/`.

### **Data Preprocessing**

```bash
cd /path/to/your/project
python scripts/data_preparation/${DATASET_NAME}/generate_training_data.py
```

Replace `${DATASET_NAME}` with one of `PEMS04`, `PEMS07`, `PEMS08`, `TFA` or any other supported dataset. The processed data will be placed in `datasets/${DATASET_NAME}`.

Or you can pre-process all datasets by.

```bash
cd /path/to/your/project
bash scripts/data_preparation/all.sh
```

## 🎯 Train STEMLP

```bash
python examples/run.py -c examples/STEMLP/STEMLP_${DATASET_NAME}.py --gpus '0'
```

Replace `${DATASET_NAME}` with one of `PEMS04`, `PEMS07`, `PEMS08`, and `TFA`, *e.g.*,

```bash
python examples/run.py -c examples/STEMLP/STEMLP_PEMS04.py --gpus '0'
```

## Citing

```bibtex
@inproceedings{10.1145/3511808.3557702,
author = {Shao, Zezhi and Zhang, Zhao and Wang, Fei and Wei, Wei and Xu, Yongjun},
title = {Spatial-Temporal Identity: A Simple yet Effective Baseline for Multivariate Time Series Forecasting},
year = {2022},
booktitle = {Proceedings of the 31st ACM International Conference on Information & Knowledge Management},
pages = {4454–4458},
location = {Atlanta, GA, USA}
}
```