

Our code is built on [BasicTS](https://github.com/zezhishao/BasicTS), an open-source standard time series forecasting benchmark.

We recommend you use [BasicTS](https://github.com/zezhishao/BasicTS) to find more baselines and more detailed comparisons.


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
