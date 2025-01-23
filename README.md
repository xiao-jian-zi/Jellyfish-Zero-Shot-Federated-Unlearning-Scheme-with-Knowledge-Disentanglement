# Jellyfish-Zero-Shot-Federated-Unlearning-Scheme-with-Knowledge-Disentanglement
PyTorch code for the paper "Jellyfish: Zero-Shot Federated Unlearning Scheme with Knowledge Disentanglement".
## 1 Requirements
We recommended the following dependencies.

* python==3.8
* pytorch==1.11.0
* torchvision==0.12.0
* numpy==1.22.4

For more recommended dependencies, please refer to the file `requirements.txt`.

## 2 How to use
### 2.1 Training models
We use the CIFAR-10 dataset as an example.

Run `train_cifar10.py` to obtain the trained models: 

```bash
python train_cifar10.py
```
 You can specify the number of training epochs by setting the `epoch_num` in the file. If necessary, please modify the path to the dataset; otherwise, the code will automatically download the dataset to the default path when executed.

### 2.2 Unlearning
Run `cifar10-unlearn.py` to perform the unlearning process: 
```bash
python cifar10-unlearn.py
```
To customize the unlearning process, please make the following adjustments in the file:
* Modify the `dataset-path` to the path of your local dataset.
* Modify the `unlearn_epochs` value to specify the number of unlearning epochs.
* Modify the `disentangle_epochs` value to specify the number of disentangle epochs.
* Modify the `Thr` value to ensure the degree of disentanglement.
* Modify the `retain_percentage` value to retain the knowledge about the remaining data in the model.

  
These configurations will allow you to tailor the unlearning process to your specific requirements. Ensure that the values are correctly set before running the training script to avoid any errors or unexpected results.

### 2.3 Zero-Shot
Run `cifar10-zeroshot.py` to perform the visible evaluation process: 

```bash
python cifar10-zeroshot.py
```
To customize the evaluation process, please make the following adjustments in the file:
* Modify the `dataset-path` to the path of your local dataset.
* Modify the `noise_epochs` value to specify the number of noise training epochs.
* Modify the `unlearn_epochs` value to specify the number of unlearning epochs.
* Modify the `disentangle_epochs` value to specify the number of disentangle epochs.
* Modify the `Thr` value to ensure the degree of disentanglement.
* Modify the `retain_percentage` value to retain the knowledge about the remaining data in the model.


These configurations will allow you to tailor the zero-shot unlearning to your specific requirements. Ensure that the values are correctly set before running the training script to avoid any errors or unexpected results.

## 3 Code Reference
For detailed code explanations and best practices, please refer to
* [https://github.com/akamaster/pytorch_resnet_cifar10](https://github.com/akamaster/pytorch_resnet_cifar10)
* [https://github.com/IMoonKeyBoy/The-Right-to-be-Forgotten-in-Federated-Learning-An-Efficient-Realization-with-Rapid-Retraining](https://github.com/IMoonKeyBoy/The-Right-to-be-Forgotten-in-Federated-Learning-An-Efficient-Realization-with-Rapid-Retraining)
* [https://github.com/vikram2000b/bad-teaching-unlearning](https://github.com/vikram2000b/bad-teaching-unlearning)
* [https://github.com/ayushkumartarun/zero-shot-unlearning](https://github.com/ayushkumartarun/zero-shot-unlearning)
* [https://github.com/vikram2000b/Fast-Machine-Unlearning](https://github.com/vikram2000b/Fast-Machine-Unlearning)
