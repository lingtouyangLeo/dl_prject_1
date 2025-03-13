# Modified ResNet for CIFAR-10 Classification
This project implements a modified ResNet architecture for image classification on the CIFAR-10 dataset using Pytorch. The model is optimized to stay within 5 million parameters while achieving accuracy of 94.06%.

---

## Project Structure
```
📦 dl_project_1
├── 📂 data                  # Contains CIFAR-10 dataset
├── 📂 old_files             # Archive of old versions
├── 📂 src                   # Source code: model, training, and inference
│   ├── 📄 train.py          # Training script
│   ├── 📄 infer.py          # Inference script
│   ├── 📄 resnet_variant.py # Modified ResNet architecture
│   ├── 📄 utils.py          # Helper functions
│   └── 📂 __pycache__/      # Python cache files
├── 📄 .gitignore            # Files ignored by Git
├── 📄 LICENSE               # Project license
├── 📄 README.md             # Project documentation
├── 📄 submission_corrected.csv # Corrected submission results
├── 📄 training_logs.pth     # Saved PyTorch training logs
├── 📄 training_loss.png     # Training loss curve
├── 📄 validation_accuracy.png # Validation accuracy curve
├── 📄 log.txt               # Training log with accuracy/loss per epoch
├── 📄 miniproject_spring25.pdf # Project documentation
└── 📄 resnetVariant_best.pth # Best saved model checkpoint

```
---

## Prepare Dataset
The CIFAR-10 dataset should be downloaded and placed inside the data/ directory in the form of batch files.

---

## Train the Model
To train the ResNet model, run:
```bash
python src/train.py
```
This will train the model for 300 epochs and save the best checkpoint as resnetVariant_best.pth.

---

## Run Inference
To test the trained model on new images:
```bash
python src/infer.py --model resnetVariant_best.pth
```

---

## Model Overview
- The architecture is a modified ResNet with 3 residual layers instead of deeper ResNet variants.
- Uses SGD with momentum, weight decay, and cosine annealing learning rate scheduler.
- The best model achieves 94.06% accuracy on CIFAR-10 with only 2.7 million parameters.

---

## Training Logs & Results
- Training logs are stored in `log.txt`.
- Training loss, lerning rate and validation accuracy curves are available in the main folder.

---


## References
- ResNet Paper: _Kaiming He, et al., "Deep Residual Learning for Image Recognition"_, [arXiv:1512.03385](https://arxiv.org/abs/1512.03385)
- PyTorch CIFAR-10 Repository: [https://github.com/kuangliu/pytorch-cifar](https://github.com/kuangliu/pytorch-cifar)

---

## License
This project is for educational purposes and follows an open-source MIT License.

---

## Authors
- Tianyu Liu
- Chenke Wang
- Zhaochen Yang

