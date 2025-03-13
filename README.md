# Modified ResNet for CIFAR-10 Classification
This project implements a modified ResNet architecture for image classification on the CIFAR-10 dataset using Pytorch. The model is optimized to stay within 5 million parameters while achieving accuracy of 94.06%.

---

## Project Structure
This repository contains all necessary files for training, evaluating, and analyzing our modified ResNet model.
📂 data/ - Directory containing the CIFAR-10 dataset.
📂 old_files/ - Archive of previous versions and backup files.
📂 src/ - Source code including model architecture, training, and inference scripts.
📄 .gitignore - Specifies files to be ignored by Git version control.
📄 LICENSE - License information for the project.
📄 README.md - This documentation file.
📄 submission_corrected.csv - CSV file with corrected submission results.
📄 training_logs.pth - Saved PyTorch training logs for model analysis.
📄 training_loss.png - Plot showing the training loss curve over epochs.
📄 validation_accuracy.png - Plot showing validation accuracy over epochs.
📄 log.txt - Training log containing epoch-wise loss and accuracy.
📄 miniproject_spring25.pdf - Project structure and documentation.
📄 resnetVariant_best.pth - Best saved model based on validation accuracy.

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

