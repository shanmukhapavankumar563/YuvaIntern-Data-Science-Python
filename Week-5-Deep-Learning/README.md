# Week 5: Deep Learning Application in Data Science

## Objective and problem

Train a convolutional neural network to classify 28×28 grayscale clothing images into one of ten Fashion-MNIST categories. Input is one grayscale image; the model returns ten class logits and a predicted category. This is an educational computer-vision exercise, not a guarantee of performance on retail photographs.

## Dataset source and license

Fashion-MNIST was created by Zalando Research and is available from the [official repository](https://github.com/zalandoresearch/fashion-mnist). It contains 60,000 training images and 10,000 test images across ten classes, each 28×28 grayscale. The official repository publishes the dataset under the [MIT License](https://github.com/zalandoresearch/fashion-mnist/blob/master/LICENSE). The torchvision loader downloads and caches the dataset under `data/` on its first run.

## Environment and model choice

The inspected environment has PyTorch 2.8.0 CPU, torchvision 0.23.0 CPU, 12 logical CPU cores, and no CUDA GPU. TensorFlow is not installed. The project therefore uses PyTorch and a compact CNN with two convolution/pooling blocks rather than a large network. Convolutions learn local image patterns; a small fully connected classifier maps learned features to ten classes. CPU threads are capped to limit resource use.

The network is: input 1×28×28 → Conv(1,32,3×3)+ReLU+MaxPool → Conv(32,64,3×3)+ReLU+MaxPool → Flatten(3136) → Dense(128)+ReLU+Dropout(0.30) → Dense(10 logits). CrossEntropyLoss is appropriate for integer multiclass labels. Adam uses learning rate 0.001 and weight decay 0.0001. Batch size is 128, with a maximum of five epochs and validation-loss early stopping (patience two). ToTensor converts pixel integers from [0,255] to float values in [0,1]. Dropout and weight decay regularize the small model; validation checkpointing avoids selecting on the test set.

## Setup and execution

Python 3.10 or later is recommended. From this folder:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
python deep_learning_model.py
```

On macOS/Linux, activate using `source .venv/bin/activate`. The first run downloads Fashion-MNIST; internet access is needed unless the torchvision cache is already populated. The script uses deterministic seeds for Python, NumPy, and PyTorch. CPU/GPU kernels may still vary slightly across hardware and library versions.

## Outputs

Generated artifacts are saved under `outputs/`:

- `report.docx` — objective, source/license, architecture diagram, training configuration, actual curves and metrics, test confusion matrix, limitations, and conclusions.
- `training_history.png`, `training_history.csv` — train/validation loss and accuracy by epoch.
- `architecture.png` — annotated network diagram and tensor dimensions.
- `confusion_matrix.png`, `confusion_matrix.csv`, `classification_report.txt` — held-out test evaluation.
- `test_metrics.json`, `test_predictions.csv`, `test_prediction_examples.png` — measured test scores, predicted labels/probabilities, and visual examples.
- `best_model.pt`, `fashion_mnist_cnn.pt` — best validation checkpoint and model checkpoint with final test metadata.
- `dataset_inspection.json` — dimensions, class counts, split sizes, versions, device, and configuration.

## Internship instruction checklist

| Screenshot requirement | Project coverage |
|---|---|
| Apply deep learning with Python framework | PyTorch CNN implemented in `deep_learning_model.py` |
| Public dataset suitable for deep learning | Fashion-MNIST source and MIT licensing documented above and in report |
| Define the problem and output | Ten-class clothing classification from one grayscale image |
| Design a justified neural network | Two convolution blocks plus compact dense head; architecture diagram in report and outputs |
| Train and evaluate with appropriate metrics | Epoch loss/accuracy, test accuracy, confusion matrix, class-level precision/recall/F1 |
| Discuss overfitting and resource challenges | Validation curves/early stopping and CPU-only constraints are documented |
| Detailed DOCX with code, diagrams, decisions, and analysis | `outputs/report.docx`, generated from the actual run |
| Inspect and preprocess the data | Shape, classes, split counts, ToTensor normalization, stratified validation split |
| Appropriate train/validation/test partitions | Official test set untouched; validation carved from official training data |
| Training configuration documented | Optimizer, learning rate, loss, batch, epochs, patience, dropout, and weight decay recorded |
| Save model and results | Checkpoints, predictions, metrics, plots, and report saved in outputs |
| Reproducible beginner-friendly code | Seeded, commented single-file workflow and install/run instructions |
| Evaluation rigor | Checkpoint and early-stopping decisions use validation only; test used after training |
| Approximately 30–35 focused hours | Deliverables are provided; time spent is not inferred or claimed |

## Limitations and possible improvements

Fashion-MNIST images are small, grayscale, centered crops, unlike varied product photos. A compact CPU model and five-epoch cap may underfit; inspect the saved curves. Future work could add modest training-only image augmentation, a learning-rate schedule, repeated seeds, or a larger GPU model, then validate on a more representative dataset. Test results are for this dataset and run only; they do not establish retail deployment performance.
