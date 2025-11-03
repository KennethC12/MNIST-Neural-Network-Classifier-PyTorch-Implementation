# Neural Network Classifier (NumPy → Softmax → PyTorch MNIST)

This project is a single Jupyter notebook (`NerualNetworkClassifier.ipynb`) that walks through building classic ML and neural network models from scratch and then re-implementing the same ideas in PyTorch.

It starts simple (feature engineering + linear models) and then levels up to:
- binary logistic regression on synthetic data,
- multiclass softmax on 2D Gaussian blobs,
- and finally an MLP trained on MNIST with PyTorch (both a "manual params" version and the clean `nn.Module` version).

**There is no dataset to download manually** — every section either:
- generates synthetic data on the fly, or
- uses `torchvision.datasets.MNIST`, which downloads MNIST automatically the first time you run it.

So you can **clone → install → run** the notebook.

---

## 📁 Project Contents

**`NerualNetworkClassifier.ipynb`**

- **Question 1** – feature mappings + multioutput regression
  - `poly_features`, `rbf_features`, `sigmoid_features`
  - `MultioutputRegression` class with ridge (L2) support

- **Question 2** – binary logistic regression from scratch on 2D data
  - manual loss + gradient
  - accuracy / precision / recall / F1 / ROC-AUC

- **Question 3** – multiclass softmax classifier from scratch
  - synthetic 3-class Gaussians
  - stable softmax, cross-entropy, confusion matrix

- **Question 4** – PyTorch MLP on MNIST
  - dataset + dataloaders
  - manual training loop (init params, forward, backward, update)
  - clean `nn.Module` version (MLP) + `train_module(...)`
  - compare activations (relu, tanh, sigmoid)

- **Question 5 / Discussion** – gradient descent & optimization discussion

---

## 🎯 What This Shows

### How to build features for non-linear problems
- Polynomial basis
- Gaussian RBF basis
- Random sigmoid features
- How to wrap it in a reusable model class

**`MultioutputRegression`** does:
- `_make_design_matrix(...)`
- closed-form or ridge solution
- `.fit(X, Y)` / `.predict(X)`

### How logistic regression actually trains
- **forward**: ŷ = σ(wᵀx + b)
- **loss**: binary cross-entropy
- **backward**: gradients for w and b
- **loop**: update → evaluate → plot

### How softmax generalizes logistic regression
- scores → softmax → cross-entropy
- one-hot labels
- classification report

### How the exact same ideas look in PyTorch
- define model
- choose activation dynamically (`get_activation(...)`)
- loop over `DataLoader`
- compute loss
- backprop
- track train/test accuracy

---

## 🛠️ Installation
```bash
git clone <your-repo-url>
cd <your-repo>
python -m venv venv
source venv/bin/activate   # on Windows: venv\Scripts\activate
pip install -r requirements.txt
```

If you don't have a `requirements.txt` yet, this is enough for this notebook:
```
numpy
matplotlib
scikit-learn
torch
torchvision
torchaudio
jupyter
```

Then open the notebook:
```bash
jupyter notebook NerualNetworkClassifier.ipynb
```

---

## 🚀 How to Run

1. Open the notebook.
2. Run cells top to bottom.
3. Each section is self-contained:
   - **Q1** runs on generated x
   - **Q2/Q3** generate 2D points (no file I/O)
   - **Q4** downloads MNIST the first time
4. Plots will pop up inline.

---

## 🧩 Implementation Details (by section)

### 1. Feature Engineering & Regression (Question 1)

**Goal**: show that linear models can solve non-linear problems if you map inputs to richer features.

**Functions implemented**:
- `poly_features(x, degree)` → [1, x, x², …, x^degree]
- `rbf_features(x, mus, s)` → Gaussian bumps around centers
- `sigmoid_features(x, Wrand, brand)` → random projection + nonlinearity

**Model**: `MultioutputRegression`
- stores hyperparams (basis, degree, num_bases, ridge)
- builds design matrix depending on basis
- solves for weights
- predicts on new data

**Why it's cool to mention on a resume**: shows you understand basis functions, overfitting, and regularization (ridge) without relying on scikit-learn's black boxes.

### 2. Binary Logistic Regression from Scratch (Question 2)

**Data**: 2 Gaussian blobs in 2D → linearly separable-ish.

**Forward pass**:
```python
a = X @ w + b
yhat = sigmoid(a)
```

**Loss**: binary cross-entropy with small epsilon.

**Backward**:
```python
diff = yhat - t
grad_w = X.T @ diff / N
grad_b = diff.mean()
```

**Metrics used**: `accuracy_score`, `log_loss`, `precision_score`, `recall_score`, `f1_score`, ROC curve + AUC.

**Why**: to show you can write your own training loop.

### 3. Multiclass Softmax Classifier (Question 3)

**Data**: 3 Gaussian clusters in 2D.

**Softmax (stable)**:
```python
A = A - np.max(A, axis=1, keepdims=True)
expA = np.exp(A)
P = expA / np.sum(expA, axis=1, keepdims=True)
```

**Loss**: multiclass cross-entropy.

**Reporting**: confusion matrix + precision/recall/F1 per class.

**Why**: shows you understand the jump from binary → multiclass.

### 4. PyTorch MLP on MNIST (Question 4)

**Data loading**:
```python
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))
])
train_data = datasets.MNIST('./data', train=True, download=True, transform=transform)
test_data = datasets.MNIST('./data', test=False, download=True, transform=transform)
```

**Model idea**: flatten 28×28 → FC(784 → 128) → activation → FC(128 → 10)

**Two versions in the notebook**:
1. **Manual-param version** – initialize params, run forward, compute loss, call `.backward()`, update params yourself.
2. **`nn.Module` version (MLP)** – cleaner, standard PyTorch training loop.

**Extras included**:
- `get_activation(name)` so you can switch between relu, tanh, sigmoid
- accuracy evaluation on test set
- confusion matrix via sklearn

**Why this matters**: it shows you can go from scratch → framework and you know what PyTorch is actually doing under the hood.

### 5. Gradient Descent / Optimization (Question 5)

Final cells talk about gradient-based optimization:
- define a 2D energy / loss surface
- compute gradient
- step in direction of -grad
- visualize the path

**Good to mention**: you explored learning rate effects and how different update rules change convergence.

---

## 🧪 Results / What to Expect

- **Synthetic 2D classification**: ~perfect or high accuracy because data is clean.
- **Softmax on 3 Gaussians**: clear decision boundaries when plotted.
- **MNIST MLP**: exact number depends on epochs/batch size, but a small 1-hidden-layer MLP should reach **~95%+** with ReLU and a few epochs on CPU.
  - (If you run fewer epochs, accuracy will be lower — that's normal.)

---

## ✅ Skills Demonstrated

- Python + NumPy + Matplotlib
- Feature engineering (poly / RBF / random features)
- Logistic regression (binary) from scratch
- Softmax / multiclass classification from scratch
- Evaluation metrics (accuracy, precision, recall, F1, AUC, confusion matrix)
- PyTorch data pipeline (datasets, DataLoader)
- PyTorch MLP (`nn.Module`, manual vs optimizer training)
- Working with MNIST without shipping a dataset in the repo

---

## 📦 `requirements.txt`
```
numpy
matplotlib
scikit-learn
torch
torchvision
torchaudio
jupyter
```

---

## 🗺️ Future Improvements

- [ ] Add CNN version for MNIST
- [ ] Add early stopping / LR scheduler
- [ ] Save best model checkpoints
- [ ] Export notebook to `.py` script
- [ ] Turn each "Question" into its own module for cleaner GitHub structure

---

## 📄 License

[Add your license here, e.g., MIT]

## 🤝 Contributing

Feel free to open issues or submit pull requests!
