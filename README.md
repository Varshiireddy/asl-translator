# ASL Translator 🤟

A web app that reads American Sign Language (ASL) letters from your webcam and spells out words live, with the model running entirely in the browser.

I'm building it in 7 days, starting from a neural network written in plain NumPy and finishing with an ONNX model running in a web page.

## Progress

| Day | What | Status |
|---|---|---|
| 1 | 2-layer neural net in NumPy with hand-written backprop | ✅ |
| 2 | micrograd: my own autograd engine | ⏳ |
| 3 | micrograd MLP on MediaPipe hand landmarks (5 letters) | ⏳ |
| 4 | PyTorch MLP on all letters + live webcam test | ⏳ |
| 5 | Small CNN vs landmark MLP on my own hands | ⏳ |
| 6 | Fine-tuned MobileNetV3 + confusion matrix | ⏳ |
| 7 | Browser app (ONNX Runtime Web) + live demo | ⏳ |

## Day 1: Neural network from scratch

Before using PyTorch, I wrote a 2-layer network in NumPy, with the forward and backward passes done by hand, to understand exactly what `loss.backward()` does.

**Network:** 2 inputs → 16 hidden (ReLU) → 1 output (sigmoid), binary cross-entropy loss, full-batch gradient descent.

**Data:** `make_moons` (1,000 points, noise 0.2). Two curved shapes that a straight line can't separate.

### Results

| Model | Learning rate | Test accuracy |
|---|---|---|
| Logistic regression (baseline) | — | 86% |
| 2-layer NN | 1.0 | 89% |
| 2-layer NN | 10 | 98.5% |
| 2-layer NN | 50 | diverged |

**Backprop verified:** centered finite differences match my analytical gradients, with a max difference of 1.0e-11 across all first-layer weights.

### Decision boundaries

| Logistic regression | 2-layer neural net |
|---|---|
| ![](images/boundary_logreg.png) | ![](images/boundary_nn.png) |

Logistic regression can only draw a straight line. The hidden layer with ReLU lets the network bend its boundary around the moons.

![Training loss](images/loss_curve.png)

### What I learned
- Without a non-linearity like ReLU, stacked linear layers collapse into one linear layer.
- Every layer's weight gradient follows one pattern: `input.T @ error / n`.
- Gradient checking catches silent backprop bugs that never raise an error.
- The learning rate alone moved test accuracy from 89% to 98.5%, and too large a value made training diverge.

## Run it
```bash
pip install -r requirements.txt
jupyter notebook notebooks/day1_numpy_nn.ipynb
```

## Tech stack
NumPy, scikit-learn, Matplotlib. Coming this week: PyTorch, MediaPipe, OpenCV, ONNX Runtime Web.
