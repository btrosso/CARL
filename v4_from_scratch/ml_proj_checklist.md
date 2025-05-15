# PyTorch Multi-Class Classification: Core Components & Success Criteria

## 1. Dataset and Dataloader
- [ ] Implement a custom `Dataset` class (`torch.utils.data.Dataset`)
- [ ] Implement `__len__()` and `__getitem__()` methods
- [ ] Wrap the dataset using `torch.utils.data.DataLoader` with `batch_size` and `shuffle`

**✅ Success When:**
- You can call `next(iter(dataloader))` and get a `(features, labels)` batch
- `features` is a `torch.Tensor` of shape `[batch_size, channels, height, width]`
- `labels` is a `torch.Tensor` of shape `[batch_size]` with integer class indices

---

## 2. Transforms (Preprocessing & Augmentation)
- [ ] Use `torchvision.transforms.Compose()` to create a transform pipeline
- [ ] Include basic preprocessing: `Resize`, `ToTensor`, `Normalize`
- [ ] Optionally add augmentations like `RandomHorizontalFlip`, `ColorJitter`, etc.

**✅ Success When:**
- Images returned by your dataset are correctly shaped and normalized
- You can visualize one image from the dataset and it looks correctly preprocessed

---

## 3. Model
- [ ] Define a class that inherits from `nn.Module`
- [ ] Implement `__init__()` and `forward()` methods
- [ ] Include appropriate layers (Conv2D, ReLU, MaxPool, Linear, etc.)

**✅ Success When:**
- Instantiating your model and running `model(input_tensor)` returns logits of shape `[batch_size, num_classes]`

---

## 4. Loss Function
- [ ] Use `torch.nn.CrossEntropyLoss()` (no softmax needed in model)
- [ ] Pass model outputs and true labels to compute the loss

**✅ Success When:**
- Loss returns a non-zero float value when passed logits and ground truth labels
- No errors for shape mismatches (common pitfall: softmax already applied or wrong label shape)

---

## 5. Optimizer
- [ ] Use `torch.optim.Adam(model.parameters(), lr=...)` or similar
- [ ] Tie it to your model parameters

**✅ Success When:**
- No errors on `optimizer.step()` during training
- Loss begins to decrease over epochs

---

## 6. Training Loop
- [ ] Write loop over epochs and batches
- [ ] Apply `model.train()`, forward pass, `loss.backward()`, `optimizer.step()`, `optimizer.zero_grad()`

**✅ Success When:**
- Loss prints after each batch/epoch and decreases over time
- You get no gradient-related or device mismatch errors

---

## 7. Validation Loop
- [ ] Switch to `model.eval()` and wrap with `torch.no_grad()`
- [ ] Compute and store metrics like accuracy

**✅ Success When:**
- Validation accuracy prints per epoch and shows generalization
- Loop runs without affecting training gradients

---

## 8. Metric Tracking & Logging
- [ ] Track `train_loss`, `val_loss`, `accuracy`, etc.
- [ ] Use `matplotlib`, CSV logs, or logging libraries (optional: TensorBoard, WandB)

**✅ Success When:**
- You can visualize or print clear learning curves over epochs

---

## 9. Device Management
- [ ] Check for CUDA with `torch.cuda.is_available()`
- [ ] Move model and tensors to `device` with `.to(device)`

**✅ Success When:**
- Code runs on GPU (if available) and prints `cuda` as the device
- No "expected device X but got Y" runtime errors

---

## 10. Saving & Loading Checkpoints
- [ ] Use `torch.save(model.state_dict(), 'model.pth')`
- [ ] Use `model.load_state_dict(torch.load(...))` for restoring weights

**✅ Success When:**
- Model state reloads successfully and inference outputs are consistent before/after save

---

## 11. Inference Script
- [ ] Switch model to `eval()` mode
- [ ] Pass new data through model
- [ ] Use `torch.argmax(logits, dim=1)` to get predicted class

**✅ Success When:**
- Inference returns valid class indices
- Matches expected classes on known samples

---

## 12. Training Configuration (Optional)
- [ ] Use `argparse` or config files (`.json`, `.yaml`) to control hyperparameters

**✅ Success When:**
- You can re-run training with new parameters without changing core code

---


## Proposed Folder Structure

    project_root/
    │
    ├── data/                # Raw data or download scripts
    ├── datasets/            # Dataset definitions
    │   └── dataset.py
    ├── models/              # Model definitions
    │   └── model.py
    ├── training/            # Training logic
    │   ├── train.py
    │   ├── validate.py
    │   └── utils.py
    ├── configs/             # JSON/YAML or Python config files
    │   └── config.yaml
    ├── inference/           # Inference and post-processing
    │   └── predict.py
    ├── visualize/           # Tools for plots or tensorboard
    │   └── visualize.py
    ├── requirements.txt
    └── README.md
