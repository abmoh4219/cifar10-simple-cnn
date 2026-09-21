# Prompts used in this project

This project was built in phases. For each phase, I planned the work with
Claude in a chat, which drafted the implementation prompt below. I reviewed
the prompt, ran it in Claude Code, checked the output myself, and committed
before moving to the next phase.

Every prompt follows the same pattern:

- say exactly what the phase must do
- say what it must NOT touch
- name the known traps before the code is written
- ask for docstrings that explain WHY, because the code is presented

---

## Phase 0 — Scaffold and environment

```
This repo is a take-home assessment deliverable for an ML engineering role.
I will record a 15-20 minute screen walkthrough of it, so structure and
clarity matter as much as correctness.

PHASE 0 — SCAFFOLD ONLY. Do NOT implement any model, training, or data
logic in this phase. Create the project skeleton and verify the
environment works. I will implement the logic in later phases.

Create exactly this structure:

cifar10-simple-cnn/
├── CLAUDE.md
├── README.md
├── requirements.txt
├── .gitignore
├── verify_env.py
├── src/
│   ├── __init__.py
│   ├── config.py
│   ├── data.py
│   ├── model.py
│   ├── train.py
│   ├── evaluate.py
│   └── predict.py
├── artifacts/.gitkeep
└── samples/.gitkeep

File requirements:

1. .gitignore — Python standard, plus: .venv/, data/, __pycache__/,
   *.pyc, .DS_Store, .ipynb_checkpoints/. Do NOT ignore artifacts/ —
   trained model and plots are committed deliverables.

2. requirements.txt — torch, torchvision, numpy, matplotlib, pandas,
   tqdm. Use >= minimum version pins, one per line, with a brief
   comment header.

3. src/config.py — a single frozen dataclass named Config holding all
   hyperparameters and paths, with an instance named CFG. Fields:
   seed=42, batch_size=128, epochs=20, lr=1e-3, weight_decay=1e-4,
   val_split=5000, num_workers=2, data_dir="data",
   artifacts_dir="artifacts", samples_dir="samples",
   checkpoint_name="best_model.pt". Plus CIFAR-10 channel mean and std
   as tuples, and the 10 class names as a tuple in label-index order.
   This file IS implemented in this phase — it is configuration, not
   logic.

4. src/data.py, model.py, train.py, evaluate.py, predict.py — stubs
   only. Each gets a module-level docstring explaining its single
   responsibility, and the function signatures I will fill in later
   with `raise NotImplementedError`. No bodies.

5. verify_env.py — a real, working script that prints Python version,
   torch version, torchvision version, and the selected device
   (cuda if available, else mps if available, else cpu). Then runs a
   single forward pass of a throwaway 3x3 conv over a random
   (1, 3, 32, 32) tensor on that device and prints the output shape.
   Exit code 0 on success, non-zero with a clear message on failure.

6. README.md — title, one-line description, a Setup section with the
   venv and pip install commands, and a Project Structure section with
   the tree above annotated. Add a "Results" heading with the text
   "(filled in after training)". Keep it short; it gets expanded later.

7. CLAUDE.md — project context for future sessions. State: this is a
   take-home assessment; the spec requires a SIMPLE CNN with 1-2
   convolutional layers and that constraint must not be exceeded
   regardless of accuracy benefit; framework is PyTorch; every random
   source must be seeded from CFG.seed; the test set is touched exactly
   once, in evaluate.py, and model selection uses the validation split
   only; predict.py must cold-load the checkpoint from disk in a
   separate process; all hyperparameters live in config.py and are
   never hardcoded elsewhere.

Do not run pip install. Do not create any data. Show me the files when
done.
```

---

## Phase 1 — Data pipeline

```
PHASE 1 — DATA PIPELINE ONLY. Implement src/data.py. Do not touch
model.py, train.py, evaluate.py, or predict.py.

Read src/config.py first and use CFG for every constant. Do not
hardcode any value that exists in CFG.

Implement in src/data.py:

1. set_seed(seed: int) -> None
   Seed random, numpy, and torch (CPU and CUDA). Set
   torch.backends.cudnn.deterministic = True and benchmark = False.

2. get_transforms() -> tuple[Compose, Compose]
   Returns (train_tf, eval_tf).
   train_tf: RandomCrop(32, padding=4), RandomHorizontalFlip(),
             ToTensor(), Normalize(CFG.mean, CFG.std)
   eval_tf:  ToTensor(), Normalize(CFG.mean, CFG.std)
   Augmentation must appear ONLY in train_tf.

3. get_datasets() -> tuple[Dataset, Dataset, Dataset]
   Returns (train_ds, val_ds, test_ds).

   CRITICAL REQUIREMENT — read carefully:
   The validation split must NOT receive training augmentation.
   Do NOT call random_split on a single dataset object and return
   both halves, because the two subsets would share one transform
   and the validation set would be randomly cropped and flipped.

   Instead: instantiate the CIFAR-10 training data TWICE from the
   same root — once with train_tf, once with eval_tf. Generate ONE
   permutation of indices using a torch.Generator seeded with
   CFG.seed. Use Subset with the first (50000 - CFG.val_split)
   indices on the train_tf copy, and the remaining CFG.val_split
   indices on the eval_tf copy. This guarantees the split is
   identical across runs AND that the two subsets are disjoint AND
   that validation sees no augmentation.

   test_ds uses the CIFAR-10 test split with eval_tf.
   All three download to CFG.data_dir.

4. get_dataloaders(batch_size=None) -> tuple[DataLoader, DataLoader, DataLoader]
   Defaults to CFG.batch_size. shuffle=True for train only.
   num_workers=CFG.num_workers, pin_memory only if cuda is available.
   Use a seeded generator for the train loader's shuffling.

5. export_sample_images(n: int = 8) -> list[Path]
   Takes the first n images of the TEST set, saves them as PNG files
   into CFG.samples_dir named like "sample_0_<classname>.png" using
   the true label. Save the raw un-normalized image, not the tensor.
   Returns the list of written paths. These are for a later inference
   demo, so they must be readable as ordinary image files.

6. A `if __name__ == "__main__":` smoke test that prints:
   - dataset sizes for train / val / test
   - one batch's tensor shape and dtype
   - the batch's per-channel mean and std (should be near 0 and 1
     if normalization is correct)
   - the class distribution of the validation split as counts
   - confirmation that train and val index sets do not intersect
   - wall-clock time to iterate 20 training batches
   Then call export_sample_images() and print where they went.

Type-hint everything. Docstrings on each function explaining WHY,
not just what — I present this code on video.
```

---

## Phase 2 — Model and utils

```
PHASE 2 — MODEL + UTILS. Implement src/model.py and create src/utils.py.
Also make two small amendments to src/data.py. Do NOT touch train.py,
evaluate.py, or predict.py.

Read src/config.py and src/data.py first.

TASK A — create src/utils.py
  Move set_seed() out of data.py into utils.py unchanged. Add
  get_device() -> torch.device returning cuda if available, else mps
  if available, else cpu. Update data.py to import set_seed from
  utils. Do not leave a duplicate definition behind.

TASK B — amend src/data.py export_sample_images()
  Currently it takes the first n test images, which repeats classes
  (3 frogs, 2 ships). Change it to select ONE image per class, in
  label order, taking the first occurrence of each class in the test
  set. Signature becomes export_sample_images() -> list[Path] with no
  n argument; it writes exactly len(CFG.classes) images named
  "sample_<label>_<classname>.png". Delete the old sample files.

TASK C — implement src/model.py

  class SimpleCNN(nn.Module) with EXACTLY TWO convolutional layers.
  This is a hard spec constraint from the assessment brief. Do not
  add a third conv block even though it would improve accuracy. Do
  not suggest one.

  Architecture:
    Block 1: Conv2d(3, 32, kernel_size=3, padding=1) -> BatchNorm2d(32)
             -> ReLU -> MaxPool2d(2)          # 32x32 -> 16x16
    Block 2: Conv2d(32, 64, kernel_size=3, padding=1) -> BatchNorm2d(64)
             -> ReLU -> MaxPool2d(2)          # 16x16 -> 8x8
    Head:    Flatten -> Dropout(0.25) -> Linear(64*8*8, len(CFG.classes))

  Put the two conv blocks in an nn.Sequential named `features` and the
  head in an nn.Sequential named `classifier`, so the two halves are
  separable for inspection.

  BatchNorm goes BEFORE ReLU, not after. Add a brief comment saying why.

  Add:
    - count_parameters(model) -> tuple[int, int] returning
      (total, trainable)
    - a forward() that asserts the input is (N, 3, 32, 32) and raises
      a clear error otherwise

TASK D — smoke test in model.py under if __name__ == "__main__":
  1. Instantiate on get_device(), print the module repr.
  2. Print total and trainable parameter counts with thousands
     separators.
  3. Forward a random (4, 3, 32, 32) tensor; assert output shape is
     (4, 10); print it.
  4. Print the output shape of `features` alone on that tensor, to
     show the 64x8x8 feature map that determines the Linear input size.
  5. Confirm the model rejects a wrong-shaped input with a clear error.
  6. SINGLE-BATCH OVERFIT TEST — the important one:
     Pull ONE batch from the train loader. Train the model on that
     single batch alone for 100 steps with Adam(lr=CFG.lr) and
     CrossEntropyLoss. Print loss and batch accuracy every 20 steps.
     A correctly wired model must reach near-zero loss and 100%
     accuracy on one batch. Print a clear PASS/FAIL verdict based on
     final accuracy == 1.0. Call set_seed(CFG.seed) first so this is
     reproducible.
  7. Time a single forward+backward pass on a full CFG.batch_size
     batch and print ms/step, so we can estimate epoch duration.

Type-hint everything. Docstrings explain WHY. I present this on video.
```

---

## Phase 3 — Training loop

```
PHASE 3 — TRAINING LOOP. Implement src/train.py. Do NOT touch
evaluate.py or predict.py. data.py, model.py, utils.py, config.py are
done — read them first.

Delete the dead set_seed stub at train.py:15. Import set_seed and
get_device from src.utils.

Implement in src/train.py:

1. train_one_epoch(model, loader, criterion, optimizer, device) -> tuple[float, float]
   Returns (mean_loss, accuracy). model.train(). Use tqdm with a
   description showing running loss. Accumulate loss weighted by
   batch size, not a simple mean over batches, since the last batch
   is smaller.

2. validate(model, loader, criterion, device) -> tuple[float, float]
   Returns (mean_loss, accuracy).
   CRITICAL: must call model.eval() AND wrap the loop in
   torch.no_grad(). Without eval() the dropout stays active and
   BatchNorm updates its running statistics from validation data,
   which corrupts both the metric and the model. Add a comment
   saying exactly this.

3. save_checkpoint(model, optimizer, epoch, val_acc, path) -> None
   Save a dict containing: model_state_dict, optimizer_state_dict,
   epoch, val_acc, and the full CFG as a plain dict (use
   dataclasses.asdict). Saving the config alongside the weights means
   predict.py can verify it is loading a checkpoint that matches the
   current preprocessing.

4. plot_curves(history_csv_path, out_path) -> None
   Reads the history CSV with pandas and writes a single PNG with two
   side-by-side subplots: left = train vs val loss, right = train vs
   val accuracy, both against epoch. Label axes, add a legend, add a
   title. Mark the best-validation-accuracy epoch with a vertical
   dashed line and annotate it. dpi=150. This figure goes in the
   README and on video, so it must be readable.

5. main() -> None
   - set_seed(CFG.seed), get_device(), print device
   - build model, print parameter count
   - CrossEntropyLoss, Adam(lr=CFG.lr, weight_decay=CFG.weight_decay)
   - CosineAnnealingLR over CFG.epochs, stepped once per epoch
   - loop CFG.epochs times; after each epoch append a row to
     artifacts/history.csv with: epoch, train_loss, train_acc,
     val_loss, val_acc, lr, epoch_seconds
   - print a clean one-line summary per epoch
   - track best val_acc; save checkpoint to
     artifacts/<CFG.checkpoint_name> ONLY when val_acc improves, and
     print that it was saved
   - the TEST loader must never be used in this file. Do not import
     it, do not evaluate on it.
   - at the end: print total wall-clock time, best val acc and which
     epoch it came from, then call plot_curves to write
     artifacts/training_curves.png

   Write history.csv incrementally (flush each epoch), so a crash or
   interrupt still leaves usable data.

6. Guard with if __name__ == "__main__": main()

Type-hint everything. Docstrings explain WHY. I present this on video.
```

Follow-up, after the clean training run:

```
Copy the full stdout of the clean 20-epoch run from the task output
file into artifacts/train_log.txt, verbatim, with no tqdm progress
bar lines. Do not re-run training.
```

---

## Phase 4 — Evaluation

```
PHASE 4 — EVALUATION. Implement src/evaluate.py. Do NOT touch
predict.py or retrain anything. Read config.py, data.py, model.py,
utils.py, train.py first.

This is the ONLY module in the project that uses the test set, and it
uses it exactly once. Add a module docstring saying so.

Implement in src/evaluate.py:

1. load_checkpoint(path, device) -> tuple[nn.Module, dict]
   Load artifacts/<CFG.checkpoint_name> with map_location=device.
   Build the model, load state_dict, move to device, call model.eval().
   Return (model, checkpoint_metadata) where metadata excludes the
   state dicts. Print the epoch and val_acc the checkpoint came from.
   If the checkpoint's stored config disagrees with the current CFG on
   seed, batch_size, mean, or std, print a clear WARNING naming each
   mismatched field — a checkpoint trained under different
   preprocessing must not be silently evaluated.

2. evaluate(model, loader, device) -> dict
   Run the test set under model.eval() and torch.no_grad(). Collect
   all predictions, true labels, and softmax confidences. Return a
   dict with: loss, accuracy, y_true, y_pred, y_conf (numpy arrays).

3. per_class_metrics(y_true, y_pred) -> pd.DataFrame
   One row per class: support, correct, accuracy, precision, recall,
   f1. Compute these by hand with numpy — do not add scikit-learn as
   a dependency for six formulas. Sort by accuracy ascending so the
   weakest classes appear first. Append a macro-average row.

4. confusion_matrix(y_true, y_pred, num_classes) -> np.ndarray
   Plain numpy, rows = true, cols = predicted.

5. plot_confusion_matrix(cm, out_path) -> None
   Write artifacts/confusion_matrix.png. Row-normalised (each row sums
   to 1) so classes are comparable. Annotate every cell with the
   percentage. Class names on both axes, x labels rotated 45 degrees.
   Colourbar. dpi=150. Readable on a video frame — this is presented.

6. top_confusions(cm, k=5) -> list[tuple[str, str, int]]
   The k largest off-diagonal entries as
   (true_class, predicted_class, count), descending.

7. plot_misclassified(model, loader, device, out_path, n=16) -> None
   A 4x4 grid of misclassified TEST images, un-normalised for display,
   each titled "true -> pred (conf)". Write
   artifacts/misclassified.png, dpi=150.

8. main() -> None
   set_seed, get_device, load checkpoint, evaluate, then print:
   - overall test accuracy and loss
   - the per-class table
   - the top 5 confusions in plain sentences
   - mean confidence on correct vs incorrect predictions
   Write artifacts/test_metrics.json containing overall accuracy,
   loss, per-class metrics, and the confusion matrix as a nested list.
   Write both PNGs. Print where everything went.

9. Guard with if __name__ == "__main__": main()

Type-hint everything. Docstrings explain WHY. I present this on video.
```

---

## Phase 5 — Inference, plus two fixes to evaluation

```
PHASE 5 — INFERENCE + two small fixes. Implement src/predict.py and
amend src/evaluate.py. Do not retrain. Do not touch train.py.

FIX 1 — src/evaluate.py, single-pass test read
  Currently evaluate() iterates the test loader and plot_misclassified()
  iterates it again. Make the test set read exactly once: have
  evaluate() also collect up to N=16 misclassified samples (the
  de-normalised image array, true label, predicted label, confidence)
  during its single pass, returned under a "misclassified" key.
  plot_misclassified() then takes that collected list instead of a
  model and loader. Keep the figure identical.

FIX 2 — src/evaluate.py, per-class table
  Per-class accuracy and recall are the same quantity (TP / support)
  for single-label classification. Drop the redundant "accuracy"
  column; keep support, correct, precision, recall, f1. Sort ascending
  by recall. Say in the docstring why accuracy is not reported
  separately.

  Re-run evaluation after these fixes and confirm the numbers are
  unchanged: test accuracy 0.7587, loss 0.7177, cat weakest at 0.5200.

TASK — implement src/predict.py

  This runs as a SEPARATE PROCESS from training. It must cold-load
  the checkpoint from disk and work with no state carried over. Say
  so in the module docstring.

  Requirements:

  1. Reuse get_transforms() from src.data for the eval transform. Do
     NOT redefine normalization here — a second copy of those
     constants is exactly how preprocessing drifts out of sync with
     the trained model.

  2. load_image(path) -> torch.Tensor
     Open with PIL, convert to RGB (handles PNG alpha and greyscale).
     If the image is not 32x32, resize it to 32x32 and print a notice
     saying it was resized, since a non-CIFAR image being silently
     downscaled would otherwise look like a model failure. Apply the
     eval transform. Return a (1, 3, 32, 32) tensor.

  3. predict(model, tensor, device, topk=3) -> list[tuple[str, float]]
     Under model.eval() and torch.no_grad(). Softmax the logits,
     return the top-k (class_name, probability) pairs descending.

  4. parse_true_label(path) -> str | None
     Our sample files are named "sample_<idx>_<classname>.png". If the
     filename matches that pattern and the class is a known class,
     return it; otherwise return None. Used only for display, never
     to influence the prediction.

  5. main() with argparse:
     --images: one or more file paths OR a directory. Defaults to
       CFG.samples_dir.
     --checkpoint: defaults to artifacts/<CFG.checkpoint_name>.
     --topk: default 3.
     --save-figure: optional output path for a grid PNG.

     Behaviour:
     - print the device and which checkpoint was loaded, including
       the epoch and val_acc stored in it
     - for each image print a compact block: filename, true label if
       known, then the top-k predictions with probabilities as
       percentages
     - mark each as correct/incorrect with a clear symbol when a true
       label is known
     - at the end, if any true labels were known, print an accuracy
       summary over those images
     - fail with a clear message if the checkpoint is missing,
       naming the command to train one

  6. --save-figure writes a grid of the images with predicted label
     and confidence as titles, green title if correct, red if wrong,
     grey if the true label is unknown. dpi=150.

  7. Add a short section to README.md showing the inference command
     and its output.

Type-hint everything. Docstrings explain WHY. I present this on video.
```

---

## Phase 6 — Consolidation and README

```
PHASE 6 — CONSOLIDATION + README. No retraining. Do not change any
metric-producing logic.

TASK A — consolidate checkpoint loading
  load_checkpoint currently exists twice (evaluate.py and predict.py),
  and only evaluate.py's copy has the config-mismatch guard. Move a
  single load_checkpoint(path, device) -> (model, metadata) into
  src/model.py, including the _warn_on_config_mismatch guard and its
  CRITICAL_FIELDS. Both evaluate.py and predict.py import it from
  src.model. Delete both old copies. model.py must not import
  evaluate.py, predict.py, or train.py.
  Verify afterwards:
  - python -m src.evaluate reproduces test acc 0.7587, loss 0.7177
  - python -m src.predict gives 8/10, and now prints the
    "config matches current CFG" line
  - importing src.predict still does not import src.train

TASK B — requirements.txt
  Add a comment noting that CPU-only machines can install a much
  smaller torch build with:
  pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
  Do not change the version pins.

TASK C — rewrite README.md completely: overview, results with
embedded figures and findings, key design decisions, project
structure, how to run, inference, how AI tools were used (with the
concrete moments where AI output was corrected, rejected, or
verified), improving from 90% to 99%, and limitations.
```
