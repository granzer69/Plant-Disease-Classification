# Plant Disease Classification

CNN-based **leaf disease classifier** (38 classes) with a Streamlit inference UI and recorded training history.

Companion model weights also live in [plant-disease-detector](https://github.com/granzer69/plant-disease-detector) (`trained_plant_disease_model.keras`).

## Overview

Upload a plant leaf image and get a disease class prediction plus a short care recommendation. Training was done in `Train_Plant.ipynb`; inference UI is `main2.py` (Streamlit + TensorFlow/Keras).

## Tech stack

- TensorFlow / Keras
- NumPy
- Streamlit
- Training notebook + `training_hist.json`

## Features

- Multi-crop disease recognition (Apple, Corn, Grape, Potato, Tomato, … — 38 labels)
- Confidence from model softmax
- Rule-based treatment tips per class
- Team project (Ribhu S, Saketh, Varun) per UI about section

## Results (from `training_hist.json`)

Three logged epochs:

| Epoch | Train acc | Val acc |
|------|-----------|---------|
| 1 | ~57.8% | ~84.1% |
| 2 | ~85.9% | ~91.4% |
| 3 | ~91.5% | ~91.0% |

Final logged **validation accuracy ≈ 91.0%**. No additional held-out test metrics are recorded in-repo beyond this history file.

## Getting started

1. Obtain `trained_plant_disease_model.keras` (see `plant-disease-detector` or your training output).
2. `pip install tensorflow streamlit numpy`
3. `streamlit run main2.py`

## Project structure

```text
Train_Plant.ipynb          training
main2.py                   Streamlit app
training_hist.json         logged accuracy/loss
*.JPG / pptx               samples / slides
```

## Future improvements

- Publish eval on a fixed test split with confusion matrix
- Slimmer dependency story and model card
- Merge classifier + detector repos for a single recruiter entry point
