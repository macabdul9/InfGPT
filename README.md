# Speech and Multimodal Evaluation Framework

This project provides a comprehensive framework for evaluating speech and multimodal tasks using Large Language Models (LLMs). It includes tools for preprocessing, tokenization, and evaluation across different datasets and tasks.

---

## Directory Structure

### Main Folders
- **`external/`**  
  Directory for external dependencies or third-party scripts/tools.

- **`logs/`**  
  Contains logs generated during evaluations.

- **`MMMUResults/`**  
  Stores evaluation results for `MMMU` tasks.

- **`MMMUTokenized/`**  
  Contains pre-tokenized data for `MMMU` tasks.

- **`SpeechResults/`**  
  Stores evaluation results for speech tasks.

- **`SpeechTokenized/`**  
  Contains pre-tokenized data for speech tasks.

- **`SpeechTokenizer/`**  
  Repository or module for speech-specific tokenization logic.

- **`SpeechGenResults/`**  
  Stores generation-based evaluation results for speech tasks.

---

## Scripts

### Evaluation Scripts
- **`eval_mmmu.py`**  
  Evaluates `MMMU` tasks in a constrained setting.  
  Supports token-based evaluation of instruction-response tasks.

- **`eval_mmmu_gen.py`**  
  Performs generation-based evaluation for `MMMU` tasks.  
  Focuses on free-form responses.

- **`eval_speech.py`**  
  Evaluates speech tasks with pre-tokenized audio data in a constrained manner.  
  Uses prompts tailored for speech-to-text evaluation.

- **`eval_speech_gen.py`**  
  Performs free-form generation-based evaluation for speech tasks.  
  Handles tasks dynamically with multiple datasets.

### Tokenization Scripts
- **`speech_tokenization.py`**  
  Tokenizes audio files for speech tasks.  
  Outputs tokenized representations for use in evaluations.

- **`image_tokenization.py`**  
  Tokenizes image data for image-based tasks.  
  Supports multimodal evaluations.

### Shell Scripts
- **`eval_mmmu.sh` / `eval_mmmu_gen.sh`**  
  Shell scripts to run `MMMU` evaluations.

- **`eval_speech.sh` / `eval_speech_gen.sh`**  
  Shell scripts to run speech evaluations.

- **`tokenize_image_audio.sh`**  
  Script for tokenizing both image and audio data.

### Other Scripts
- **`inference.py`**  
  General inference script for running models on various tasks.

- **`anygpt_install.sh`**  
  Script to install dependencies and set up the environment.

---

## Key Files

- **`speech_tasks.json`**  
  JSON file containing configurations for speech datasets.

- **`README.md`**  
  This file, providing an overview of the project.

---

## Setup

### Prerequisites

- Python 3.8 or higher
- Required Python packages (install using the provided installation script):
  ```bash
  bash anygpt_install.sh

