# Layer-Wise Analysis of Emotion and Speaker Information in WavLM Speech Representations
Project Description: This project will investigate which layers of a frozen WavLM Base+ speech model contain the most useful information for categorical emotional-expression recognition, and how strongly that information is associated with speaker identity. Using the speech-only portion of RAVDESS, the study will classify happy, sad, angry, and neutral acted expressions. A compact traditional baseline based on MFCC, pitch, and energy features will provide an interpretable comparison. Emotion recognition will use actor-independent evaluation, while a separate speaker-identification probe will measure how accessible actor identity is in each representation. WavLM will be used as a frozen feature extractor; it will not be trained or fine-tuned.
# Installation
1. Checking the runtime environment
MATLAB and the Signal Processing Toolbox are required, as well as Python 3.12 and the relevant dependencies.
If the environment for the current project is already set up, proceed directly to the following two steps for MATLAB and Python.
Important: When installing Python dependencies on a new Windows machine, first install Python 3.12,
then navigate to the project root directory containing `requirements-lock.txt` and `run_ravdess_week8.m`.
Right-click on an empty area within this root directory, select ‘Open PowerShell’ or ‘Open Command Prompt here’, and press Enter to open PowerShell or the Command Prompt in that location.
Execute the following three commands in sequence:
```bash
  py -3.12 -m venv .venv
  .\.venv\Scripts\python.exe -m pip install torch==2.13.0 --index-url https://download.pytorch.org/whl/cpu
  .\.venv\Scripts\python.exe -m pip install -r requirements-lock.txt
```
First command: Creates the .venv virtual environment for the current project.
Second command: Installs the CPU version of PyTorch from PyTorch’s CPU software repository.
Third command: pip reads the requirements-lock.txt file, downloads and installs the dependencies specified in the manifest into the .venv environment.
The -r option means ‘read the installation manifest from the specified file’, so there is no need to install packages manually one by one.
Installing packages usually requires an internet connection.
If the .venv environment already exists and the dependencies are already installed, there is no need to recreate or reinstall them; simply proceed to the subsequent steps.
If there is no `py` launcher, after ensuring that `python` points to Python 3.12,
the first step can be changed to `python -m venv .venv`.
Translated with DeepL.com (free version)
