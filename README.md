## Layer-Wise Analysis of Emotion and Speaker Information in WavLM Speech Representations
Project Description: This project will investigate which layers of a frozen WavLM Base+ speech model contain the most useful information for categorical emotional-expression recognition, and how strongly that information is associated with speaker identity. Using the speech-only portion of RAVDESS, the study will classify happy, sad, angry, and neutral acted expressions. A compact traditional baseline based on MFCC, pitch, and energy features will provide an interpretable comparison. Emotion recognition will use actor-independent evaluation, while a separate speaker-identification probe will measure how accessible actor identity is in each representation. WavLM will be used as a frozen feature extractor; it will not be trained or fine-tuned.
## Installation
# a). Checking the runtime environment
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
# b). Run MATLAB first
1. Open MATLAB.
2. Set the ‘Current Folder’ to the project root directory, i.e. the directory containing the two .m files.
3. Open run_ravdess_week8.m in the editor.
4. Click the green ‘Run’ button, or press F5.
5. Wait for the programme to finish running. By default, the images will be saved and the figure window closed; the images are located in the `results/figures/` directory within the current directory.
   Upon successful completion, the following message will be displayed:
   Saved 672 x 33 acoustic features and descriptive figures.
After clicking ‘Run’, the main programme first checks the `data/raw/` directory and its subdirectories:
  · If complete WAV data is present: it is read and processed directly.
  · If the directory does not exist or contains no WAV files: the data is prepared automatically. If there is no original compressed archive in `data/downloads/`,
    Audio_Speech_Actors_01-24.zip is downloaded from Zenodo; if an archive already exists, it is reused directly.
    Once the checksum has been verified, the archive is extracted to `data/raw/`, after which the generation of two sets of segmentations, pre-processed audio, acoustic features and figures continues.
  · Some WAV files are present but the data is incomplete: an error is reported; the data will not be automatically supplemented.
Therefore, the main MATLAB programme can be run first; it will create the `data` directory and retrieve the data as required,
without requiring the recipient to manually download the original recordings in advance. Access to Zenodo is required for automatic downloads; an error will be reported if this fails.
There is no need to run `clean_f0_track.m` or any functions from the toolbox separately.
# c). Run Python again
Once MATLAB has finished successfully, open PowerShell in the same project root directory and run:
```bash
  $env:HF_HOME = Join-Path (Get-Location) “.cache\huggingface”
  .\.venv\Scripts\python.exe -u python\extract_wavlm.py --revision 4c66d4806a428f2e922ccfa1a962776e232d487b
```
Python will read the `results/manifest.csv` file and the `results/prepared_audio/` directory,
then extract all target layers and save the results to `results/wavlm/`.
Upon successful completion, the following message will be displayed:
  COMPLETE: actual frozen WavLM cache shape (672, 4, 768).
If the model is not cached, an internet connection is required for the first run to download it.
