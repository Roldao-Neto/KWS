# KWS (Keyword Spotting) in Raspberry Pi Zero 2W

This project implements Keyword Spotting (KWS) optimized to run on a Raspberry Pi Zero 2W. It contains different models trained with the Google Speech Dataset.

This repo is organized into folders, each of which contains: the development Jupyter notebook, the inference script, the trained models, and the requirements.txt file for the step/model described by the folder's name.

## Hardware Requirements (Optional):

* Raspberry Pi Zero 2W
* MicroSD Card
* Microphone for audio capture
* LEDs for visual inference

## Software Requirements

* Python 3.11
* Jupyter Notebook / Jupyter Lab
* Raspberry Pi OS
* Other requirements ought to be downloaded for their respective venvs:

```bash
# For Model Training:
pip install -r requirements-pc.txt

# For inference:
pip install -r requirements-rasp.txt
```

**PS-1:** The following packages should be downloaded to run the inference:

* portaudio19-dev
* python3-dev

**PS-2:** These models were optimized with the `tf-lite-runtime` python library, which is now deprecated. Its successor is LiteRT and supports the API calls of the previous library. Therefore, you should consider upgrading your Python version and the optimization Framework

## License

MIT
