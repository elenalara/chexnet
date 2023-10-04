# CheXNet Replica

This project implements a replica of the CheXNet model, a convolutional neural network trained for detecting various lung conditions from X-ray images, following the implementation presented [here](https://github.com/zoogzog/chexnet).

## Project Description

The `train.py` script contains the implementation of the CheXNet model and its training. The model is trained using X-ray image data and corresponding labels for 14 classes of lung conditions.

## Project Structure

- `train.py`: Main script that includes the model definition, training functions, and evaluation.
- `split/`: Directory containing the split of data into training, validation, and test sets.
- `models/`: Directory where trained models are saved.

## Usage Instructions

1. Download the [ChestX-ray14 database](https://nihcc.app.box.com/v/ChestXray-NIHCC/folder/37178474737) in the `data_chexnet/` folder.
2. Unpack files in separate directories (e.g. images_01.tar.gz into images_001)
3. Ensure you have Python and the required libraries installed.
4. Run the `train.py` script to train and evaluate the CheXNet model.
5. Trained models will be saved in the `models/` directory and training logs in the `logs/` directory.

## Dependencies

The project requires the following dependencies:

- Python 3.x
- PyTorch
- scikit-learn
- torchvision
- PIL (Pillow)
- numpy
- tensorboard