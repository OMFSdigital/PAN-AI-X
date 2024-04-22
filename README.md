## PAN-AI-X

This repository contains the Python implementation of the *Insights into Predicting Tooth Extraction from Panoramic Dental Images: Artificial Intelligence vs. Dentists* Paper.

### Models

The different [models](https://github.com/OMFSdigital/PAN-AI-X/tree/usage_example/models) depicted in the image above that were obtained from training with different image % margins can be found under models.
The [2% margin model](https://github.com/OMFSdigital/PAN-AI-X/blob/usage_example/models/panaix_2.pth) performed best during testing and should be used to perform inference.

### Installation

First, you will need to install CUDA on your machine.
This code has been developed with Python 3.11 and CUDA 11.8.

1. `python -m venv panaix`
2. Activate the virtual environment:
    - On Windows: `panaix/Scripts/activate`
    - On Linux: `source panaix/bin/activate`
3. Install an appropriate version of torch, torchvision, and CUDA.
4. `pip install -r requirements.txt`


### Usage

```python
# Example usage:
weights_path = '/.../model.pth'
image_path = '/.../image.png'
train_dataset_path = '/.../train_dir'
val_dataset_path = '/.../val_dir'
test_dataset_path = '/.../test_dir'

# Create an instance of the PAN_AI_X toothpredictor
tooth_predictor = PAN_AI_X(weights_path)

# Use predict method for inference
prediction = tooth_predictor.predict(image_path)
print(prediction)

# Use train method for training
tooth_predictor.train(train_dataset_path, val_dataset_path, test_dataset_path)
```

### Training Dataset Setup

### Dataset directory structure

Datasets must follow the regular pytorch training guidelines for simple classification tasks. Images for each class therefore have to be stored in a seperate directory. The structure of the training datasets should look similar to this:

    datasets/
    ├── train
	│   ├── 00WP
	│   │   ├── preserve0001.png
	│   │   ├── ...
	│   └── 01EX
	│       ├── extract0001.png
	│       ├── ...
    ├── val
	│   ├── 00WP
	│   │   ├── preserve0900.png
	│   │   ├── ...
	│   └── 01EX
	│       ├── extract0100.png
	│       ├── ...
    └── test
	    ├── 00WP
	    │   ├── preserve1000.png
	    │   ├── ...
	    └── 01EX
	        ├── extract0110.png
	        ├── ...
	 
An example image for an extracted tooth would be this tooth that is also shown in the figures of our publication.

<p align="center">
	<img style="margin-left: auto; margin-right: auto;" alt="Example Tooth Margin 2%" src="https://github.com/OMFSdigital/PAN-AI-X/assets/38540238/9d3d68e1-36de-4cd1-995a-bb260c7f55b3">
</p>

### How to cite

```
@article{pan-ai-x2024,
    title={Insights into Predicting Tooth Extraction from Panoramic Dental Images: Artificial Intelligence vs. Dentists}, 
    author={Ila Motmaen and Kunpeng Xie and Leon Schönbrunn and Jeff Berens and Kim Grunert and Anna Maria Plum and Johannes Raufeisen and Andre Ferreira and Alexander Hermans and Jan Egger and Frank Hölzle and Daniel Truhn and Behrus Puladi},
    year={2024},
}
```
