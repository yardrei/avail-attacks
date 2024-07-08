# Stealthy Training-time Availability Attacks

### Quick start:
* Use python 3.9
* `pip install -r requirements.txt`
* `python main.py`


### Notes:
* formatter used: `black` (default arguments)

### To run attack baseline:
`python main.py --use-noise-dl 1 --noise-fraction 0.1`


## Running examples:
Run simple train, no malicious loader: 
```
python main.py
```
Run simple PGD (untargeted):
```
python main.py --use-pgd True --attack-epochs 30 --pgd-step-size 0.05
```
Run targeted PGD, with a different selection method:
```
python main.py --use-pgd True --pgd-targeted 2 --pgd-samples-selection smallest_model_loss
```
Run CW attack:
```
python main.py --use-cw True --cw-c-constant 10
```
Run grad exploder:
```
python main.py --epochs 10 --use-grad-exploder --num-idlg-batches 1 --pgd-samples-selection smallest_model_loss
```
Use PGD cifar dataset + VGG model architecture:
```
python main.py --use-pgd True --dataset cifar --model-number 1
```
Use PGD cifar dataset + ResNet model architecture:
```
python main.py --use-pgd True --dataset cifar --model-number 2
```
Run black box NES estimation PGD attack:
```
python main.py --use-pgd True --black-box True
```
