# evoman

This repo contains some Evolutionary Computing experiments applied on the [EvoMan Framework](https://github.com/karinemiras/evoman_framework)

# Python Virtual Environment

First please install python 3.8

```sh
# MacOS
brew instal python@3.8
```

```sh
# Create virtual environment
python3.8 -m venv venv

# Activate virtual environment
source venv/bin/activate

# Install requirements
pip install -r requirements.txt
```

# Run

### To train the generalist agent 

```sh
python z_generalist_main.py train
python z_generalist_main.py train_results
```

### To test the generalist agent already trained

```sh
python z_generalist_main.py test_all
python z_generalist_main.py test_all_results
```

### To plot the generalist training and test data

```sh
python z_generalist_main.py visuals
```

### To generate a file containing the best of the best

```sh
python z_generalist_main.py best
```

Results will be saved in the results' folder. 
Some results are already visible in results/generalist_final_1.
If you run the training phase the results will be overwritten, if you don't want to do it, change the base dir in the main file.

