####Requirements:
* Python 3.6.5

To install:
```bash
pip install -r requirements.txt
```

Dogs and cats dataset: [https://www.kaggle.com/c/dogs-vs-cats/data](https://www.kaggle.com/c/dogs-vs-cats/data)

Monkey species dataset: [https://www.kaggle.com/slothkong/10-monkey-species](https://www.kaggle.com/slothkong/10-monkey-species)

Special note on Monkey Species dataset: Contains 10 different species of monkey. In order to specify a specifc class for binary classification, training and validation sets should be made such that one species is one class and all other species are the 2nd class

Experiment scripts:
* dogscats_base.py
* dogscats_ibtl.py
* monkey_base.py
* monkey_ibtl.py

####How to run Dogs vs Cats experiment:
1. Download Dogs vs Cats data: [https://www.kaggle.com/c/dogs-vs-cats/data](https://www.kaggle.com/c/dogs-vs-cats/data)
2. Extract dog training images into data/dogscats/train/dogs
3. Extract cat training images into data/dogscats/train/cats
4. Extract dog validation images into data/dogscats/validation/dogs
5. Extract cat validation images into data/dogscats/validation/cat
6. Run:
```bash
python dogscats_base.py
python dogscats_ibtl.py
```

####How to run Monkey Species experiment:
1. Download Monkey Species data: [https://www.kaggle.com/slothkong/10-monkey-species](https://www.kaggle.com/slothkong/10-monkey-species)
2. Extract monkey training image folders into data/monkey/train
3. Extract monkey validation images into data/monkey/validation
4. Run:
```bash
python monkey_base.py
python monkey_ibtl.py
```