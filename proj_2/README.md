# Klasifikace

https://www.kaggle.com/datasets/prathamtripathi/drug-classification


# Vysledky

V projektu jsem overoval klasifikaci `Drug`, aj.
Vysledky jsou k nalezeni v `run.sh.log` a `run.sh_all.log`.
Datovy soubor je rozdelen na nekolik casti: trenovaci data, testovaci data pro klasifikovanou promennou i pro promenne, ke kterym je klasifikace pouzita. 

Nejlepe se povedlo klasifikovat `Drug` s uspesnosti `1.0`, viz.

```
Input file is " ./drug200.csv "
Target variable is " Drug "
Data:
['Age', 'Sex', 'BP', 'Cholesterol', 'Na_to_K', 'Drug']
Target:
Drug
train
classify
Model Accuracy: 1.0
```

Duvodem je, ze data tomuto skutecne odpovidaji.

Pro overeni jsem model naucil i proti jinym promennym v datovem soubouru. Test klasifikace na modelu natrenovanem na jinych promennych pak plne neodpovidal ocekavani.

