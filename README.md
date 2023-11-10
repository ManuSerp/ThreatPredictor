# ThreatPredictor

## Install

**build.sh**
This script setup the data folder for running the code.
Can be safely deleted after running.
build.sh download the 10% percent dataset.
Use `-a` option for full dataset

**requirements.txt**
This file lists all the Python package the project depends on.
Dependency can be install with `python install -r requirements.txt`
It is recommanded to use a virutal environment to avoid conflicts with system-wide Python pakcages.

## Current issue

ACtually I have some issue whoth one hot encoding of the symbolic data

- It uses a lot OF SPACE. 4.5 GB pour le 10% dataset donc ça serait 45GB pour le dataset total

Je vais l'optimiser:

- Sinon au lieu de **OHE** on pourrait utiliser le **binary encoding** qui rajoute seulement ln2(nb_possible) colonne au lieu de nb_possible

  **_FAIT. On passe de 4.5G a 1.6G ce qui est deja bien mieux._**

- Utilisation de sparce matrices. Je vise X100 en reduction de memoire -> respectivment 45MB et 450MB ce qui serait bien acceptable. Meme si on a que x5 en reduction c'est ok.

**_FAIT on est a 100MB _**
