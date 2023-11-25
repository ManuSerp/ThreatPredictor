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

## What is done ?

- We have parsing with binary encoding of symbolic data
- We have a FeatureReduction class which implements TruncatedSVD

## Current issue

Actually I have some issue with DBSCAN and SpectralClustering which ask to many ressources (eg. RAM and hard disk memory) even if we reduced the number of features.

ACtually I have some issue whoth one hot encoding of the symbolic data

- It uses a lot OF SPACE. 4.5 GB pour le 10% dataset donc ça serait 45GB pour le dataset total

Je vais l'optimiser:

- Sinon au lieu de **OHE** on pourrait utiliser le **binary encoding** qui rajoute seulement ln2(nb_possible) colonne au lieu de nb_possible

  **_FAIT. On passe de 4.5G a 1.6G ce qui est deja bien mieux._**

- Utilisation de sparce matrices. Je vise X100 en reduction de memoire -> respectivment 45MB et 450MB ce qui serait bien acceptable. Meme si on a que x5 en reduction c'est ok.

**_FAIT on est a 100MB _**

- Maybe try SelectKBest with chi2 or f_classif

As we can see in kmean results some attack are very well detected and some are not.

MAKE CUSTOM TRAIING SET MORE REPRESENTIVE OF THE DIFFERENT ATTACKS BUT WAY SMALLER THAN THE DATASET TO REDUCE COMPUTATION TIME AND MASS RAM USAGE


# To do

kmean no similarity

build results for freq encoding

Try hcluster, mean shift, db scan, spectr, with small set and the 2 encoding to build data.


Check similarité
Faire le target encoding 


Expliquer que on veut chercher des anomalies pour les clusterise ! Détection d'inconnues!

Faire un rate de normal/pas normal  ce qui montrera la capacité a trouver des anomalies

Rq1 encodage?

Rq2 model cluster?

Rq3 prediction