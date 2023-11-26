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

WE MADE CUSTOM TRAIING SET MORE REPRESENTIVE OF THE DIFFERENT ATTACKS BUT WAY SMALLER THAN THE DATASET TO REDUCE COMPUTATION TIME AND MASS RAM USAGE

| Label            | Length |
| ---------------- | ------ |
| normal.          | 97278  |
| buffer_overflow. | 30     |
| loadmodule.      | 9      |
| perl.            | 3      |
| neptune.         | 107201 |
| smurf.           | 280790 |
| guess_passwd.    | 53     |
| pod.             | 264    |
| teardrop.        | 979    |
| portsweep.       | 1040   |
| ipsweep.         | 1247   |
| land.            | 21     |
| ftp_write.       | 8      |
| back.            | 2203   |
| imap.            | 12     |
| satan.           | 1589   |
| phf.             | 4      |
| nmap.            | 231    |
| multihop.        | 7      |
| warezmaster.     | 20     |
| warezclient.     | 1020   |
| spy.             | 2      |
| rootkit.         | 10     |

# To do

kmean no similarity

Try hcluster, mean shift, db scan, spectr, with small set and the 3 encoding to build data.

Expliquer que on veut chercher des anomalies pour les clusterise ! Détection d'inconnues!
expliquer pk 100 cluster dans kmean

mettre un CNN pour demontrer la baseline de classification

Hyperparameter tuning

Rq1 encodage?

Rq2 model cluster?

Rq3 prediction
