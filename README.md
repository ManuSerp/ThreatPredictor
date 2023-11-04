# ThreatPredictor

## Install

**build.sh**
This script setup the data folder for running the code.
Can be safely deleted after running.
build.sh download the 10% percent dataset.
Use `-a` option for full dataset



ACtually I have some issue whoth one hot encoding of the symbolic data

- It uses a lot OF SPACE. 4.5 GB pour le 10% dataset donc ça serait 45GB pour le dataset total

Je vais l'optimiser:
- Utilisation de sparce matrices. Je vise X100 en reduction de memoire -> respectivment 45MB et 450MB ce qui serait bien acceptable. Meme si on a que x5 en reduction c'est ok.
- Il ya des fois des categorie symbolique qui sont en fait 0 ou 1. Peut etre pas besoin de faire du **OHE** dessus. A voir si ça n'introduit pas une relation ordinale artificielle non voulu entre eux ? je ne pense pas car ce symbolisme est pour moi du oui non donc comme une **OHE**

- Sinon au lieu de **OHE** on pourrait utiliser le **binary encoding** qui rajoute seulement ln2(nb_possible) colonne au lieu de nb_possible
- Ou encore tester le **frequency encoding** pour donner une valeur relevant numerique
