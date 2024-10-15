
---

# XIE JAROD 28710097


# Projet PPAR FFT

## Introduction
Ce projet vise à paralléliser une implémentation séquentielle de la transformée de Fourier rapide (FFT) en utilisant OpenMP et à explorer la vectorisation (SIMD) pour améliorer les performances. L'objectif principal est d'obtenir une exécution efficace sur un seul nœud de calcul.

## Contenu du Projet
Le code source est réparti en plusieurs fichiers :
- `fft_omp_first.c` : Implémentation séquentielle de la FFT avec OpenMP.
- `fft_vect.c` : Version vectorisée de la FFT avec OpenMP et SIMD.
- `utils.c` : Fonctions utilitaires partagées.
- `main.c` : Programme principal pour générer du bruit blanc, effectuer la FFT, appliquer des transformations et produire un fichier audio.

## Compilation
### FFT avec OpenMP

Vous avez juste besoin de run le makefile avec :



`make` 

## Utilisation

### FFT avec OpenMP



`./fft_omp --size [TAILLE] --seed [GRAINE] --output [FICHIER_SORTE] 

### FFT Vectorisée avec OpenMP et SIMD


`./fft_vect --size [TAILLE] --seed [GRAINE] --output [FICHIER_SORTE]   

## Options

-   `--size` : Taille des données d'entrée (2^k).
-   `--seed` : Graine pour la génération de bruit blanc.
-   `--output` : Nom du fichier de sortie audio.
-   `--cutoff` : Valeur de coupure pour ajuster les coefficients de Fourier.

## Benchmarking

Pour mesurer les performances du programme, vous pouvez utiliser la commande suivante :
Les résultats sont dans des fichiers textes.

`for i in {1..25}; do ./fft_vect --size $((2**$i)); done` 

Pour avoir les résultats graphiques, il faut d'abord utiliser les 3 commandes suivantes :

`for i in {1..25}; do ./fft_vect --size $((2**$i)); done`

`for i in {1..25}; do ./fft_omp_first --size $((2**$i)); done`

`for i in {1..25}; do ./fft --size $((2**$i)); done`

Puis faites : `python graph.py`, le graphique est dans le répertoire Graph_of_Time
Pour avoir le graphique en base 2 , suivre les instructions en commentaire dans `graph.py`.

## Auteurs

XIE JAROD : Jarod.Xie@etu.sorbonne-universite.fr

