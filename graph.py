import matplotlib.pyplot as plt
import os

name = ['FFT_VECT_TIME.txt', 'FFT_OMP_FIRST_TIME.txt', 'FFT_NAIVE_TIME.txt']

# Fonction pour extraire les données d'un fichier
def extract_data(filename):
    with open(filename, 'r') as file:
        lines = file.readlines()

    # Extraction des données de chaque ligne
    data = []
    for line in lines:
        parts = line.strip().split()
        data.append(float(parts[0]))

    return data

# Extraction des données de chaque fichier
y = extract_data(name[0])
y1 = extract_data(name[1])
y2 = extract_data(name[2])

# Création du répertoire de sortie s'il n'existe pas
output_directory = 'Graph_of_time'
if not os.path.exists(output_directory):
    os.makedirs(output_directory)

# Création de l'axe des x en fonction de la taille des données
x = [i for i in range(0, len(y))]
# Utiliser x2 dans les plots pour avoir en base 2
x2 = [2**i for i in range(0, len(y))]

# Tracé du graphique avec les trois ensembles de données
plt.plot(x, y, linestyle='-', color='b', label=name[0])
plt.plot(x, y1, linestyle='-', color='g', label=name[1])
plt.plot(x, y2, linestyle='-', color='r', label=name[2])

# Enlever commentaire pour avoir en base 2
#plt.yscale('log', base=2)
#plt.xscale('log', base=2)

# Ajout des étiquettes et du titre
plt.title('Graph of time/size')
plt.xlabel('Size')
plt.ylabel('Time in Seconds')

# Ajout de la légende
plt.legend()

# Sauvegarde du graphique
plt.savefig(os.path.join(output_directory, 'size.png'))

# Affichage du graphique
plt.show()
