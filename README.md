------ Simulateur Deep -----


Groupe du Projet de Programmation :
- Maël Bonneaud
- Malika Lin-Wee-Kuan
- Melvin Bardin


		Prérequis pour l'execution du projet:
		
- Python 3.6 ou plus récent (projet testé uniquement sur python 3.6, 3.7 et 3.8)
- les librairies suivante sont nécessaires:
	-Flask; numpy; sklearn; matplotlib; panda; seaborn; sys; json; math
	
-installation via le terminal via la commande pip install <nomLib>  ou  pip3 install <nomLib>

		
	
		Execution du projet:
lancer le serveur:
py view2.py   ou   python view2.py

Ouvrez un navigateur et allez à l'adresse
http://127.0.0.1:5000/index.html/

Le site a été testé sur les navigateurs:
- Chrome 
- Mozilla Firefox.

le site a été testé sur les systèmes d'exploitation:
- Linux (Xubuntu)
- Window 10
- Mac (mojave 10.14)




		Composition du site Web:
- Une zone pour entrer les paramètres
- Un panneau de contrôle pour vérifier le statut du serveur, le nombre d'époques courantes ainsi que permettre d’exécuter l'animation
- Le réseau de neurone
- Un repère où se dessine les courbes
- Un cadre affichant l'image lorsque l'animation a terminée

		
		Utilisation:
- Configurer votre réseau comme bon vous semble (maximum 9 layer)
		ATTENTION: la dernière layer doit posséder un unique neurone
- Choisissez vos préférence de calcul dans les paramètres
- Appuyer sur "Load" lorsque vous êtes satisfait
- Lorsque ceci fait dans le panneau de contrôle 'status:ready' passe en 'Loading..' indiquant que le serveur travaille
		ATTENTION: évitez de "load" lorsque le statut est "Loading...", le serveur pourrait planter!
- Lorsque les calculs sont finis et que les données on été retourné au site, 'status: Loading...' redevient "status: Ready!"
- Vous pouvez maintenant décider d'avancer étape par étape en appuyant sur "step + 1" ou lancer l'animation avec "rush all epochs"

- Le réseau de neurone s'anime, les courbes de precision et de coût se dessine et lorsque ceci est finis, le résultat via une image s'affiche.

