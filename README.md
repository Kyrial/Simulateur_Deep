------ Simulateur Deep -----


Groupe du Projet de Programmation :
- Maël Bonneaud
- Malika Lin-Wee-Kuan
- Melvin Bardin


		Prérequis pour l'execution du projet:
		
- Python 3.6 ou plus récent (projet testé uniquement sur python 3.6, 3.7 et 3.8)
- les librairies suivante sont nécéssaires:
	-Flask; numpy; sklearn; matplotlib; panda; seaborn; sys; json; math
	-installation via le terminal via la commande pip install <nomLib>  ou  pip3 install <nomLib>

		
	
		Execution du projet:
lancé le serveur:
py view2.py   ou   python view2.py

Ouvrez un navigateur et alle a l'addresse
http://127.0.0.1:5000/index.html/

Le site a été testé sur les navigateur:
- Chrome 
- Mozilla Firefox.

le site a été testé sur les systèmes d'exploitation:
- Linux (Xubuntu)
- Window 10
- Mac (mojave 10.14)




		Composition du site Web:
- Une zone pour entrer les paramètres
- Un panneau de controle pour verifier le statue du serveur, le nombre d'époque courante ainsi que permettre d'executer l'animation
- le reseau de neurone
- un repère où se dessine les courbes
- Un cadre affichant l'image lorsque l'animation a terminer

		
		Utilisation:
- Configurer votre réseau comme bon vous semble (maximum 9 layer)
		ATTENTION: la dernière layer doit posseder un unique neurone
- Choisisser vos préférence de calcul dans les paramètres
- Appyer sur "Load" lorsque vous etes satisfait
- lorsque ceci fait dans le panneau de controle 'status:ready' passe en 'Loading..' indiquand que le serveur travaille
		ATTENTION: evitez de "load" lorsque le status est "Loading...", le serveur pourrait planter
- lorsque les calculs sont finis et que les données on été retourné au site, 'status: Loading..' redevient "status: Ready"
- vous pouvez maintenant décidé d'avancé etape par étape en appuyant sur "step + 1" ou lancer l'animation avec "rush all epochs"

- le réseau de neurone s'anime, les courbes de presision et de coût se déssine et lorsque ceci est finis, le résultat via une image s'affiche.

