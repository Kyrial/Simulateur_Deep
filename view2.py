import numpy as np
from sklearn.datasets import make_moons
from sklearn.datasets import make_circles
from sklearn import datasets
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import OneHotEncoder
from matplotlib.colors import ListedColormap
import matplotlib.pyplot as plt
import matplotlib
from sklearn.model_selection import train_test_split 
from sklearn import linear_model
from matplotlib.legend_handler import HandlerLine2D
from sklearn.metrics import accuracy_score
import pandas as pd
import seaborn as sns
import numpy as np

from flask import Flask,request, render_template,jsonify


import sys

import json


#inclusion des fichiers:
from MyNetwork import *




app = Flask(__name__)
#app.config['SEND_FILE_MAX_AGE_DEFAULT'] = 0

@app.route('/index.html/')
def index():
    return render_template('index.html',
                             truc='non',
                             acc_hist= "non",
                             cost_hist="non",
                             epochs='100')
      #obliger de renseigner tout les paramètres sinon bug sur page web

#@app.route('/ ... /<arg>')
#def ...(arg):
#   return 'hello {}'.format(name.capitalize())

@app.route('/index.html/',methods=['POST'])
def ChoisirNetwork():
    #print(request.form["nbr_epochs"])
    #print(request.form["rate"])
    #print(request.form["valid_size"])
    print(request.form["neuronne"].split(","))
    configNeurones=request.form["neuronne"].split(",")
    reponse="non"
    
    if(request.form["Network"]=="Moons"):
        param_hist,acc_hist,cost_hist=makeMoons(int(request.form["nbr_epochs"]),
                          float(request.form["rate"]),
                          float(request.form["valid_size"]),
                          configNeurones)
    if(request.form["Network"]=="Circle"):
        param_hist,acc_hist,cost_hist=makeCircles(int(request.form["nbr_epochs"]),
                            float(request.form["rate"]),
                            float(request.form["valid_size"]),
                            configNeurones)
    if(request.form["Network"]=="GQ"):
        param_hist,acc_hist,cost_hist=makeGaussianQuantiles(int(request.form["nbr_epochs"]),
                            float(request.form["rate"]),
                            float(request.form["valid_size"]),
                            configNeurones)
    #print(acc_hist)
    #print(cost_hist)
    #reponse=json.dumps(reponse)
    #return render_template('index.html',
    #                      truc=param_hist,
     #                      acc_hist=acc_hist,
      #                     cost_hist=cost_hist,
       #                    epochs=request.form["nbr_epochs"])
    #return jsonify(reponse)
    print("traitement terminer")
    #a= input("taper quelque chose pour continuer")
    
    return jsonify(truc = param_hist, acc_hist=acc_hist, cost_hist=cost_hist)


if __name__ == "__main__":
    print(" page d'acceuil: http://127.0.0.1:5000/index.html/ ")
    app.run()


