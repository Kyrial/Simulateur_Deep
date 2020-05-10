

function add_layer() {
	
	var a=document.getElementById("nbr_colonnes").innerHTML;
	var b = parseInt(a);
	if (b<9) {
		b++;
		a=b+"";
		document.getElementById("nbr_colonnes").innerHTML=a;
		var noeud = document.createElement("div")
		noeud.setAttribute("id","Layer"+a)
		document.getElementById("nbr_layers").appendChild(noeud); //Ajoute layer
		add_button_nbNeurone(a)
	}
	else{}

}


function remove_layer() {
	var a=document.getElementById("nbr_colonnes").innerHTML;
	var b = parseInt(a);
	if (b>0) {
		document.getElementById("nbr_layers").removeChild(document.getElementById("Layer"+b)); //suppr le layer en trop
		//suppression des carrés dans SVG
		//document.getElementsByClassName('CarreLayer'+b).remove()
		$('.CarreLayer'+b).remove();
		//suppression des courbes dans SVG
		$('.courbeCarreLayer'+(b-1)+"CarreLayer"+b).remove();

		b--;
		a=b+"";
		document.getElementById("nbr_colonnes").innerHTML=a;
		
		
	}
	else{}
}



//////

function add_button(parent,name, texte, callback)  {
	//ajout d'un bouton

		var boutonPlus = document.createElement("button")
		boutonPlus.setAttribute("id",name)
		boutonPlus.setAttribute("class","mdl-button mdl-js-button mdl-button--icon")
		boutonPlus.setAttribute("onclick",callback)
		parent.appendChild(boutonPlus);
		//ajout contenant du bouton
		var boutonContenant = document.createElement("i")
		boutonContenant.setAttribute("class","material-icons")
		boutonContenant.textContent=texte
		boutonPlus.appendChild(boutonContenant);

}


function add_button_nbNeurone(numLayer)
{
	//ajout du conteneur des boutons
	document.getElementById("Layer"+numLayer);//.innerHTML=a;

		var decal=150;
		var noeud = document.createElement("div")
		noeud.setAttribute("id","bouton"+numLayer)
		noeud.setAttribute("class","Bouton")
		noeud.setAttribute("style","margin-left:"+decal*(numLayer-1)+"px")
		
		document.getElementById("Layer"+numLayer).appendChild(noeud);
		add_button(noeud,"ajout-neurone"+numLayer, '+',"add_neurone(this,"+numLayer+")")
		add_button(noeud,"suppr-neurone"+numLayer, '-',"remove_neurone(this,"+numLayer+")")

		var compteur = document.createElement("span")
		compteur.setAttribute("id","nbr_neurones"+numLayer)
		compteur.setAttribute("class","span")
		compteur.textContent="2"
		noeud.appendChild(compteur);
		
		ajoutNcarre(0,2,150*numLayer,'CarreLayer'+numLayer)
		parcoursLayer("CarreLayer"+(numLayer-1),"CarreLayer"+(numLayer), traceCourbe)


/*
	<div id=test>
          <button id="ajout-neurone" class="mdl-button mdl-js-button mdl-button--icon" onclick="add_neurone(this)">
          <i class="material-icons">add</i></button>
          <button id="supprime-neurone" class="mdl-button mdl-js-button mdl-button--icon" onclick="remove_layer(this)"><i class="material-icons">remove</i></button>
        </div>
        <span id="nbr_neurones">1</span> neurones*/

    }


///ajoute neurone dans un layer
function add_neurone(idneurone,numLayer) {


	var idparent=$('#'+idneurone.id).parent().parent().attr('id')
	var parent=document.getElementById(idparent)


	//console.log($('#'+idneurone.id).parent().parent().attr('id')) 
	var noeud=parent.getElementsByClassName("span")


	var a=noeud[0].innerHTML;
	var b = parseInt(a);
	if (b<10) {
		b++;
		a=b+"";
		noeud[0].innerHTML=a;
	
		ajoutNcarre(b-1,b,150*numLayer,'CarreLayer'+numLayer)


		parcoursTraceCourbe(document.getElementById("id"+b+"CarreLayer"+numLayer),"CarreLayer"+(numLayer+1),traceCourbe,"CarreLayer"+(numLayer))
		parcoursLayerInverse("CarreLayer"+(numLayer-1),document.getElementById("id"+b+"CarreLayer"+numLayer),traceCourbe,"CarreLayer"+(numLayer))
}
}

function remove_neurone(idneurone,numLayer){
	var idparent=$('#'+idneurone.id).parent().parent().attr('id')
	var parent=document.getElementById(idparent)

	
	//console.log($('#'+idneurone.id).parent().parent().attr('id')) 
	var noeud=parent.getElementsByClassName("span")

	var a=noeud[0].innerHTML;
	var b = parseInt(a);
		if (b>1) {
		b--;
		a=b+"";
		noeud[0].innerHTML=a;
		
		//suppression des carrés dans SVG

		$('#id'+(b+1)+'CarreLayer'+numLayer).remove();
		
		//suppression des courbe dans SVG
		for(let j of document.getElementsByClassName("CarreLayer"+(numLayer+1))){

			$('#id'+(b+1)+'CarreLayer'+numLayer+'et'+j.getAttribute('id')).remove();
		}
		for(let j of document.getElementsByClassName("CarreLayer"+(numLayer-1))){

			$('#'+j.getAttribute('id')+'etid'+(b+1)+'CarreLayer'+numLayer).remove();			

		}	
	} 	
}
	




	var a =document.getElementById("nbr_colonnes").innerHTML;
	document.getElementById("nbr_colonnes").innerHTML=a;
	var noeud = document.createElement("div")
	noeud.setAttribute("id","Layer"+a)
	document.getElementById("nbr_layers").appendChild(noeud); //Ajoute layer
	add_button_nbNeurone(a)


add_neurone(document.getElementById('ajout-neurone1'),'1')
add_layer()
add_neurone(document.getElementById('ajout-neurone2'),'2')
add_neurone(document.getElementById('ajout-neurone2'),'2')
add_neurone(document.getElementById('ajout-neurone2'),'2')
add_layer()
remove_neurone(document.getElementById('suppr-neurone3'),'3')


