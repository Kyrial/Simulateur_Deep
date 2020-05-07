

function genereSVG(longueur, hauteur,div,id=''){
d3.select('#'+div).append('svg').attr("width",longueur).attr("height",hauteur).attr('class','miaou').attr('id',id);
}


function traceCourbe(Xdep, Ydep, Xarr, Yarr,classe="",numId=""){
	//test ajout DOM
//d3.select('svg').append('circle').attr('cx',50).attr('cy',50).attr('r',40).attr('fill','blue')
	Xmid=(Xarr+Xdep)/2;
	d3.select('svg')
	.append('path')
	.attr('d','M'+Xdep+' '+Ydep+' C '+Xmid+' '+ Ydep+','+Xmid+' '+Yarr+', '+Xarr+' '+Yarr)
	.attr('stroke',"black")
	.attr('fill','none')
	.attr('class',classe+' courbe' )

	.attr('id',numId)
	.attr('stroke-width',1)
	.attr('poid',1)
	/*.attr('onmouseover', "console.log('miaou')")*/

//	.attr('stroke-dasharray', 3)
  	//.attr('animation', 'dash 5s linear')

	
	//d3.select('svg').append('path').attr('d','m50,50 C250 50,250 500 500 500').attr('stroke',"black").attr('fill','none')
/*<circle cx="50" cy="50" r="40" fill="blue" />*/
}





function parcoursLayerInverse(layerCible,cible, callback,layerArrive){
	
	for(let element of document.getElementsByClassName(layerCible)){
		let xdep= parseInt(element.getAttribute('x'),10)+parseInt(element.getAttribute('width'))
		let ydep= parseInt(element.getAttribute('y'))+(parseInt(element.getAttribute('width'))/2)

		let xarr= parseInt(cible.getAttribute('x'))
		let yarr= parseInt(cible.getAttribute('y'))+(parseInt(cible.getAttribute('width'))/2)				
		callback(xdep,ydep,xarr,yarr,"courbe"+layerCible+layerArrive,element.getAttribute('id')+"et"+cible.getAttribute('id'))
	}
}

function parcoursLayer(layerCible,layerArrive, callback){
	for(let i of document.getElementsByClassName(layerCible)){
		parcoursTraceCourbe(i,layerArrive,callback,layerCible)
	}
}




function parcoursTraceCourbe(element,layerArrive, callback,layerCible){
	//for(let i of document.getElementsByClassName(element)){
		//d3.selectAll('.layer1').attr('stroke','green')
		
		
		let xdep= parseInt(element.getAttribute('x'),10)+parseInt(element.getAttribute('width'))
		let ydep= parseInt(element.getAttribute('y'))+(parseInt(element.getAttribute('width'))/2)
		console.log("element "+element+" et "+layerArrive)
		for(let j of document.getElementsByClassName(layerArrive)){

			let xarr= parseInt(j.getAttribute('x'))
			let yarr= parseInt(j.getAttribute('y'))+(parseInt(j.getAttribute('width'))/2)

			
			
			callback(xdep,ydep,xarr,yarr,"courbe"+layerCible+layerArrive,element.getAttribute('id')+"et"+j.getAttribute('id'))



		}

	//}

}

function couleurCourbe(poids){
var val=Math.floor(poids*20)

switch(true) {
  case (poids < 0) : //bleu

    return "rgb("+(100+val)+", "+(149+val)+", 240)"
    break;
  case (poids > 0 ): //rouge
     return "rgb(240, "+(149-val)+", "+(100-val)+")"
    break;
  default:
    return "rgb(0, 100, 0)" //vert
} 

/*switch(true) {
  case (poids < 0) : //bleu
    return "rgb(100, 149, 240)"
    break;
  case (poids > 0 ): //rouge
     return "rgb(240, 149, 100)"
    break;
  default:
    return "rgb(0, 100, 0)" //vert
} */
} 



console.log("miaou")
//genereSVG(1000,1000,'body')
/*
traceCourbe(50,200,500,100)
traceCourbe(50,200,500,300)
traceCourbe(50,200,500,500)

traceCourbe(50,400,500,100,'layer1')
traceCourbe(50,400,500,300,'layer1')
traceCourbe(50,400,500,500,'layer1')
*/

//d3.selectAll() pour tout selectionner

//document.getElementById("body").classList.add("layer1")
//d3.selectAll('.layer1').attr('stroke','green')
/*
ajoutNcarre(0,2,100,'layer1')
ajoutNcarre(0,4,300,'layer2')
ajoutNcarre(0,2,500,'layer3')
ajoutNcarre(0,3,700,'layer4')
ajoutNcarre(0,4,900,'layer5')
parcoursLayer("layer1","layer2",traceCourbe)
parcoursLayer("layer2","layer3",traceCourbe)
parcoursLayer("layer3","layer4",traceCourbe)
parcoursLayer("layer4","layer5",traceCourbe)*/

console.log('malika, tu peux pas test :) ')
/*
var elem = document.getElementsByClassName('courbe');
var elem = document.getElementsByClassName('CarreLayer2');
var elem = document.getElementById('1CarreLayer2');
//console.log("elem                      "+elem[0])
    elem.onmouseover = function() {
        console.log("coucou")
    };

*/




