

function creerCarre(x, y, w, h,classe='',id){
// x et y sont la position que le carré à dans le svg
// w et h sont la largeur et hauteur du carré

d3.select('svg').append('rect')
		.attr('x',x)
		.attr('y',y)
		.attr('width',w)
		.attr('height',h)
		.attr('opacity','0.5')
		.attr('class',classe)
		.attr('id',id)
		.attr('stroke','black')
		.attr('stroke-width',2)

}


function ajoutNcarre(pos,n, x,nomClasse=''){// passer le layer correspondant en paramètre + recup attr x 
 	let baseY=120;
	let y=100;
	for(let a=pos; a<n; a++){
		creerCarre(x,y*a+baseY,30,30,nomClasse,"id"+(a+1)+nomClasse);
}






}



ajoutNcarre(0,2,0,"CarreLayer0")
//genereSVG(100,100,'carre')
/*creerCarre(20, 20, 30, 30)
ajoutNcarre(0
,5,400,"jj")*/

