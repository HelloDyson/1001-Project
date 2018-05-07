var fs = require('file-system');
var extractFrames = require('ffmpeg-extract-frames');

//-----------document event listeners----------------//



document.addEventListener("DOMContentLoaded", function(e){
	
	//click events
	document.addEventListener("click", e =>{
		if (e.target.nodeName == "BUTTON"){
			console.log("clicked the button");
		}
	});

	//run animation after all the frames have been extracted/processed.
  	startAnimating();
});


function extractFrames(){
	var videosrc = document.getElementById('video');

	//videosrc should be an html element
	if (videosrc.src != ""){
		var totalTime = videosrc.duration*1000; //convert to milliseconds
		var timeStep = Math.floor(1000/15); //roughly 15 fps

		var timeOffsets = [];
		//time array
		for (var i=0; (timeStep*i)<totalTime; i++){
			timeOffsets.push(timeStep*i);
		}

		//Do the deed
		extractFrames({
			input: videosrc.src, 
			output: './input_frames/frame-%i.jpg', 
			offsets: timeOffsets
		}).then(console.log("done"));
	}
}


//Animation in #myimage div running at 15 fps
//Flashes for some reason, I don't know why.

var fpsInterval, then, startTime, now, elapsed;
var totalFrames, frameNum, elem;

function startAnimating(){
	fpsInterval = 1000/15; //15 frames per second
	totalFrames = fs.readdirSync("./image src").length; //placeholder
	frameNum = 1; //initialize
	then = Date.now();
	startTime = then;
	elem = document.querySelector("#myimage")
	animateBackground();
}

function animateBackground(){
	requestAnimationFrame(animateBackground);

	now = Date.now();
	elapsed = now - then;

	if (elapsed > fpsInterval){
		//update time
		then = now - (elapsed % fpsInterval);
		elem.style.backgroundImage = "url('./image src/frame-" + (frameNum) + ".jpg')";

		if (frameNum>totalFrames){
			frameNum = 1;
		}
		else{
			frameNum++;
		}
	}
}
