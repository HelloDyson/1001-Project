// document.addEventListener("DOMContentLoaded", function(e){
//   startAnimating();
// });

//running at 15 fps
//Flashes for some reason, I don't know why.

var fpsInterval, then, startTime, now, elapsed;
var totalFrames, frameNum, elem;

function startAnimating(){
  fpsInterval = 1000/10; //15 frames per second
  totalFrames = 39; //placeholder
  frameNum = 1; //initialize
  then = Date.now();
  startTime = then;
  elem = document.getElementById("myimage")
  animateBackground();
}

function animateBackground(){
  requestAnimationFrame(animateBackground);

  now = Date.now();
  elapsed = now - then;

  if (elapsed > fpsInterval){
    //update time
    then = now - (elapsed % fpsInterval);
    elem.style.backgroundImage = "url('./sampleoutput/output" + (frameNum) + ".jpg')";

    if (frameNum>totalFrames){
      frameNum = 1;
    }
    else{
      frameNum++;
    }
  }
}