const ctx = canvas.getContext('2d');
var scaleX = 1, scaleY = 1;

const pageIndexInput = document.getElementById('pageIndex');
const imageSlider = document.getElementById('imageSlider');

function init(){
    pageIndexInput.value = currentFrame;
    imageSlider.max = num_images - 1;

    image = new Image();
    image.src = `/send_first_image/${folder_name}`;
    // Once the image has loaded, draw it and then the annotations
    image.onload = function() {
        setCanvasSize(canvas, image); // Set canvas size
        clearAndPlot(ctx, canvas, image);
        scaleX = image.naturalWidth / drawingCanvas.width;
        scaleY = image.naturalHeight / drawingCanvas.height;
    }
}

function load_image(index){
    image = new Image();
    image.src = `/send_i_image/${folder_name}/${index}`;
    // Once the image has loaded, draw it and then the annotations
    image.onload = function() {
        setCanvasSize(canvas, image); // Set canvas size
        clearAndPlot(ctx, canvas, image);
        scaleX = image.naturalWidth / drawingCanvas.width;
        scaleY = image.naturalHeight / drawingCanvas.height;
    }
}

init();

function _update_frame(){
    load_image(currentFrame);
}

window._update_frame = _update_frame;

function updateFrame() {
    currentFrame = Math.max(0, Math.min(currentFrame, num_images - 1));
    imageSlider.value = currentFrame;
    pageIndexInput.value = currentFrame;
    console.log('update to', currentFrame);
    window._update_frame(currentFrame);
}

function changeFrame(offset) {
    currentFrame = currentFrame + offset;
    updateFrame();
}

function changeFrameByText() {
    currentFrame = parseInt(pageIndexInput.value, 10);
    updateFrame();
}

function changeFrameBySlider() {
    currentFrame = parseInt(imageSlider.value, 10);
    updateFrame();
}