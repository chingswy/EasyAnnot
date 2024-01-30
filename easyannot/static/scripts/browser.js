const canvas = document.getElementById('imageCanvas');
const ctx = canvas.getContext('2d');
// 设置画布尺寸
function setCanvasSize(image) {
    const width = document.querySelector('.container').offsetWidth;
    canvas.width = width;
    canvas.height = width * (image.naturalHeight / image.naturalWidth); // 保持宽高比
}

window.addEventListener('resize', setCanvasSize); // 调整窗口大小时更新画布大小

let currentIndex = 0;

function loadImage(index) {
    const annotationsUrl = `/annotations/${folder_name}/${images[index]}`; // The URL to your annotations file
    // Set the image src to load it
    // Asynchronously load the annotations JSON
    fetch(annotationsUrl)
        .then(response => response.json())
        .then(data => {
            const image = new Image();
            image.src = `/images/${folder_name}/${images[index]}`;
            // Once the image has loaded, draw it and then the annotations
            image.onload = function() {
                setCanvasSize(image); // Set canvas size
                ctx.clearRect(0, 0, canvas.width, canvas.height);
                ctx.drawImage(image, 0, 0, canvas.width, canvas.height);

                const scaleX = canvas.width / image.naturalWidth;
                const scaleY = canvas.height / image.naturalHeight;
                // Draw the bounding boxes from the annotations
                data.annots.forEach(annot => {
                    const bbox = annot.bbox;
                    ctx.beginPath();
                    ctx.rect(bbox[0] * scaleX, bbox[1] * scaleY, (bbox[2] - bbox[0]) * scaleX, (bbox[3]-bbox[1]) * scaleY);
                    ctx.strokeStyle = getBoxColor(annot.personID); // Get color based on personID
                    ctx.lineWidth = 2; // Set line width if needed
                    ctx.stroke();
                    // plot keypoints
                    const keypoints = annot.keypoints;
                    ctx.fillStyle = getBoxColor(annot.personID); // 点的颜色
                    const pointSize = 5; // 可以根据需要调整
                    for (let i = 0; i < keypoints.length; i += 1) {
                        const x = keypoints[i][0] * scaleX;
                        const y = keypoints[i][1] * scaleY;
                        const score = keypoints[i][2];
                        if(score < 0.1)continue;
                        ctx.beginPath(); // 开始路径绘制
                        ctx.arc(x, y, pointSize, 0, Math.PI * 2, true); // 绘制圆形来表示点
                        ctx.fill(); // 完成绘制
                    }
                });
            }
        })
        .catch(error => {
            console.error('Error fetching annotation data:', error);
        });
    return 0;
}

const pageIndexInput = document.getElementById('pageIndex');
const imageSlider = document.getElementById('imageSlider');

// 获取图片列表
pageIndexInput.value = currentIndex + 1;
imageSlider.max = images.length;
_update_index();

function _update_index(){
    console.log('update in 2d');
    loadImage(currentIndex);
}

window._update_index = _update_index;