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
let images = [];

function loadImage() {
    const image = new Image();
    image.onload = function() {
        setCanvasSize(image); // 初始化时设置画布大小
        ctx.clearRect(0, 0, canvas.width, canvas.height);
        ctx.drawImage(image, 0, 0, canvas.width, canvas.height);
    }
    image.src = images[currentIndex]; // 替换为你的图片路径
    pageIndexInput.value = currentIndex + 1;
    imageSlider.value = currentIndex + 1;
    // image.src = `/images/${folder}/${images[currentIndex]}`;
    return image;
}

let folder = '511-balance';

const pageIndexInput = document.getElementById('pageIndex');
const imageSlider = document.getElementById('imageSlider');

// 获取图片列表
fetch(`/images/${folder}`)
    .then(response => response.json())
    .then(data => {
        images = data;
        loadImage();
        pageIndexInput.value = currentIndex + 1;
        imageSlider.max = images.length;
    });

document.getElementById('prev').addEventListener('click', () => {
    if (currentIndex > 0) {
        currentIndex--;
        loadImage();
    }
});

document.getElementById('next').addEventListener('click', () => {
    if (currentIndex < images.length - 1) {
        currentIndex++;
        loadImage();
    }
});

document.getElementById('goPage').addEventListener('click', () => {
    let index = parseInt(pageIndexInput.value, 10) - 1;
    if (index >= 0 && index < images.length) {
        currentIndex = index;
        loadImage();
    }
});

imageSlider.addEventListener('input', () => {
    currentIndex = imageSlider.value - 1;
    loadImage();
});

// 添加键盘事件监听
window.addEventListener('keydown', (e) => {
    if (e.key === 'ArrowLeft') { // 左箭头键
        if (currentIndex > 0) {
            currentIndex--;
            loadImage();
        }
    } else if (e.key === 'ArrowRight') { // 右箭头键
        if (currentIndex < images.length - 1) {
            currentIndex++;
            loadImage();
        }
    }
});