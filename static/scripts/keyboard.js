function update_index(new_index){
    if(new_index >=0 && new_index < images.length){
        currentIndex = new_index;
        pageIndexInput.value = new_index + 1;
        imageSlider.value = new_index + 1;
        window._update_index();
    }
}

document.getElementById('prev').addEventListener('click', () => {
    update_index(currentIndex - 1);
});

document.getElementById('next').addEventListener('click', () => {
    update_index(currentIndex + 1);
});

document.getElementById('goPage').addEventListener('click', () => {
    let index = parseInt(pageIndexInput.value, 10) - 1;
    update_index(index);
});

imageSlider.addEventListener('input', () => {
    update_index(imageSlider.value - 1);
});

// 添加键盘事件监听
window.addEventListener('keydown', (e) => {
    if (e.key === 'ArrowLeft') { // 左箭头键
        update_index(currentIndex - 1);
    } else if (e.key === 'ArrowRight') { // 右箭头键
        update_index(currentIndex + 1);
    }
});