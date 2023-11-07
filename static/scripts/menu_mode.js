let annotationMode = 'point'; // 默认模式

document.getElementById('modePoint').addEventListener('click', () => {
    annotationMode = 'point';
});

document.getElementById('modeLine').addEventListener('click', () => {
    annotationMode = 'line';
});

document.getElementById('modeBox').addEventListener('click', () => {
    annotationMode = 'box';
});
