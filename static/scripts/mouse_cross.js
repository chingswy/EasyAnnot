function drawCrosshair(x, y) {
    const crosshairSize = 10; // 十字架的大小
    ctx.beginPath();
    // 绘制水平线
    ctx.moveTo(x - crosshairSize, y);
    ctx.lineTo(x + crosshairSize, y);
    // 绘制垂直线
    ctx.moveTo(x, y - crosshairSize);
    ctx.lineTo(x, y + crosshairSize);
    ctx.strokeStyle = 'white';
    ctx.lineWidth = 1;
    ctx.stroke();
}

canvas.addEventListener('mousemove', function(event) {
    if (!imageLoaded) return;

    const rect = canvas.getBoundingClientRect();
    const scaleX = image.naturalWidth / rect.width;
    const scaleY = image.naturalHeight / rect.height;

    const x = (event.clientX - rect.left) * scaleX;
    const y = (event.clientY - rect.top) * scaleY;

    // 在绘制十字架前，清除之前的画布内容
    ctx.clearRect(0, 0, canvas.width, canvas.height);
    // 重新绘制图像
    ctx.drawImage(image, 0, 0, canvas.width, canvas.height);

    // 绘制鼠标移动的十字架
    drawCrosshair(x/scaleX, y/scaleY);

    // 绘制标注
    switch (annotationMode) {
        case 'line':
            // 如果定义了起点，则绘制起点到当前位置的线
            if (startPoint) {
                drawLine(startPoint.x/scaleX, startPoint.y/scaleY, x/scaleX, y/scaleY);
            }
            break;
        case 'box':
            // 如果定义了起点，则绘制矩形预览
            if (startPoint) {
                drawBox(startPoint.x/scaleX, startPoint.y/scaleY, x/scaleX, y/scaleY);
            }
            break;
        // 不需要为'point'模式绘制任何特殊的预览
    }
});
