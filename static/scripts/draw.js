function drawPoint(x, y) {
    // 设置点的大小
    const pointSize = 5; // 可以根据需要调整

    // 开始绘图
    ctx.fillStyle = '#ff2626'; // 点的颜色
    ctx.beginPath(); // 开始路径绘制
    ctx.arc(x, y, pointSize, 0, Math.PI * 2, true); // 绘制圆形来表示点
    // ctx.fill(); // 完成绘制
}

function drawLine(x1, y1, x2, y2) {
    // 绘制线段
    ctx.beginPath();
    ctx.moveTo(x1, y1);
    ctx.lineTo(x2, y2);
    ctx.strokeStyle = 'blue'; // 线段的颜色
    ctx.lineWidth = 2;
    ctx.stroke();

    // 绘制起点
    drawPointWithColor(x1, y1, 'red'); // 起点使用绿色

    // 绘制终点
    drawPointWithColor(x2, y2, 'green'); // 终点使用红色
}

function drawPointWithColor(x, y, color) {
    const pointSize = 5; // 点的大小
    ctx.fillStyle = color; // 使用传入的颜色
    ctx.beginPath();
    ctx.arc(x, y, pointSize, 0, Math.PI * 2, true);
    ctx.fill();
}

function drawBox(x1, y1, x2, y2) {
    // 根据对角线的起点和终点计算矩形的位置和尺寸
    const x = Math.min(x1, x2);
    const y = Math.min(y1, y2);
    const width = Math.abs(x2 - x1);
    const height = Math.abs(y2 - y1);

    // 设置矩形的样式
    ctx.strokeStyle = '#00ff00'; // 边框颜色
    ctx.lineWidth = 2; // 边框宽度
    ctx.beginPath();
    ctx.rect(x, y, width, height); // 绘制矩形
    ctx.stroke(); // 完成边框的绘制
    // 如果需要填充矩形，可以使用ctx.fillStyle设置颜色，然后调用ctx.fill()
}
