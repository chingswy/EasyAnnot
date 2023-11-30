function drawCross(ctx, x, y, col='white', crosssize=10, linewidth=1) {
    ctx.beginPath();
    // 绘制水平线
    ctx.moveTo(x - crosssize, y);
    ctx.lineTo(x + crosssize, y);
    // 绘制垂直线
    ctx.moveTo(x, y - crosssize);
    ctx.lineTo(x, y + crosssize);
    ctx.strokeStyle = col;
    ctx.lineWidth = linewidth;
    ctx.stroke();
}