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

function drawRect(ctx, x, y, x1, y1, col='white', linewidth=1){
    ctx.beginPath();
    const width = Math.abs(x1 - x);
    const height = Math.abs(y1 - y);
    ctx.rect(x, y, width, height);
    ctx.strokeStyle = col;
    ctx.lineWidth = linewidth;
    ctx.stroke();
}