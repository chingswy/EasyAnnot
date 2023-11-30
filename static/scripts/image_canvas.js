function setCanvasSize(canvas, image) {
    const width = canvas.parentNode.offsetWidth;
    canvas.width = width;
    canvas.height = width * (image.naturalHeight / image.naturalWidth); // 保持宽高比
    const rect = canvas.getBoundingClientRect();
}

function clearAndPlot(ctx, canvas, image) {
    ctx.clearRect(0, 0, canvas.width, canvas.height);
    ctx.drawImage(image, 0, 0, canvas.width, canvas.height);
}

function updateImage(image, imageUrl) {
    image.src = imageUrl;
}

function getMousePosition(canvas, event) {
    const rect = canvas.getBoundingClientRect();
    const scaleX = canvas.width / rect.width;    // canvas实际宽度与显示宽度的比例
    const scaleY = canvas.height / rect.height;  // canvas实际高度与显示高度的比例

    return {
        x: (event.clientX - rect.left) * scaleX,  // 调整鼠标X坐标
        y: (event.clientY - rect.top) * scaleY    // 调整鼠标Y坐标
    };
}

function getImagePosition(canvas, image, pos) {
    const scaleX = image.naturalWidth / canvas.width;    // canvas实际宽度与显示宽度的比例
    const scaleY = image.naturalHeight / canvas.height;  // canvas实际高度与显示高度的比例

    return {
        x: pos.x * scaleX,  // 调整鼠标X坐标
        y: pos.y * scaleY    // 调整鼠标Y坐标
    };
}

function image2canvasPosition(canvas, image, pos) {
    const scaleX = image.naturalWidth / canvas.width;    // canvas实际宽度与显示宽度的比例
    const scaleY = image.naturalHeight / canvas.height;  // canvas实际高度与显示高度的比例

    return {
        x: pos.x / scaleX,  // 调整鼠标X坐标
        y: pos.y / scaleY    // 调整鼠标Y坐标
    };
}
