import * as THREE from 'three';

function getChessboard(grid_size=5, divisions = 10, white='#ffffff', black='#444444', texture_size = 1024) {
    // 创建棋盘格纹理
    var canvas = document.createElement('canvas');
    canvas.width = canvas.height = texture_size;
    var context = canvas.getContext('2d');

    var step = texture_size / divisions;
    for (var i = 0; i < divisions; i++) {
        for (var j = 0; j < divisions; j++) {
            context.fillStyle = (i + j) % 2 === 0 ? white : black;
            context.fillRect(i * step, j * step, step, step);
        }
    }

    var texture = new THREE.CanvasTexture(canvas);
    // 假设 texture 已经被加载
    texture.minFilter = THREE.NearestFilter;
    texture.magFilter = THREE.NearestFilter;
    texture.generateMipmaps = false;
    // 创建平面几何体
    var planeGeometry = new THREE.PlaneGeometry(grid_size, grid_size); // 根据需要调整尺寸

    // 创建材料
    var planeMaterial = new THREE.MeshStandardMaterial({
        map: texture,
        side: THREE.DoubleSide, // 渲染材料的两面
    });

    // 创建网格
    var plane = new THREE.Mesh(planeGeometry, planeMaterial);
    plane.receiveShadow = true;
    plane.rotation.x = -Math.PI; // 旋转以使平面水平

    return plane;
}

function getCoordinate(axisLength) {
    // 创建一个用于存放坐标轴的组
    var axes = new THREE.Group();

    // 定义轴的材料
    var materialX = new THREE.LineBasicMaterial({ color: 0xff0000 }); // 红色 X轴
    var materialY = new THREE.LineBasicMaterial({ color: 0x00ff00 }); // 绿色 Y轴
    var materialZ = new THREE.LineBasicMaterial({ color: 0x0000ff }); // 蓝色 Z轴

    // 创建轴线（X轴，Y轴，Z轴）
    var xAxisGeometry = new THREE.BufferGeometry().setFromPoints([new THREE.Vector3(0, 0, 0), new THREE.Vector3(axisLength, 0, 0)]);
    var yAxisGeometry = new THREE.BufferGeometry().setFromPoints([new THREE.Vector3(0, 0, 0), new THREE.Vector3(0, axisLength, 0)]);
    var zAxisGeometry = new THREE.BufferGeometry().setFromPoints([new THREE.Vector3(0, 0, 0), new THREE.Vector3(0, 0, axisLength)]);

    var xAxis = new THREE.Line(xAxisGeometry, materialX);
    var yAxis = new THREE.Line(yAxisGeometry, materialY);
    var zAxis = new THREE.Line(zAxisGeometry, materialZ);

    // 将轴添加到组中
    axes.add(xAxis);
    axes.add(yAxis);
    axes.add(zAxis);

    return axes;
}

export { getChessboard, getCoordinate };
// 使用示例
// var plane = getChessboard();
// scene.add(plane);
