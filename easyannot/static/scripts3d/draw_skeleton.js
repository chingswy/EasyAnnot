import * as THREE from 'three';
// 请帮我实现使用three.js可视化骨架的函数，我的输入是一个骨架的每个关键点的坐标，我希望可视化两个东西：1. 在每个关键点的位置处，可视化一个球的mesh；2. 在edge里包含的每条边，可视化一个四棱锥的mesh，这个四棱锥的长度是这条边的长度，四棱锥的方向和这条边的方向一致
const edges = [[ 1,  0],
        [ 2,  1],
        [ 3,  2],
        [ 4,  3],
        [ 5,  1],
        [ 6,  5],
        [ 7,  6],
        [ 8,  1],
        [ 9,  8],
        [10,  9],
        [11, 10],
        [12,  8],
        [13, 12],
        [14, 13],
        [15,  0],
        [16,  0],
        [17, 15],
        [18, 16],
        [19, 14],
        [20, 19],
        [21, 14],
        [22, 11],
        [23, 22],
        [24, 11]];

var geometries = [];

function visualizeSkeleton(keypoints, scene, radius_joint=0.02, radius_limb=0.03) {
    // 创建球体表示关键点
    geometries.forEach(line => {
        scene.remove(line);
        line.geometry.dispose();
        line.material.dispose();
    });
    geometries = [];

    keypoints.forEach(point => {
        const sphereGeometry = new THREE.SphereGeometry(radius_joint, 32, 32); // 半径为 0.5 的球体
        const sphereMaterial = new THREE.MeshStandardMaterial({ color: 0xff0000 });
        const sphere = new THREE.Mesh(sphereGeometry, sphereMaterial);
        sphere.position.set(point[0], point[1], point[2]);
        geometries.push(sphere);
        scene.add(sphere);
    });

    // 创建四棱锥表示边
    edges.forEach(edge => {
        if (keypoints[edge[0]][3] < 0.1 || keypoints[edge[1]][3] < 0.1){
            return;
        }
        const start = keypoints[edge[0]];
        const end = keypoints[edge[1]];

        const direction = new THREE.Vector3().subVectors(
            new THREE.Vector3(end[0], end[1], end[2]), 
            new THREE.Vector3(start[0], start[1], start[2])
        );
        const length = direction.length();

        // 创建一个椭球，其最长轴表示边的长度
        const ellipsoidGeometry = new THREE.SphereGeometry(radius_limb, 32, 32);
        const ellipsoidMaterial = new THREE.MeshStandardMaterial({ color: 0x0000ff });
        const ellipsoid = new THREE.Mesh(ellipsoidGeometry, ellipsoidMaterial);

        // 缩放椭球使其成为一个细长的形状
        ellipsoid.scale.set(1, 1, length / 2 / radius_limb); // 保持x和y的半径为0.05，沿z轴缩放以匹配边的长度

        // 将椭球的位置设置到起点和终点的中点
        ellipsoid.position.set((start[0]+end[0])/2, (start[1]+end[1])/2, (start[2]+end[2])/2);

        // 旋转椭球，使其最长轴指向边的终点
        ellipsoid.lookAt(end[0], end[1], end[2]);

        geometries.push(ellipsoid);
        scene.add(ellipsoid);
    });
}

function visualizeAllSkeleton(infos, scene, radius_joint=0.02, radius_limb=0.03) {
    // 创建球体表示关键点
    geometries.forEach(line => {
        scene.remove(line);
        line.geometry.dispose();
        line.material.dispose();
    });
    geometries = [];

    infos.forEach(info => {
        const keypoints = info.keypoints3d;
        const edges = info.edges;
        const sphereGeometry = new THREE.SphereGeometry(radius_joint, 32, 32); // 半径为 0.5 的球体
        const sphereMaterial = new THREE.MeshStandardMaterial({ color: 0xff0000 });
        keypoints.forEach(point => {
            const sphere = new THREE.Mesh(sphereGeometry, sphereMaterial);
            if(point.length > 3 && point[3] < 0.1){
                return;
            }
            sphere.position.set(point[0], point[1], point[2]);
            geometries.push(sphere);
            scene.add(sphere);
        });

        const ellipsoidGeometry = new THREE.SphereGeometry(radius_limb, 32, 32);
        const ellipsoidMaterial = new THREE.MeshStandardMaterial({ color: 0x0000ff });
        // 创建四棱锥表示边
        edges.forEach(edge => {
            if (edge[0] >= keypoints.length || edge[1] >= keypoints.length){
                return;
            }
            if (keypoints[edge[0]].length > 3 && keypoints[edge[0]][3] < 0.1 || keypoints[edge[1]].length > 3 && keypoints[edge[1]][3] < 0.1){
                return;
            }
            const start = keypoints[edge[0]];
            const end = keypoints[edge[1]];
    
            const direction = new THREE.Vector3().subVectors(
                new THREE.Vector3(end[0], end[1], end[2]), 
                new THREE.Vector3(start[0], start[1], start[2])
            );
            const length = direction.length();
    
            // 创建一个椭球，其最长轴表示边的长度
            const ellipsoid = new THREE.Mesh(ellipsoidGeometry, ellipsoidMaterial);
    
            // 缩放椭球使其成为一个细长的形状
            ellipsoid.scale.set(1, 1, length / 2 / radius_limb); // 保持x和y的半径为0.05，沿z轴缩放以匹配边的长度
    
            // 将椭球的位置设置到起点和终点的中点
            ellipsoid.position.set((start[0]+end[0])/2, (start[1]+end[1])/2, (start[2]+end[2])/2);
    
            // 旋转椭球，使其最长轴指向边的终点
            ellipsoid.lookAt(end[0], end[1], end[2]);
    
            geometries.push(ellipsoid);
            scene.add(ellipsoid);
        });
    })

}

export { visualizeSkeleton, visualizeAllSkeleton };