import * as THREE from 'three';
import { getChessboard, getCoordinate } from './create_ground.js';

function create_plane(scene){
    const planeGeometry = new THREE.PlaneGeometry(20, 20);
    const planeMaterial = new THREE.MeshStandardMaterial({ color: 0x808080 });
    const plane = new THREE.Mesh(planeGeometry, planeMaterial);
    plane.position.y = -1;
    plane.receiveShadow = true; // 让平面接收阴影
    scene.add(plane);
}

function create_cube(scene){
    // 添加立方体 - 您的 mesh
    const cubeGeometry = new THREE.BoxGeometry();
    const cubeMaterial = new THREE.MeshPhongMaterial({ color: 0xffffff });
    const cube = new THREE.Mesh(cubeGeometry, cubeMaterial);
    cube.position.y = 1;
    cube.castShadow = true; // 让立方体投射阴影
    scene.add(cube);
}

function create_scene(scene, camera, renderer, use_ground=true){
    const width = document.querySelector('.container').offsetWidth;
    const height = width;
    camera.position.set(0, -3, 3); // 将相机放置在正Z轴方向，朝向原点
    camera.up.set(0, 0, 1); // 将相机的上方设为Z轴的正方向
    camera.lookAt(new THREE.Vector3(0, 0, 1.5)); // 让相机朝向场景的原点
    
    scene.background = new THREE.Color( 0xeeeeee );
    
    const ambientLight = new THREE.AmbientLight(0xffffff, 1.);
    scene.add(ambientLight);

    renderer.shadowMap.enabled = true;
    renderer.shadowMap.type = THREE.PCFSoftShadowMap; // 使用软阴影
    // 在场景中添加一个光源
    // 添加太阳光（方向光）
    // const directionalLight = new THREE.DirectionalLight(0xffffff, 10); // 白色光源，强度为1
    // directionalLight.position.set(3, 0, 3); // 设置光源位置
    // directionalLight.castShadow = true; // 允许光源投射阴影
    // scene.add(directionalLight);
    const directionalLight = new THREE.DirectionalLight(0xffffff, 2);
    directionalLight.position.set(0, -3, 3);
    directionalLight.castShadow = true;
    // directionalLight.shadow.mapSize.width = 1024;  // 阴影分辨率
    // directionalLight.shadow.mapSize.height = 1024;
    // directionalLight.shadow.camera.near = 0.5;
    // directionalLight.shadow.camera.far = 50;
    scene.add(directionalLight);

    if(use_ground){
        var plane = getChessboard();
        scene.add(plane);

        var coord = getCoordinate(1);
        scene.add(coord);
    }
    // renderer
    // 配置渲染器以支持阴影
    return 0
}

export { create_scene };
