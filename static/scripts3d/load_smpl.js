import * as THREE from 'three';

async function load_smpl_with_shapes(shapes){
    const urls = [
        '/static/assets/dump_smpl/v_template.bin',
        '/static/assets/dump_smpl/faces.bin',
        '/static/assets/dump_smpl/skinWeights.bin',
        '/static/assets/dump_smpl/skinIndice.bin',
        '/static/assets/dump_smpl/keypoints.bin',
    ];
    console.log(shapes.length);
    const geometry = new THREE.BufferGeometry();
    const buffers = await Promise.all(urls.map(url => fetch(url).then(response => response.arrayBuffer())));
    const v_template = new Float32Array(buffers[0]);
    const offsets = await Promise.all(
        shapes.map((_, i) => fetch('/static/assets/dump_smpl/shapeoffset_'+i+'.bin').then(response => response.arrayBuffer()).then(buffer => new Float32Array(buffer)))
    )
    offsets.forEach((offset, i) => {
        for (let j = 0; j < v_template.length / 3; j++) {
            v_template[3*j] += offset[3*j] * shapes[i];
            v_template[3*j+1] += offset[3*j+1] * shapes[i];
            v_template[3*j+2] += offset[3*j+2] * shapes[i];
        }
    });
    const faces = new Uint16Array(buffers[1]);
    const skinWeights = new Float32Array(buffers[2]);
    const skinIndices = new Uint16Array(buffers[3]);
    const keypoints = new Float32Array(buffers[4]);
    // edges包含骨架链接关系
    const edges = [-1, 0, 0, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 9, 9, 12, 13, 14, 16, 17, 18, 19, 20, 21];
    // 假设 jointPositions 是一个 J x 3 的数组，每个元素是一个包含 X, Y, Z 坐标的数组
    var rootBone = new THREE.Bone();
    rootBone.position.set(keypoints[0], keypoints[1], keypoints[2]);
    // scene.add(rootBone);
    var bones = [rootBone];
    // 创建骨骼
    for (let i = 1; i < keypoints.length / 3; i++) {
        const bone = new THREE.Bone();
        const parentIndex = edges[i];
        bone.position.set(
            keypoints[3 * i] - keypoints[3 * parentIndex],
            keypoints[3 * i + 1] - keypoints[3 * parentIndex + 1],
            keypoints[3 * i + 2] - keypoints[3 * parentIndex + 2]);
        console.log(i, bone.position);
        bones.push(bone);
        bones[parentIndex].add(bone);
    }
    var skeleton = new THREE.Skeleton(bones);
    geometry.setIndex(new THREE.BufferAttribute(faces, 1));
    geometry.setAttribute('position', new THREE.BufferAttribute(v_template, 3));
    geometry.setAttribute('skinIndex', new THREE.BufferAttribute(skinIndices, 4));
    geometry.setAttribute('skinWeight', new THREE.BufferAttribute(skinWeights, 4));
    geometry.computeVertexNormals();
    console.log(geometry);
    // const material = new THREE.MeshBasicMaterial({ color: 0x00ff00, skinning: true, side: THREE.DoubleSide, });
    const material = new THREE.MeshStandardMaterial({
        color: 0xffffff,
        skinning: true,
        side: THREE.DoubleSide
    });
    var mesh = new THREE.SkinnedMesh(geometry, material);
    mesh.castShadow = true;
    mesh.receiveShadow = true;
    mesh.add(bones[0]);
    mesh.bind(skeleton);
    bones[0].rotation.x = Math.PI / 2;
    bones[0].position.z = 1.1;
    return { bones, skeleton, mesh };
}

function reshapeArrayTo2D(float32Array, rows) {
    const twoDArray = [];
    const cols = float32Array.length / rows;
    for (let i = 0; i < rows; i++) {
        const row = new Float32Array(cols);
        for (let j = 0; j < cols; j++) {
            row[j] = float32Array[i * cols + j];
        }
        twoDArray.push(row);
    }
    return twoDArray;
}


export { load_smpl_with_shapes };