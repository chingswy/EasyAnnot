# EasyAnnot


## 简单的脚本调用

1. 可视化3D骨架序列：

```bash
python scripts/vis_keypoints3d.py --root ${root} --scale 1. --port 2345
```
默认单位是米，如果输入数据是厘米，那么需要设置`--scale 0.01`。

2. 可视化SMPL序列：

```bash
python scripts/vis_smpl.py --root ${root} --port 2345
```

3. 可视化BVH序列

将数据软链接到 `easyannot/static/data/a2gdata`下，然后运行

```bash
python3 scripts/show_bvh_database.py --root easyannot/static/data/a2gdata --port 2357 --debug
```


## Install

TODO: update to pip

```bash
pip install ./
```

## Usage

Visualize all data of a EasyMoCap log

```bash
easyannot easymocap ./
```

Contains:

- Images
- Annots
    - plot bbox, keypoints onto images
- Match log
    - distance
    - matching steps:
- Recon log
    - 3D keyypoints in each frame

### Format

1. Matching log

```python
[
    {
        'indices': [], # len = views, -1 if invalid
        'proposals': [(view, id, conf), ...],  
    }
]
```

## Vis Cameras

```bash
easyannot camera ./
```

## Vis Synchronized Multiple Images

```bash
easyannot images ./images
```

## 标注匹配点

```bash
easyannot match ./ --camera output/calibrate/calibrate_final
```

## 标注vanishing line



## Deprecated


## Usage

### 1. 可视化3D骨架序列：

```python
python3 vis_keypoints3d.py --root ${root} --port 2345
```

${root}目录包含一序列的3D骨架数据，格式为：
```bash
- 000000.json
[
    {
        'id': 0,
        'keypoints3d': [[x0, y0, z0, c0], [x1, y1, z1, c1], ...],
    },
    {
        'id': 1,
        'keypoints3d': [[x0, y0, z0, c0], [x1, y1, z1, c1], ...],
    }
]
- 000001.json
...
```

### 2. 可视化SMPL序列: 

```python
python3 vis_keypoints3d.py --root ${root} --port 2345
```

假设同一段数据里，每个人的shape参数不会发生变化。${root}目录包含一序列的SMPL参数，格式为：
```bash
- 000000.json
[
    {
        'id': 0,
        'shapes': [[s0, s1, s2, s3, ..., s9]],
        'poses': [[p0, p1, p2, p3, ..., p72]],
        'Rh': [[r0, r1, r2]],
        'Th': [[t0, t1, t2]]
    },
    {
        'id': 1,
        'shapes': [[s0, s1, s2, s3, ..., s9]],
        'poses': [[p0, p1, p2, p3, ..., p72]],
        'Rh': [[r0, r1, r2]],
        'Th': [[t0, t1, t2]]
    }
]
- 000001.json
...
```


## 标注部分

启动：

```python
python3 app.py --root ${root} --port 3456
```

### 0. 可视化多个图片文件夹

访问网站`http://127.0.0.1:3456/gallery`

### 1. 标注视频的多个clip

访问网站`http://127.0.0.1:3456/clip`

> prompt: 我正在使用python+flask实现一个数据标注网站，请帮我进行网站设计。我已经正确设置好网站所需的环境，你可以省略这部分，只用回答功能部分代码。
我希望实现的功能：我将一个视频按顺序的拆分成了一个包含所有图片的文件夹。我希望标注起始帧和结束帧，获得视频的一段clip标记。在这一段视频里我可能会标注出多个不同的clip。我希望界面上有：图片canvas；display一个进度条，进度条的长度为总共图片数量，进度条上的滑块位置在当前的帧数，进度条上可视化已经标注过的clips；需要实现的button：前一帧，前10帧，后一帧，后10帧，标记start；标记end，标记完成导出。

```python

```

2. 标注相机参数

> prompt: 请帮我进行javascript+python网络设计，假设你是一名前端编程专家，请帮我实现一个前端页面。这个页面用于标注多张图片里面的匹配点。页面首先显示一个canvas，这个canvas可以进行对当前图片的点的标注。接着下面是许多小的图片，当点击每张图片时，canvas上聚焦的是选中的那张图片，接着可以对那张图片进行标注。所有视角的标注点共享一样的序号，他们表示同一个点在不同的视角里的位置，他们需要根据序号按照同一个颜色显示出来。


```python

```

### 3. 标注矩形区域

TODO

### 4. 标注vanishing lines

0. File Structure

```bash
root
- images
    - video0
        - 000000.jpg
        - 000001.jpg
        ...
    - video1
    - video2
    ...
```

1. Start

```bash
data=/path/to/image_data
python3 app.py --root ${data} --port 2345
```

Open the browser:

```
http://0.0.0.0:2345/vanish
```

Select one folder and click in.

2. Annotate

通过选择轴功能选择当前标注的X/Y/Z轴，在图像上点击并拖动即可画线。
一般定义与地面平行的两个垂直的方向为X/Y轴，垂直于地面指向上的方向为Z轴。

绘制完成后点击提交即可保存标注结果。请确保在数据目录下有写入的权限。数据会保存到`${data}/vanish_points.yml`中。

