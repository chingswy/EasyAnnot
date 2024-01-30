function seedRandom(seed) {
    return function() {
        seed = (seed * 9301 + 49297) % 233280;
        return seed / 233280;
    };
}

function generateRandomColors(numColors, seed) {
    let colors = [];
    let random = seedRandom(seed);

    for (let i = 0; i < numColors; i++) {
        let color = "#" + ("000000" + Math.floor(random() * 16777215).toString(16)).slice(-6);
        colors.push(color);
    }

    return colors;
}

const randomColors = [
    "#FF5733", // 亮红色
    "#33FF57", // 亮绿色
    "#3357FF", // 亮蓝色
    "#F1C40F", // 金黄色
    "#8E44AD", // 紫色
    "#FFC300", // 柠檬色
    "#C70039", // 暗红色
    "#A3E4D7", // 淡青色
    "#34495E", // 暗蓝色
    "#7D3C98", // 深紫色
    "#F39C12", // 橙色
    "#16A085", // 青色
    "#B03A2E", // 褐红色
    "#2980B9", // 中蓝色
    "#D35400", // 西瓜红
];
