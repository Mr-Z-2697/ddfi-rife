# ddfi
#### dedup frame interpolate
一个视频自动去重插帧的沙雕脚本 (主要用于有重复画面的恒定24fps动画 (重复画面指一拍二之类的技术带来的结果)

## 基本思路:
1. 去除重复帧 (得到一个理论上最低8fps的vfr视频)
2. 插帧到8倍 (保证最低的8fps也能插到60fps以上)
3. 从去重的视频提取timestamps，计算插入的帧的timestamps
4. 使用计算出的timestamps“校正”视频流
5. 转换到60fps输出

*(对，基本上除了计算新timestamps的部分以外就只是自动执行命令行)*

## 用法:
运行 `ddfi2.py -h` 查看详细

## 例子:
左：此脚本 (rife版) | 右：直接rife (两次，配合mvtools以达到24->96->60)

https://user-images.githubusercontent.com/74594146/142829178-ff08b96f-9ca7-45ab-82f0-4e95be045f2d.mp4

## 缺点:
更多可见瑕疵

rife版：
![IMG](https://user-images.githubusercontent.com/74594146/142829294-1b17c073-f587-4e49-8a72-c3c8b4149a53.png)
svp版：
![IMG](https://github.com/Mr-Z-2697/ddfi/blob/main/example/artifacts.webp?raw=true)
