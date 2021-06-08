# ddfi
#### dedup frame interpolate
一个视频自动去重插帧的沙雕脚本 (主要用于有重复画面的恒定24fps动画 (重复画面指一拍二之类的技术带来的结果)

## 基本思路:
1. 去除重复帧 (得到一个理论上最低8fps的vfr视频)
2. 插帧到8倍 (保证最低的8fps也能插到60fps以上)
3. 从去重的视频提取timestamps，计算插入的帧的timestamps
4. 在混流时加入timestamps
5. 转换到60fps(60000/1001) 输出

*(对，基本上除了计算新timestamps的部分以外就只是自动执行命令行)*

## 用法:
运行 `ddfi.py -h` 查看详细

## 例子:
上: 此脚本 | 下: 直接svp

![](https://github.com/Mr-Z-2697/ddfi/blob/main/example/ddfi.webp?raw=true)
![](https://github.com/Mr-Z-2697/ddfi/blob/main/example/simp.webp?raw=true)

## 缺点:
更多可见瑕疵
![](https://github.com/Mr-Z-2697/ddfi/blob/main/example/artifacts.webp?raw=true)
