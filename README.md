[中文说明](https://github.com/Mr-Z-2697/ddfi/blob/main/README.chi.md)☚
# ddfi
#### dedup frame interpolate
A clumsy video auto duplication remove and frame interpolate script (mainly for 24fps cfr animation with dup-frames (by dup-frames I mean the twos, threes and etc. things)

## The basic idea:
1. Remove duplicated frames (resulting a vfr video that technically has minimal fps 8)
2. Interpolate 8x (to ensure minimal fps parts will be above 60fps after interpolation)
3. Extract timestamps from de-dupped video, then calculate the timestamps of interpolated frames
4. Mux interpolated video with caculated timestamps
5. Convert to 60fps(60000/1001) cfr video

*(yeah, its basically just simply excuting commands automatically except new-timestamps calculating part)*

## Usage:
run `ddfi.py -h` for detail

## Example:
a: this script | b: use svp directly

![](https://github.com/Mr-Z-2697/ddfi/blob/main/example/ddfi.webp?raw=true)
![](https://github.com/Mr-Z-2697/ddfi/blob/main/example/simp.webp?raw=true)

## Downsides:
more visible artifacts
![](https://github.com/Mr-Z-2697/ddfi/blob/main/example/artifacts.webp?raw=true)
