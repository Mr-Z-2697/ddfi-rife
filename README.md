# ddfi
A clumsy video auto duplication remove and frame interpolate script

The basic idea:
1. Remove duplicated frames (resulting a vfr video that has minimal fps 8)
2. Interpolate 8x (to ensure minimal fps parts will be above 60fps after interpolation)
3. Extract timestamps from de-dupped video, then calculate the timestamps of interpolated frames
4. Mux interpolated video with caculated timestamps
5. Convert to 60fps(60000/1001) cfr video

*(yeah, its basically just simply excuting commands automatically except new-timestamps calculating part)*
