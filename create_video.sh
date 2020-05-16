cd intermediate
ffmpeg -y -i '%4d.png' -r 30  -c:v ffv1 -qscale:v 0 output.avi
mv output.avi ../
