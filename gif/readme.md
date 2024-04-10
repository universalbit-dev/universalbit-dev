#### Create your Gif From Mp4 Video

Recording .mp4 Videos with: 
### [SimpleScreenRecorder](https://en.wikipedia.org/wiki/SimpleScreenRecorder)
```bash
sudo apt install simplescreenrecorder
```

### [FFMPEG](https://ffmpeg.org/)

```bash
sudo apt install ffmpeg
ffmpeg -i input_video_file output.gif
```

<img src="https://github.com/universalbit-dev/universalbit-dev/blob/main/gif/create-gif-from-mp4.gif" width="auto"></img>


### Reduce Gif Size: (scale=1080)
```bash
ffmpeg -y -i input-video.mp4 -filter_complex "fps=5,scale=1080:-1:flags=lanczos,split[s0][s1];[s0]palettegen=max_colors=32[p];[s1][p]paletteuse=dither=bayer" out-file-name.gif
```
