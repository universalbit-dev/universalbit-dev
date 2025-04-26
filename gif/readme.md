# Create Your GIF from MP4 Video

This guide provides instructions for creating and optimizing GIFs from MP4 videos. Follow the steps below to get started.

---

## Support UniversalBit Project
- [Support the UniversalBit Project](https://github.com/universalbit-dev/universalbit-dev/tree/main/support)  
- [Disambiguation](https://en.wikipedia.org/wiki/Wikipedia:Disambiguation)  
- [Bash References](https://www.gnu.org/software/bash/manual/)

---

## Steps to Create a GIF

### 1. Record MP4 Videos
We recommend using [SimpleScreenRecorder](https://en.wikipedia.org/wiki/SimpleScreenRecorder) to record MP4 videos. Install it using the following command:

```bash
sudo apt install simplescreenrecorder
```

---

### 2. Convert MP4 to GIF
Use [FFMPEG](https://ffmpeg.org/) to convert MP4 videos to GIFs. Install FFMPEG with the command below:

```bash
sudo apt install ffmpeg
```

To convert an MP4 video to a GIF, use this command:

```bash
ffmpeg -i input_video_file output.gif
```

---

### 3. Optimize GIFs
#### Reduce GIF Size
To reduce the size of a GIF (scaled to 1080p with a frame rate of 5 FPS), use the following command:

```bash
ffmpeg -y -i input-video.mp4 -filter_complex "fps=5,scale=1080:-1:flags=lanczos,split[s0][s1];[s0]palettegen=max_colors=32[p];[s1][p]paletteuse=dither=bayer" out-file-name.gif
```

---

### Additional Notes
- Ensure you install the required tools before starting.
- Experiment with different scaling and FPS values to achieve the desired GIF quality.

---
