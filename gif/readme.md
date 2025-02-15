##### [Support UniversalBit Project](https://github.com/universalbit-dev/universalbit-dev/tree/main/support) -- [Disambiguation](https://en.wikipedia.org/wiki/Wikipedia:Disambiguation) -- [Bash Reference Manual](https://www.gnu.org/software/bash/manual/html_node/index.html) -- [Join Mastodon](https://mastodon.social/invite/wTHp2hSD) -- [Website](https://www.universalbit.it/) -- [Content Delivery Network](https://universalbitcdn.it/)

#### Create your Gif From Mp4 Video
copilot explain

The `gif/readme.md` file provides instructions for creating and optimizing GIFs from MP4 videos. Here are the key points:

1. **Support and References**:
   - Links to support the UniversalBit project, disambiguation, and Bash references.

2. **Creating GIFs from MP4 Videos**:
   - **Recording MP4 Videos**: Suggests using [SimpleScreenRecorder](https://en.wikipedia.org/wiki/SimpleScreenRecorder) and provides installation instructions:
     ```bash
     sudo apt install simplescreenrecorder
     ```
   - **Converting MP4 to GIF**: Uses [FFMPEG](https://ffmpeg.org/) for conversion and provides installation and usage instructions:
     ```bash
     sudo apt install ffmpeg
     ffmpeg -i input_video_file output.gif
     ```

3. **GIF Optimization**:
   - **Reducing GIF Size**: Provides a command to reduce the size of GIFs (scaled to 1080):
     ```bash
     ffmpeg -y -i input-video.mp4 -filter_complex "fps=5,scale=1080:-1:flags=lanczos,split[s0][s1];[s0]palettegen=max_colors=32[p];[s1][p]paletteuse=dither=bayer" out-file-name.gif
     ```

4. **Visual Example**:
   - Includes an example GIF created from an MP4 video.
---

