from moviepy.video.io import ImageSequenceClip
import numpy as np
import os

def save_frames_as_gif(
    frames,
    out_path,
    filename     = 'test.gif',
    fps          = 15,
    frame_reduce = 2):
    """
        Save numpy frames as a gif.

        Arguments:
            frames        A list of rendered frames as rgb arrays.
            out_path      The path to save the gif to.
            filename      The name of the gif.
            fps           Frames per second.
            frame_reduce  Reduce frames 2^frame_reduce. This can help with speed
                          when displaying gifs in a browser.
    """
    #
    # Cut the frames in half for for every frame_reduce.
    #
    for i in range(frame_reduce):
        frames = np.array(frames)[1::2]

    clip = ImageSequenceClip(list(frames), fps=fps)
    clip.write_gif(os.path.join(out_path, filename), fps=fps)
