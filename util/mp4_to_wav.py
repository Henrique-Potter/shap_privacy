import glob
import subprocess
from pathlib import Path

import numpy as np
from tqdm import tqdm

video_files_path = "c:\\ravdess\\"
audio_files_path = "c:\\ravdess\\converted\\"

v_files = glob.glob("{}/**/*.mp4".format(video_files_path), recursive=True)

lst = []
for full_fname in tqdm(v_files):
    lst.append(full_fname)

lst_np = np.array(lst)
uniques = np.unique(lst_np)

for full_fname in tqdm(v_files):
    file_name = Path(full_fname).name
    f_name_only = file_name[:-4]
    # Using ffmpeg to convert the mp4 in wav
    # Example command: "ffmpeg -i C:/test.mp4 -ab 160k -ac 2 -ar 44100 -vn audio.wav"
    command = "ffmpeg -i {} -ab 160k -ac 2 -ar 44100 -vn {}".format(full_fname, audio_files_path + f_name_only + '.wav')

    try:
        subprocess.call(command, shell=True)

    # Skip the file in case of error
    except ValueError:
        print(ValueError)
        continue
