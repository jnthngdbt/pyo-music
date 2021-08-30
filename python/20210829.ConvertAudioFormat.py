# 20210829.ConvertAudioFormat.py

import os
import audiosegment

dir = "C:\\Users\\jgodbout\\Documents\\git\\sandbox\\python\\songs\\"
for file in os.listdir(dir):
  if file.endswith(".m4a"):
    seg = audiosegment.from_file(dir + file)
    seg.export(dir + file + ".mp3")