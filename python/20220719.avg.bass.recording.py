import audiosegment
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import winsound
from threading import Thread
import scipy.io.wavfile

#

matplotlib.style.use(['dark_background'])
plt.rcParams['grid.color'] = '#333333'
plt.rcParams['figure.facecolor'] = '#222222'
plt.rcParams['axes.facecolor'] = '#222222'
plt.rcParams["font.size"] = "11"
plt.rcParams["font.family"] = "consolas"

plt.close('all')

#

# seg = audiosegment.from_file("./data/Clean Combo#03.wav") # E1 2 bps
seg = audiosegment.from_file("./data/Clean Combo#04.wav") # E1 2 bps darker
# seg = audiosegment.from_file("./data/Clean Combo#14.wav") # E1 2 bps smooth
# seg = audiosegment.from_file("./data/Clean Combo#16.wav") # E1 0.? bps smooth
# seg = audiosegment.from_file("./data/Clean Combo#07.wav") # E3 4 bps eye of the tiger
# seg = audiosegment.from_file("./data/Clean Combo#08.wav")
# seg = audiosegment.from_file("./data/Clean Combo#08.wav")
# seg = audiosegment.from_file("./data/Clean Combo#08.wav")
# seg = audiosegment.from_file("./data/Clean Combo#08.wav")
# seg = audiosegment.from_file("./data/Clean Combo#08.wav")
# seg = audiosegment.from_file("./data/Clean Combo#08.wav")
# seg = audiosegment.from_file("./data/Clean Combo#08.wav")

bpm = 120
ticksPerSegment = 0.5

#

fs = seg.frame_rate
x = seg.to_numpy_array()
x = x / x.max()

if x.ndim == 1:
  x = x.reshape(len(x), 1)
  x = np.concatenate([x, x], axis=1)

#

def discretize(x, volume=1.0):
  A = volume * (2**15-1) # int16 scale factor; 2^16/2, since signed, -1 to avoid saturation
  y = x.copy()
  y *= A/np.max(np.abs(x)) # normalize the signal to span the int16 domain
  return y.astype(np.int16)

def write(x, name, volume=1.0):
  scipy.io.wavfile.write(name, fs, discretize(x, volume))

def loop(s, f):
  play(s, f, loop=True)

def play(s, f, volume=1.0, loop=False):
  f = "./songs/" + f
  write(s, f, volume=volume)
  play = True
  while (play):
    winsound.PlaySound(f, 0)
    play = loop

def playSound(x):
  tmpWav = "temp.wav"
  loop(x, tmpWav)

#

bps = bpm / 60.
signalLen = x.shape[0]
numChans = x.shape[1]

tickIntervalSec = 1. / bps
tickInterval = int(tickIntervalSec * fs)

segmentLen = int(tickInterval * ticksPerSegment)
segmentPos = np.arange(0, signalLen, segmentLen)

#

avg = np.zeros((segmentLen, numChans))
numSegments = np.floor(signalLen / segmentLen)

numAvg = 0

for i in np.arange(1, numSegments): # skip first
  i1 = int(i * segmentLen)
  i2 = int((i + 1) * segmentLen)
  xi = x[i1:i2, :]

  avg += xi
  numAvg += 1

  plt.plot(xi, alpha=.2)

avg *= 1. / numAvg 

plt.plot(10. * avg, 'r', linewidth=2)

#

# Just to test looping
y = np.tile(avg, (50, 1))

#

music_thread = Thread(target=lambda: playSound(y))
music_thread.start()

# music_thread = Thread(target=lambda: playSound(x[5 * tickInterval:6 * tickInterval, :]))
# music_thread.start()

#
plt.figure()

plt.plot(x)
plt.plot(segmentPos, np.zeros(len(segmentPos)), '.')

plt.show()