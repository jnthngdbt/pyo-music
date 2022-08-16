import audiosegment
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import scipy.io.wavfile
import winsound
from threading import Thread

matplotlib.style.use(['dark_background'])
plt.rcParams['grid.color'] = '#333333'
plt.rcParams['figure.facecolor'] = '#222222'
plt.rcParams['axes.facecolor'] = '#222222'
plt.rcParams["font.size"] = "11"
plt.rcParams["font.family"] = "consolas"

plt.close('all')

## -------------------------------------------------------
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

## -------------------------------------------------------
def interpolateFft(F, k):
  Nf = F.shape[0]
  Ng = int(Nf * k)

  Gr = np.zeros((Ng, 2))
  Gi = np.zeros((Ng, 2))
  G = np.zeros((Ng, 2))

  f = np.fft.fftfreq(Nf)
  f = np.fft.fftshift(f)
  F = np.fft.fftshift(F, axes=0)

  g = np.linspace(f[0], f[-1], num=Ng)

  for i in np.arange(F.shape[1]):
    Gr[:,i] = np.interp(g, f, F[:,i].real)
    Gi[:,i] = np.interp(g, f, F[:,i].imag)

  G = Gr + 1j * Gi
  G = np.fft.fftshift(G)

  return G

## -------------------------------------------------------

def exportCompressed(x, name, fs):
  y = audiosegment.from_numpy_array(discretize(x), fs)
  y.export(name)

def playSound(x):
  tmpWav = "temp.wav"
  loop(x, tmpWav)

## -------------------------------------------------------
seg = audiosegment.from_file("./data/PadSynth.Raw.mp3")

fs = seg.frame_rate
x = seg.to_numpy_array()
x = x / x.max()

if x.ndim == 1:
  x = x.reshape(len(x), 1)
  x = np.concatenate([x, x], axis=1)

exportCompressed(x, "./songs/ir.convolve.input.mp3", fs)

## -------------------------------------------------------
# seg = audiosegment.from_file("./data/IR.BottleHall.wav")
# seg = audiosegment.from_file("./data/IR.Large Long Echo Hall.wav")
# seg = audiosegment.from_file("./data/IR.St Nicolaes Church.wav")
seg = audiosegment.from_file("./data/IR.On a Star.wav")

ir = seg.to_numpy_array()
ir = ir / ir.max()

if ir.ndim == 1:
  ir = ir.reshape(len(ir), 1)
  ir = np.concatenate([ir, ir], axis=1)

## -------------------------------------------------------

xir0 = np.convolve(x[:,0], ir[:,0])
xir1 = np.convolve(x[:,1], ir[:,1])
xir = np.concatenate([xir0.reshape(len(xir0), 1), xir1.reshape(len(xir1), 1)], axis=1)

exportCompressed(xir, "./songs/ir.convolve.result.mp3", fs)

## -------------------------------------------------------
Tk = 6 # desired final sample duratio

F = np.fft.fft(xir, axis=0)
F = interpolateFft(F, 1)
F = np.abs(F) * np.exp(1j * np.random.rand(F.shape[0], F.shape[1]) * 2.0 * np.pi) # randomize phases

xr = np.real(np.fft.ifft(F, axis=0))

exportCompressed(xr, "./songs/ir.convolve.reconstruct.mp3", fs)

music_thread = Thread(target=lambda: playSound(xr))
music_thread.start()

plt.show()
