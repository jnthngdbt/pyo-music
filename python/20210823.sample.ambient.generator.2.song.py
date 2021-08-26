# 20210823.sample.ambient.generator.py

import audiosegment
import numpy as np
import matplotlib.pyplot as plt
import scipy.io.wavfile
import winsound
import os
from scipy import signal

## -------------------------------------------------------
def discretize(x, volume=1.0):
  A = volume * (2**15-1) # int16 scale factor; 2^16/2, since signed, -1 to avoid saturation
  x *= A/np.max(np.abs(x)) # normalize the signal to span the int16 domain
  return x.astype(np.int16)

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
  Ng = Nf * k

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

def randomizePhase(F):
  F = np.abs(F) * np.exp(1j * np.random.rand(F.shape[0], F.shape[1]) * np.pi)
  return F

def applyWindow(x, w):
  Nw = int(0.5 * len(w))
  Nx = x.shape[0]

  for i in np.arange(x.shape[1]):
    x[0:Nw, i] = x[0:Nw, i] * w[0:Nw]
    x[Nx-Nw:, i] = x[Nx-Nw:, i] * w[Nw:]

  return x

## -------------------------------------------------------
def processSample(x, k):
  F = np.fft.fft(x, axis=0)
  F = interpolateFft(F, k)
  F = randomizePhase(F)

  x = np.real(np.fft.ifft(F, axis=0))
  x = applyWindow(x, signal.windows.hann(int(x.shape[0])))

  return x

def addSignal(s, x):
  if len(s) <= 0:
    return x
  else:
    return np.concatenate((s, x), axis=0) 

def mixSignal(s, x, n):
  if len(s) <= 0:
    return x
  else:
    Ns = s.shape[0] 
    Nx = x.shape[0]
    Ny = Ns + Nx - n
    y = np.zeros((Ny, s.shape[1]))
    y[0:Ns, :] = s
    y[Ns-n:,:] += x
    return y

## -------------------------------------------------------
# seg = audiosegment.from_file("./data/03 Mission Two.m4a")
seg = audiosegment.from_file("./data/04 Mission Three.m4a")
# seg = audiosegment.from_file("./data/07 Mission Six.m4a")
# seg = audiosegment.from_file("./data/07 Mission Six.m4a")
# seg = audiosegment.from_file("./data/07 Mission Six.m4a")
# seg = audiosegment.from_file("./data/11 Mission Ten.m4a")
# seg = audiosegment.from_file("./data/Big Rock.1.m4a")
# seg = audiosegment.from_file("./data/Alone.3.m4a")
# seg = audiosegment.from_file("./data/Jump.12.m4a")
# seg = audiosegment.from_file("./data/Press.5.m4a")
# seg = audiosegment.from_file("./data/Late.06.m4a")
# seg = audiosegment.from_file("./data/Sam Sung 3.m4a") # wow
# seg = audiosegment.from_file("./data/Aly Wood 2.m4a")
# seg = audiosegment.from_file("./data/Beverly Aly Hills 5.m4a")
# seg = audiosegment.from_file("./data/insects.m4a")
# seg = audiosegment.from_file("./data/smallthings.m4a")

fs = seg.frame_rate
x = seg.to_numpy_array()
x = x / x.max()

if x.ndim == 1:
  x = x.reshape(len(x), 1)
  x = np.concatenate([x, x], axis=1)

## -------------------------------------------------------
Ts = 4. # step duration
Tw = 0.25 # sample duration
Tk = 8 # desired final sample duration, should be longer than sample duration

Ns = int(Ts * fs)
Nw = int(Tw * fs)
Nk = int(Tk * fs)
k = int(Nk / Nw)

nbSteps = x.shape[0] - Nw - 1

s = []
for i in np.arange(nbSteps, step=Ns):
  print('{0}/{1}'.format(i, nbSteps))
  s = mixSignal(s, processSample(x[i:i+Nw, :], k), int(0.1 * Nk))

sos = signal.butter(2, 3000, 'lp', fs=fs, output='sos')
s = signal.sosfilt(sos, s, axis=0)

y = audiosegment.from_numpy_array(discretize(s), fs)
y.export("./songs/sample.ambient.generated.song.m4a")

tmpWav = "temp.wav"
loop(s, tmpWav)

