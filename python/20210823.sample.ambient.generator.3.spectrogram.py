# 20210823.sample.ambient.generator.py

import audiosegment
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from numpy.core.fromnumeric import argmax
import scipy.io.wavfile
import winsound
import os
from scipy import signal
from scipy import ndimage
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

def randomizePhase(F):
  F = np.abs(F) * np.exp(1j * np.random.rand(F.shape[0], F.shape[1]) * 2.0 * np.pi)
  return F

def applyWindow(x, w):
  Nw = int(0.5 * len(w))
  Nx = x.shape[0]

  for i in np.arange(x.shape[1]):
    x[0:Nw, i] = x[0:Nw, i] * w[0:Nw]
    x[Nx-Nw:, i] = x[Nx-Nw:, i] * w[Nw:]

  return x

## -------------------------------------------------------
def spectrogram(x, fs, Ts, Tw):
  stepLen = int(Ts * fs)
  windowLen = int(Tw * fs)

  maxPos = x.shape[0] - windowLen - 1
  nbSteps = int(maxPos / stepLen) - 1
  nbChans = x.shape[1]

  S = np.zeros((windowLen, nbSteps, nbChans)) * np.exp(1j * np.zeros((windowLen, nbSteps, nbChans)))
  
  for i in np.arange(nbSteps):
    print('{0}/{1}'.format(i, nbSteps))
    pos = i * stepLen
    xi = x[pos: pos + windowLen, :]
    S[:, i, :] = np.fft.fft(xi, axis=0)

  return S

def mergeSpectra(S, c):
  R = np.zeros(S.shape)

  iStart = 0
  N = 1

  Sk = np.reshape(S[:,0,:], (S.shape[0], 1, S.shape[2]))

  for i in np.arange(1, S.shape[1]):
    Si = np.reshape(S[:,i,:], (S.shape[0], 1, S.shape[2]))

    Pi = np.log(np.squeeze(np.sum(Si, axis=2)))
    Pk = np.log(np.squeeze(np.sum(Sk, axis=2)))
    C = np.corrcoef(Pi, Pk)

    if np.all(C >= c):
      Sk = Sk + (Si - Sk) / (N + 1)
      N += 1
    else:
      if iStart == i-1: R[:, iStart, :] = np.squeeze(Sk)
      else: R[:, iStart:i-1, :] = Sk
     
      iStart = i
      Sk = Si

  return R

def smoothSpectrogram(S, n):
  return ndimage.convolve1d(S, np.ones(n), axis=1)

def reconstructSample(F, k, w):
  F = interpolateFft(F, k)
  F = randomizePhase(F)

  x = np.real(np.fft.ifft(F, axis=0))
  x = applyWindow(x, signal.windows.hann(int(w * x.shape[0])))

  return x

def filterSound(x, fc, fs):
  sos = signal.butter(6, fc, 'lp', fs=fs, output='sos')
  x = signal.sosfilt(sos, x, axis=0)
  return x

def boostBass(x, k, fc, fs):
  sos = signal.butter(2, fc, 'lp', fs=fs, output='sos')
  return x + k * signal.sosfilt(sos, x, axis=0)

def exportCompressed(x, name, fs):
  y = audiosegment.from_numpy_array(discretize(x), fs)
  y.export(name)

def playSound(x):
  tmpWav = "temp.wav"
  loop(x, tmpWav)

## -------------------------------------------------------
# seg = audiosegment.from_file("./data/03 Mission Two.m4a") # 72*0.05, 88*0.05
# seg = audiosegment.from_file("./data/04 Mission Three.m4a") # 24*0.05, 38*0.05, 234*0.05
# seg = audiosegment.from_file("./data/07 Mission Six.m4a") # 331*0.05, 545*0.05, 1760*0.05
seg = audiosegment.from_file("./data/11 Mission Ten.m4a") # 494*0.05, 727*0.05
# seg = audiosegment.from_file("./data/Big Rock.1.m4a")
# seg = audiosegment.from_file("./data/Alone.3.m4a")
# seg = audiosegment.from_file("./data/Jump.12.m4a")
# seg = audiosegment.from_file("./data/Press.5.m4a")
# seg = audiosegment.from_file("./data/Late.06.m4a")
# seg = audiosegment.from_file("./data/Sam Sung 3.m4a") # wow
# seg = audiosegment.from_file("./data/Aly Wood 2.m4a")
# seg = audiosegment.from_file("./data/Beverly Aly Hills 5.m4a") # t: 3.55, 7.5, 12.6
# seg = audiosegment.from_file("./data/insects.m4a")
# seg = audiosegment.from_file("./data/smallthings.m4a") # t: 2.85, 16.5, 31.55, 47.95
# seg = audiosegment.from_file("./data/Background noise with voice.m4a") # 43*0.05

fs = seg.frame_rate
x = seg.to_numpy_array()
x = x / x.max()

if x.ndim == 1:
  x = x.reshape(len(x), 1)
  x = np.concatenate([x, x], axis=1)

## -------------------------------------------------------
Ts = 0.05 # step duration
Tw = 0.5 # sample duration
lowPass = 8000
doBoostBass = False # when using a recording

x = filterSound(x, lowPass, fs)
if doBoostBass:
  x = boostBass(x, 5, 500, fs)

exportCompressed(x, "./songs/sample.ambient.generated.song.m4a", fs)

S = spectrogram(x, fs, Ts, Tw)
# S = mergeSpectra(S, 0.98) # meh, must have more tools to visualize and debug
# S = smoothSpectrogram(S, 3)

t = np.linspace(0, x.shape[0] / fs, num=S.shape[1])
f = np.fft.fftfreq(S.shape[0], 1 / fs)

## -------------------------------------------------------
Tk = 20 # desired final sample duration
timePosSec = 727*0.05
fadeRatio = 0.1

ti = argmax(t > timePosSec) # sample index to play

Si = np.squeeze(S[:,ti,:])
s = reconstructSample(Si, Tk / Tw, fadeRatio)

exportCompressed(s, "./songs/sample.ambient.generated.sample.mp3", fs)

music_thread = Thread(target=lambda: playSound(s))
music_thread.start()

## ------------------------------------------------------

P = np.log(np.abs(S))
fp = np.squeeze(P[:,ti,:])

Ntrend = fp.shape[0] / 50.
ftrend = ndimage.convolve1d(fp, np.ones(int(Ntrend)), axis=0) / Ntrend
fdetrend = fp - ftrend
Nsmooth = 1 # 1 for no effect
fsmooth = ndimage.convolve1d(fdetrend, np.ones(int(Nsmooth)), axis=0) / Nsmooth
plt.figure()
plt.plot(f, fp)
plt.plot(f, ftrend)
plt.plot(f, fsmooth)
plt.figure()
plt.plot(f,fsmooth)

# Nfp = fp.shape[0]
# fshift = np.fft.fftshift(fp)
# fharm = np.zeros(fshift.shape)
# for i in np.arange(3, fshift.shape[0]):
#   comb = np.arange(i, Nfp, i)
#   fharm[i] = np.mean(fshift[comb, :], axis=0)
# fharm = np.fft.fftshift(fharm)
# plt.figure()
# plt.plot(f, fp)
# plt.plot(f, fharm)
# plt.figure()
# plt.plot(f, fharm)

# ffp = np.fft.fft(fp, axis=0)
# plt.figure()
# plt.plot(f, ffp)

## ------------------------------------------------------

maxFreq = 10000
fmax = argmax(f > maxFreq)
P0 = P[:fmax, :, 0]
f0 = f[:fmax]
plt.figure()
plt.pcolormesh(t, f0, P0)



C = np.corrcoef(P0.T)

plt.figure()
plt.pcolormesh(t, t, C)

plt.show()
