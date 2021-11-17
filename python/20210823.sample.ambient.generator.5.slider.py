# 20210823.sample.ambient.generator.py

import audiosegment
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from numpy.core.fromnumeric import argmax, size
import scipy.io.wavfile
import tkinter as tk
from PIL import Image, ImageOps, ImageTk
from scipy import signal
import winsound
from threading import Thread

matplotlib.style.use(['dark_background'])
plt.rcParams['grid.color'] = '#333333'
plt.rcParams['figure.facecolor'] = '#222222'
plt.rcParams['axes.facecolor'] = '#222222'
plt.rcParams["font.size"] = "11"
plt.rcParams["font.family"] = "consolas"

plt.close('all')

def discretize(x, volume=1.0):
  A = volume * (2**15-1) # int16 scale factor; 2^16/2, since signed, -1 to avoid saturation
  y = x.copy()
  y *= A/np.max(np.abs(x)) # normalize the signal to span the int16 domain
  return y.astype(np.int16)

def write(x, name, volume=1.0):
  scipy.io.wavfile.write(name, fs, discretize(x, volume))

def play(s, f, volume=1.0):
  f = "./songs/" + f
  write(s, f, volume=volume)
  winsound.PlaySound(f, winsound.SND_ASYNC | winsound.SND_LOOP | winsound.SND_FILENAME) # cannot play async from memory

def interpolateFft(F, k):
  Nf = F.shape[0]
  Ng = int(Nf * k)

  F = np.abs(F)
  G = np.zeros((Ng, 2))

  f = np.fft.fftfreq(Nf)
  f = np.fft.fftshift(f)
  F = np.fft.fftshift(F, axes=0)

  g = np.linspace(f[0], f[-1], num=Ng)

  for i in np.arange(F.shape[1]):
    G[:,i] = np.interp(g, f, F[:,i])

  G = np.fft.fftshift(G)

  return G

def applyWindow(x, w):
  Nw = len(w)
  Nh = int(0.5 * Nw)
  Nx = x.shape[0]

  for i in np.arange(x.shape[1]):
    x[0:Nh, i] = x[0:Nh, i] * w[0:Nh]
    x[Nx-Nh:, i] = x[Nx-Nh:, i] * w[Nw-Nh:]

  return x

def crossFadeWindowCosine(n): # for equal power fading
  t = np.arange(n)
  x = np.sin(np.pi * t / n)
  return x

def reconstructSample(F, k, fadeLen, phases):
  F = interpolateFft(F, k)

  # Use random phases when cross-fading to avoid interferences and hear beats due to similar signals.
  if len(phases) == 0: 
    phases = np.random.rand(F.shape[0], F.shape[1]) * 2.0 * np.pi

  F = np.abs(F) * np.exp(1j * phases)

  x = np.real(np.fft.ifft(F, axis=0))

  if fadeLen > 0:
    x = applyWindow(x, crossFadeWindowCosine(fadeLen))

  return x, phases

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

  return np.abs(S)

## -------------------------------------------------------
# name = "03 Mission Two"
# name = "04 Mission Three"
# name = "05 Mission Four"
# name = "07 Mission Six"
# name = "09 Mission Eight"
name = "11 Mission Ten"
# name = "Big Rock.1"
# name = "Alone.3"
# name = "Jump.12"
# name = "Press.5"
# name = "Late.06"
# name = "Sam Sung 3" # wow
# name = "Aly Wood 2"
# name = "Beverly Aly Hills 5"
# name = "insects"
# name = "smallthings" # t: 31
# name = "Sam Buca - Outdoor Tone Car"
# name = "Sam Buca - Outdoor Tone"
# name = "Tone night 21h Aug"
# name = "Tone outside 17h August"
# name = "Sam Buca - Outdoor Water"
# name = "Sam Buca - Indoor Water Mild"
# name = "Sam Buca - Indoor Water"
# name = "Tron Ouverture"
# name = "Background noise with voice"
# name = "TC RS212 vs SVT15E"
# name = "bass recording jam miche 1"

nameIn = "./data/" + name + ".m4a"

seg = audiosegment.from_file(nameIn)

## -------------------------------------------------------
Ts = 0.05 # step duration
Tw = 0.25 # sample duration
lowPass = 20000
doBoostBass = False # when using a recording

Tk = 4.0 # desired final sample duration (slow down factor)
fadeDur = 0.0

sliderWidth = 1200
dpi = 100
figsizeSpectro = (1.05 * sliderWidth / dpi, 0.5*sliderWidth / dpi)
figsizeFft = (1.05 * sliderWidth / dpi, 0.2*sliderWidth / dpi)
maxFreqPlots = 4000

sliderMinFreq = 0
sliderMaxFreq = maxFreqPlots

## -------------------------------------------------------

fs = seg.frame_rate
x = seg.to_numpy_array()
x = x / x.max()

if x.ndim == 1:
  x = x.reshape(len(x), 1)
  x = np.concatenate([x, x], axis=1)

x = filterSound(x, lowPass, fs)
if doBoostBass:
  x = boostBass(x, 5, 500, fs)

exportCompressed(x, "./songs/sample.ambient.generated.song.m4a", fs)

S = spectrogram(x, fs, Ts, Tw)
f = np.fft.fftfreq(S.shape[0], 1 / fs)
t = np.linspace(0, x.shape[0] / fs, num=S.shape[1])

## -------------------------------------------------------

windowLen = int(Tw * fs)
stepLen = int(Ts * fs)
nbSteps = S.shape[1]

r = tk.Tk()
sliderPosValue = tk.IntVar()
sliderLowFreqValue = tk.IntVar()
sliderHighFreqValue = tk.IntVar()
sliderVolumeValue = tk.IntVar()

fftfig, fftax = plt.subplots(figsize=figsizeFft, dpi=dpi)
plt.tight_layout()

fft2fig, fft2ax = plt.subplots(figsize=figsizeFft, dpi=dpi)
plt.tight_layout()

def playSample(p):
  xi = x[p: p + windowLen, :]
  Fi = np.fft.fft(xi, axis=0)
  Fi = applyFilter(f, Fi, sliderLowFreq.get(), sliderHighFreq.get())
  si, p = reconstructSample(np.abs(Fi), Tk / Tw, int(fadeDur * 2 * fs), [])

  Fo = np.fft.fft(si, axis=0) # output signal FFT
  showSpectrum(np.abs(Fo[:,0]), fft2ax)

  tmpWav = "temp.wav"
  play(si, tmpWav, volume=sliderVolume.get()/100.0)

def showSpectrum(F, ax):
  f = np.fft.fftfreq(F.shape[0], 1 / fs)
  ax.clear()
  ax.plot(f, F)
  ax.set_xlim([0, maxFreqPlots])

  Fw = getFilterWindow(f, sliderLowFreq.get(), sliderHighFreq.get())
  ax.plot(f, Fw * 0.5 * np.max(F))

def duplicate1dArrayToChannels(x, nbChannels):
  return np.tile(x, (nbChannels, 1)).T # copy as 2 equal columns

def getFilterWindow(f, minFreq, maxFreq):
  ti = argmax(f > minFreq)
  tj = argmax(f > maxFreq)
  Nw = tj - ti

  w = signal.windows.boxcar(Nw)

  Fw = np.zeros(len(f))
  Fw[ti:tj] = w
  Fw[-tj:-ti] = w
  return Fw

def applyFilter(f, F, minFreq, maxFreq):
  if maxFreq < sliderMaxFreq:
  w = getFilterWindow(f, minFreq, maxFreq)
  w = duplicate1dArrayToChannels(w, F.shape[1])
    F = F * w
  return F

def update():
  playSample(sliderPos.get() * stepLen)
  showSpectrum(S[:,sliderPos.get(),0], fftax)

def sliderPosChanged(event):
  print('{}s'.format(sliderPos.get() * Ts))
  update()

sliderPos = tk.Scale(
    r,
    length=sliderWidth,
    from_=0,
    to=nbSteps,
    orient='horizontal',
    variable=sliderPosValue,
    # command=slider_changed
)
sliderPos.bind("<ButtonRelease-1>", sliderPosChanged)
sliderPos.pack()

def sliderLowFreqChanged(event):
  update()

sliderLowFreq = tk.Scale(
    r,
    length=sliderWidth,
    from_=sliderMinFreq,
    to=sliderMaxFreq,
    orient='horizontal',
    variable=sliderLowFreqValue,
    # command=slider_changed
)
sliderLowFreq.bind("<ButtonRelease-1>", sliderLowFreqChanged)
sliderLowFreq.pack()

def sliderHighFreqChanged(event):
  update()

sliderHighFreq = tk.Scale(
    r,
    length=sliderWidth,
    from_=sliderMinFreq,
    to=sliderMaxFreq,
    orient='horizontal',
    variable=sliderHighFreqValue,
    # command=slider_changed
)
sliderHighFreq.set(sliderMaxFreq)
sliderHighFreq.bind("<ButtonRelease-1>", sliderHighFreqChanged)
sliderHighFreq.pack()

def sliderVolumeChanged(event):
  update()

sliderVolume = tk.Scale(
    r,
    length=sliderWidth,
    from_=0,
    to=100,
    orient='horizontal',
    variable=sliderVolumeValue,
    # command=slider_changed
)
sliderVolume.set(50)
sliderVolume.bind("<ButtonRelease-1>", sliderVolumeChanged)
sliderVolume.pack()

fmax = argmax(f > maxFreqPlots)
S0 = S[:fmax, :, 0]
P0 = np.log(S0, where=S0>0)
f0 = f[:fmax]
plt.figure(figsize=figsizeSpectro, dpi=dpi, ) # try to fit slider width
plt.pcolormesh(t, f0, P0)
plt.tight_layout()

plt.ion() # non blocking
plt.show()

r.mainloop()
