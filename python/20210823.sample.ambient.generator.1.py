# 20210823.sample.ambient.generator.py

from operator import mul
import audiosegment
from pydub.playback import play
import numpy as np
import matplotlib.pyplot as plt
import scipy.io.wavfile
import winsound
from scipy import signal
from scipy.fft import fftshift
from scipy import interpolate
# from pyo import *

## -------------------------------------------------------
inputName = "MissionThree.m4a"
seg = audiosegment.from_file(inputName)

name = "MissionThree.aif"
seg.export(name)

## -------------------------------------------------------
## AUDIOSEGMENT SPECTROGRAM
## does not work, but i think should take first channel (column)

# freqs, times, amplitudes = seg.spectrogram(window_length_s=2., overlap=0., duration_s=10)
# amplitudes = 10 * np.log10(amplitudes + 1e-9)
# # Plot
# plt.pcolormesh(times, freqs, amplitudes)
# plt.xlabel("Time in Seconds")
# plt.ylabel("Frequency in Hz")
# plt.show()

## -------------------------------------------------------
## SCIPY SPECTROGRAM

# f, t, Sxx = signal.spectrogram(s, fs, nperseg=int(2 * fs), noverlap=int(2 * fs)-1000)

# plt.pcolormesh(t, f, Sxx**0.1)
# plt.ylabel('Frequency [Hz]')
# plt.xlabel('Time [sec]')

# plt.show()

## -------------------------------------------------------
## LOOP SEGMENT WITH PYO

# s = Server().boot()



# x = seg.to_numpy_array()
# x = x / x.max()

# fs = seg.frame_rate

# # xx = x[440000:445000, :]
# xx = x[200000:240000, :]

# # w = signal.windows.hann(100)
# # wl = w[0:50]
# # wr = w[50:]
# # xx[0:len(wl), 0] = wl * xx[0:len(wl), 0]
# # xx[-len(wr):, 0] = wr * xx[-len(wr):, 0]
# # xx[0:len(wl), 1] = wl * xx[0:len(wl), 1]
# # xx[-len(wr):, 1] = wr * xx[-len(wr):, 1]

# xx = np.concatenate([xx, np.flipud(xx)], axis=0)

# # Loads the sound file in RAM. Beginning and ending points
# # can be controlled with "start" and "stop" arguments.
# # t = SndTable(name, start=400000, stop=444000)
# # t = DataTable(a.shape[1], chnls=a.shape[0], init=a.tolist())
# # t = DataTable(xx.shape[0], init=xx.tolist())
# t = DataTable(xx.shape[0], chnls=xx.shape[1], init=[xx[:,0].tolist(), xx[:,1].tolist()])

# # Gets the frequency relative to the table length.
# freq = t.getRate()

# # Simple stereo looping playback (right channel is 180 degrees out-of-phase).
# o = Osc(table=t, freq=freq)#.out()
# # o.ctrl()
# d = Delay(o)#.out()
# d.ctrl()
# fv = Freeverb(d).out()
# fv.ctrl()


# # sf = SfPlayer(name, loop=True, mul=0.4).out() # ERROR

# # sc = Scope(osc)

# s.gui(locals())

## -------------------------------------------------------
## UPSAMPLE FFT FROM SHORT SAMPLE

fs = seg.frame_rate
x = seg.to_numpy_array()
x = x / x.max()
x = x[200000:240000, :]

def write(x, name, volume=1.0):
  A = volume * (2**15-1) # int16 scale factor; 2^16/2, since signed, -1 to avoid saturation
  x *= A/np.max(np.abs(x)) # normalize the signal to span the int16 domain
  scipy.io.wavfile.write(name, fs, x.astype(np.int16))

def loop(s, f):
  play(s, f, loop=True)

def play(s, f, volume=1.0, loop=False):
  f = "./songs/" + f
  write(s, f, volume=volume)
  play = True
  while (play):
    winsound.PlaySound(f, 0)
    play = loop


F = np.fft.fft(x, axis=0)

k = 20
Gr = np.zeros((F.shape[0] * k, 2))
Gi = np.zeros((F.shape[0] * k, 2))
G = np.zeros((F.shape[0] * k, 2))
fi = np.fft.fftfreq(x.shape[0])
fi = np.fft.fftshift(fi)
fo = np.linspace(fi[0], fi[-1], num=len(fi) * k)

# fo = np.fft.fftshift(fo)
# fi = np.fft.fftshift(fi)
F = np.fft.fftshift(F, axes=0)


Gr[:,0] = np.interp(fo, fi, F[:,0].real)
Gr[:,1] = np.interp(fo, fi, F[:,1].real)
Gi[:,0] = np.interp(fo, fi, F[:,0].imag)
Gi[:,1] = np.interp(fo, fi, F[:,1].imag)

# fx = interpolate.interp1d(fi, F[:,0].real)
# Gr[:,0] = fx(fo)
# fx = interpolate.interp1d(fi, F[:,1].real)
# Gr[:,1] = fx(fo)
# fx = interpolate.interp1d(fi, F[:,0].imag)
# Gi[:,0] = fx(fo)
# fx = interpolate.interp1d(fi, F[:,1].imag)
# Gi[:,1] = fx(fo)

G = Gr + 1j * Gi
G = np.fft.fftshift(G)

G = np.abs(G) * np.exp(1j * np.random.rand(G.shape[0], G.shape[1]) * np.pi)

print("interpolation done")

# F = np.concatenate([np.zeros((10000, 2)), F, np.zeros((10000, 2))], axis=0)
x = np.real(np.fft.ifft(G, axis=0))
x[:,0] = x[:,0] * signal.windows.hann(x.shape[0])
x[:,1] = x[:,1] * signal.windows.hann(x.shape[0])

print("ifft done")

loop(x, 'test.sample.wav')

# play(seg)