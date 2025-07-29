import numpy as np
import matplotlib.pylab as plt
from scipy.io import wavfile
from scipy.signal import fftconvolve
import pyroomacoustics as pra
import sounddevice as sd

# add sound source to room
fs, signal = wavfile.read("Samples/guitar_16k.wav")
# # Play the sound
# sd.play(signal, fs)

# # Wait until playback is finished
# sd.wait()

# create a shoebox 2D room
corners = np.array([[0,0], [0,3], [4,3], [4,0]]).T
# print(corners)
room = pra.Room.from_corners(corners, fs=fs, ray_tracing=True, air_absorption=True)
# add source to the room
room.add_source([1.,1.], signal=signal)

# add microphone array
R = pra.square_2D_array(center=[2., 1.5], M=2, N =2, phi=4, d= 0.2) # 4 microphones in total
room.add_microphone_array(pra.MicrophoneArray(R, room.fs))

fig,ax = room.plot()
ax.set_xlim([-1,5])
ax.set_ylim([-1,4])
plt.show()