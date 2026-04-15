#!/usr/bin/env python3
"""
Starting with example plot_input.py from https://python-sounddevice.readthedocs.io/en/0.3.14/examples.html#plot-microphone-signal-s-in-real-time

Docs to reference:
Sounddevice: https://python-sounddevice.readthedocs.io/en/0.5.3/
Sounddevice InputStream: https://python-sounddevice.readthedocs.io/en/latest/api/streams.html#sounddevice.InputStream
Matplotlib Animation: https://matplotlib.org/stable/api/animation_api.html
Numpy Roll: https://numpy.org/doc/stable/reference/generated/numpy.roll.html
"""
import argparse
import queue
import sys

from matplotlib.animation import FuncAnimation
import matplotlib.pyplot as plt
import numpy as np
import sounddevice as sd

from scipy.fft import fft, fftfreq, rfft, rfftfreq
from scipy.signal import find_peaks, savgol_filter

# https://pypi.org/project/noisereduce/
import noisereduce as nr


def int_or_str(text):
    """Helper function for argument parsing."""
    try:
        return int(text)
    except ValueError:
        return text

# TAKE IN COMMAND LINE ARGUMENTS
parser = argparse.ArgumentParser(add_help=False)

parser.add_argument(
    '-l', '--list-devices', action='store_true',
    help='show list of audio devices and exit')
args, remaining = parser.parse_known_args()
if args.list_devices:
    print(sd.query_devices())
    parser.exit(0)

parser = argparse.ArgumentParser(
    description=__doc__,
    formatter_class=argparse.RawDescriptionHelpFormatter,
    parents=[parser])

parser.add_argument(
    'channels', type=int, default=[1], nargs='*', metavar='CHANNEL',
    help='input channels to plot (default: the first)')

parser.add_argument(
    '-d', '--device', type=int_or_str,
    help='input device (numeric ID or substring)')

parser.add_argument(
    '-w', '--window', type=float, default=200, metavar='DURATION',
    help='visible time slot (default: %(default)s ms)')

parser.add_argument(
    '-i', '--interval', type=float, default=30,
    help='minimum time between plot updates (default: %(default)s ms)')

parser.add_argument(
    '-b', '--blocksize', type=int, default=4096, help='block size (in samples)')

parser.add_argument(
    '-r', '--samplerate', type=float, help='sampling rate of audio device')

parser.add_argument(
    '-n', '--downsample', type=int, default=20, metavar='N',
    help='display every Nth sample (default: %(default)s)')

args = parser.parse_args(remaining)

if any(c < 1 for c in args.channels):
    parser.error('argument CHANNEL: must be >= 1')
mapping = [c - 1 for c in args.channels]  # Channel numbers start with 1


q = queue.Queue()


def inject_test_signal():
    """Push a known sine wave mix into the queue instead of mic input."""
    duration = args.window / 1000  # seconds
    t = np.linspace(0, duration, int(args.samplerate * duration), endpoint=False)
    
    # Create a signal with known frequencies: 440 Hz (A4) + 1000 Hz tone
    test_signal = (
        0.5 * np.sin(2 * np.pi * 440 * t) +
        0.3 * np.sin(2 * np.pi * 1000 * t)
    )
    
    # Shape to match what the callback produces: (samples//downsample, channels)
    test_signal = test_signal[::args.downsample].reshape(-1, 1)
    q.put(test_signal)

def audio_callback(indata, frames, time, status):
    """This is called (from a separate thread) for each audio block."""
    if status:
        print(status, file=sys.stderr)
    q.put(indata[::args.downsample, mapping])

def harmonic_product_spectrum(amplitude, num_harmonics=5):
    hps = amplitude.copy()
    for h in range(2, num_harmonics + 1):
        # Downsample amplitude by factor h and multiply
        downsampled = amplitude[::h]
        hps[:len(downsampled)] *= downsampled
    return hps

def get_fundamental(amplitude, xf, min_freq=60, max_freq=200):
    # Restrict search range first
    mask = (xf >= min_freq) & (xf <= max_freq)
    masked_amp = np.zeros_like(amplitude)
    masked_amp[mask] = amplitude[mask]
    
    hps = harmonic_product_spectrum(masked_amp)
    
    # Find peak within valid range
    valid = np.where(mask)[0]
    peak_idx = valid[np.argmax(hps[valid])]
    return xf[peak_idx]

def update_plot(frame):
    global plotdata
    while True:
        try:
            data = q.get_nowait()
        except queue.Empty:
            break

        shift = len(data)
        plotdata = np.roll(plotdata, -shift, axis=0)
        plotdata[-shift:, :] = data

    N = len(plotdata)
    downsampled_rate = args.samplerate / args.downsample
    xf = rfftfreq(N, 1/downsampled_rate)
    mask = (xf >= 60) & (xf <= 2000)
    
    # take the data for each channel
    for column, line in enumerate(lines):
        channel_data = plotdata[:, column]

        channel_data = channel_data - np.mean(channel_data)

        yf = rfft(channel_data)
        amplitude = np.abs(yf)
        
        # bin_width = xf[1] - xf[0]
        # print(f"bin_width={bin_width:.4f} Hz, N={N}, samplerate={args.samplerate}, length={length}")

        
        raw_peak = xf[np.argmax(amplitude * mask)]
        print(f"raw peak: {raw_peak:.1f} Hz")
        # print(get_fundamental(amplitude, xf))
        # Plot amplitude vs frequencies
        line.set_data(xf, amplitude)
    return lines


try:
    if args.samplerate is None:
        device_info = sd.query_devices(args.device, 'input')
        args.samplerate = device_info['default_samplerate']
        # Add this right after args.samplerate is set:
        bin_width = 1000 * args.downsample / args.window
        print(f"samplerate={args.samplerate}, window={args.window}ms, downsample={args.downsample}")
        print(f"bin width = {bin_width:.2f} Hz — nearest bin to 440 Hz: {round(440/bin_width)*bin_width:.1f} Hz")

    # length of input sample in s = visible window (ms) * sample rate / 1000 (to convert to s) * see every downsampleth sample
    length = int(args.window * args.samplerate / (1000 * args.downsample))
    # set up empty np array the size of our sample
    plotdata = np.zeros((length, len(args.channels)))
    
    # set up empty plot
    fig, ax = plt.subplots()
    lines = ax.plot(plotdata)
    if len(args.channels) > 1:
        ax.legend(['channel {}'.format(c) for c in args.channels],
                  loc='lower left', ncol=len(args.channels))
   
    # ax.set_yticks([0])
    ax.yaxis.grid(True)
    ax.set_xlim(0, args.samplerate / (2*args.downsample))
    ax.set_ylim(0,10)
    ax.set_xlabel("Frequency (Hz)")
    ax.set_ylabel("Amplitude")
    ax.set_title("Live FFT Spectrum")
    fig.tight_layout()

    stream = sd.InputStream(
        device=args.device, channels=max(args.channels),
        samplerate=args.samplerate, callback=audio_callback)

    ani = FuncAnimation(fig, update_plot, interval=args.interval, blit=True, cache_frame_data=False)
    with stream:
        plt.show()
except Exception as e:
    parser.exit(0, f"{type(e).__name__} : {str(e)}")
