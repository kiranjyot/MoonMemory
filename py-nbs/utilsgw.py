import os
import numpy as np
from scipy import fftpack, signal, interpolate
from scipy.fft import fft, fftfreq, fftshift
from scipy.signal import butter, lfilter, freqz, get_window
from scipy.interpolate import splrep, splev
import matplotlib.pyplot as plt
from matplotlib.ticker import AutoMinorLocator
from matplotlib.colors import Normalize, LogNorm
from matplotlib.cm import ScalarMappable
from gwpy.timeseries import TimeSeries

def compute_fft(time, signal):
    """
    Compute the one-sided Fourier Transform of a signal with doubled amplitudes.

    Parameters:
        time (array): Evenly spaced time array.
        signal (array): Signal array.

    Returns:
        freq (array): One-sided frequency array.
        fft_values (array): One-sided FFT values with doubled amplitudes.
    """
    # Calculate the time step
    delta_t = time[1] - time[0]

    # Compute the Fourier Transform
    fft_values = fft(signal) * delta_t
    freq = fftfreq(len(time), delta_t)

    # Use only the positive frequencies and double the amplitude for non-DC components
    half_n = len(freq) // 2
    freq = freq[:half_n]  # Positive frequencies only
    fft_values = fft_values[:half_n]  # Positive frequencies only
    fft_values[1:] = 2 * np.abs(fft_values[1:])  # Double the amplitude for non-DC components

    return freq, np.abs(fft_values)


def compute_asd(times, signal):
    """
    Compute the Amplitude Spectral Density (ASD) from the signal.

    Args:
        times (numpy array): Array of time values.
        signal (numpy array): Array of signal values.

    Returns:
        frequencies (numpy array): Array of frequencies corresponding to the FFT.
        asd (numpy array): ASD of the signal.
    """
    # Get right-sided FFT
    f, X = compute_fft(times, signal)
    delta_t = times[1] - times[0]
    T = len(times) * delta_t  # Total time length of the signal
    
    # Scale to Power Spectral Density (PSD)
    psd = (1 / (2 * T)) * np.abs(X)**2  # PSD: Proper scaling by total time duration
    
    # Return ASD = sqrt(PSD)
    return f, np.sqrt(psd)


def butterworth_filter(data, fs, f0=10, n=10):
    nyquist = 0.5 * fs
    normal_cutoff = f0 / nyquist
    b, a = signal.butter(n, normal_cutoff, btype='low', analog=False)
    filtered_data = signal.filtfilt(b, a, data)
    return filtered_data


def model_list(model_set, var1='freq', var2='psd', sub=None):
    """
    Extract specified variables from a set of models.

    Parameters:
        model_set (dict): Dictionary containing GW models (matter or neutrino)
        var1 (str): extracts frequencies from each GW model
        var2 (str): extracts the frequencies for each associated PSD 
        sub (str, optional): option access nested data within each GW model (default: None).

    Returns:
        tuple: Two lists containing the extracted values for var1 and var2 from each model.
    """
    if sub:
        return ([model_set[m][sub][var1] for m in model_set], 
                [model_set[m][sub][var2] for m in model_set])
    return ([model_set[m][var1] for m in model_set], 
            [model_set[m][var2] for m in model_set])

def resample_signal(time, signal, new_sampling_rate, method='linear'):
    """
    Resample a signal to a new sampling rate using specified interpolation method.

    Parameters:
        time (array): Original time array.
        signal (array): Original signal array corresponding to the time array.
        new_sampling_rate (float): Desired new sampling rate in Hz.
        method (str): Interpolation method, either 'linear' or 'cubic' (default: 'linear').

    Returns:
        tuple: Resampled time array and corresponding resampled signal array.
    """
    # Validate input lengths
    if len(time) != len(signal):
        raise ValueError("Time and signal arrays must have equal lengths")

    # Calculate the new time array based on the new sampling rate
    new_time = np.arange(time[0], time[-1], 1.0 / new_sampling_rate)

    # Interpolate the signal to the new time array
    if method == 'linear': 
        new_signal = np.interp(new_time, time, signal)
    elif method == 'cubic':
        tck = splrep(time, signal, s=0) # b-spline representation of a 1-D curve; setting s=0 ensures that the spline fits the data points exactly
        new_signal = splev(new_time, tck, der=0) # der=0 specifies that one wants to evaluate the spline itself (not its derivatives)
    else:
        raise ValueError("Unsupported interpolation method: choose 'linear' or 'cubic'")
    return new_time, new_signal

def taper_signal(t, h, fs, T_tail=5.0):
    """
    Taper the given signal data by adding a cosine-shaped tail.

    Parameters:
        t (array): Time array of the original signal.
        h (array): Strain signal array.
        fs (float): Sampling frequency of the signal in Hz.
        T_tail (float): Duration of the tapering tail in seconds (default: 5.0).

    Returns:
        tuple: Extended time and tapered strain arrays.
    """
    # Calculate the frequency of the tail
    f_tail = 1.0 / (T_tail * 2)  # Division by 2 because the tail is half of a cosine function
    # Create the tail
    nr = int(fs * T_tail)  # Number of data points in the tail
    t_tail = t[-1] + np.arange(nr) / fs  # Time array for the tail
    h_max = h[-1]  # Maximum value of the original strain
    h_tail = 0.5 * (h_max + h_max * np.cos(2 * np.pi * f_tail * (t_tail - t_tail[0])))  # Cosine-shaped tail
    # Append the time and strain portions of the tail to the original arrays
    t = np.concatenate((t, t_tail))
    h = np.concatenate((h, h_tail))
    return t, h

def plot_waveforms(models, freqs, strains, ax=None, cmap=None, lw=10, which=None, colors=None):
    """
    Plot the strains for each GW model. Optionally show only a subset of GW models.

    Parameters:
        models (list): List of GW model names.
        freqs (list of arrays): List of frequency arrays for each GW model.
        strains (list of arrays): List of strain arrays for each GW model.
        ax (AxesSubplot, optional): Matplotlib axes to plot on. If None, current axes are used.
        cmap (Colormap, optional): Matplotlib colormap for coloring models.
        lw (int, optional): Line width of the plots.
        which (list, optional): List of GW models to plot. If None, plot all GW models.
        colors (array-like, optional): List of colors for each GW model.

    Returns:
        list: List of plot objects for each GW model's plot.
    """
    if ax is None:
        ax = plt.gca()
    # Normalize colors based on number of models
    norm = Normalize(vmin=0, vmax=len(models))
    # Generate colors if not provided
    if colors is None:
        cmap = cmap or plt.get_cmap('tab20').colors 
        colors = cmap(norm(range(len(models))))
    # Determine which models to plot
    which = which or models
    # Initialize list to store plot objects
    plts = []
    # Iterate over models and plot PSDs
    for i, model in enumerate(models):
        if model not in which:
            continue
        # Plot PSD with units conversion for 10 kpc
        ax.loglog(freqs[i], strains[i] / 3.086e22, color='w', lw=15)
        p, = ax.loglog(freqs[i], strains[i] / 3.086e22, label=f'{model} M$_\odot$', color=colors[i], lw=lw)
        plts.append(p)
    return plts

def make_combined_plot(models, all_frequencies, all_strains):
    """
    Create a combined plot of GW models and GW detector sensitivity curves.
    Parameters:
        models (list): List of GW model names.
        all_frequencies (list of arrays): List of frequency arrays for each GW model.
        all_strain (list of arrays): List of strain arrays for each GW model.

    Returns:
        None
    """
    # Create figure and axes
    fig, ax = plt.subplots(figsize=(41, 24))

    # Plot signal curves
    signal_curves = plot_waveforms(models, all_frequencies, all_strains, ax=ax,lw=7)
    sensitivity_curves = plot_sensitivity_curves(ax)

    # Set labels and formatting
    ax.set_xlabel(r'Frequency [${\rm Hz}$]', fontsize=75)
    ax.set_ylabel(r'Amplitude Spectral Density', fontsize=75)

    # Increase the size of tick labels and tick marks
    ax.tick_params(axis='both', which='major', length=15, width=3, labelsize=55)  # Larger tick labels and marks
    ax.tick_params(axis='both', which='minor', length=10, width=2, labelsize=40)  # Minor tick marks also resized

    # Set limits and grid
    ax.set_xlim(10**-1, 100)
    ax.set_ylim(10**-26, 10**-19.4)
    ax.grid(True, which='both', axis='both')

    # Create legends
    first_legend = ax.legend(handles=signal_curves, loc='upper right', fontsize=40, ncol=2,handleheight=1.8, labelspacing=0.5, handlelength=0.8, markerscale=7)
    ax.add_artist(first_legend)
    ax.legend(handles=sensitivity_curves, loc='lower left', fontsize=45, ncol=2, handleheight=1.8, labelspacing=0.5, handlelength=0.8, markerscale=7)

    # Display the plot
    plt.show()


def style_plot(ax, legends=None):
    """
    Plot styling. 
    Parameters:
        ax (AxesSubplot): Matplotlib axes to style.
        legends (tuple of lists, optional): Tuple containing two lists of legend handles for dual legends.

    Returns:
        None
    """
    ax.set_xlabel(r'Frequency [${\rm Hz}$]', fontsize=75)
    ax.set_ylabel(r'Amplitude Spectral Density [Hz$^{-1/2}$]', fontsize=75)
    ax.xaxis.set_tick_params(pad=10)
    ax.set_xlim(10**-1, 100)
    ax.set_ylim(10**-26, 10**-19.4)
    ax.grid(True, which='both', axis='both')
    if legends is None:
        ax.legend(loc="lower left", ncol=2, handleheight=1.8, fontsize=40)
    else:
        first_legend = ax.legend(handles=legends[0], loc='upper right', fontsize=45, ncol=2,handleheight=1.8, labelspacing=0.5, handlelength=1.5, markerscale=7)
        ax.add_artist(first_legend)
        ax.legend(handles=legends[1], loc='lower left', fontsize=45, ncol=2, handleheight=1.8, labelspacing=0.5, handlelength=1.5, markerscale=7)

def calculate_snr_waveform(time_samples, waveform, frequency_asd, amplitude_asd):
    """
    Calculate the signal-to-noise ratio (SNR) of a waveform.

    Parameters:
        time_samples (array): Time array.
        waveform (array): Waveform array.
        frequency_asd (array): Frequency array for the ASD.
        amplitude_asd (array): Amplitude spectral density array.

    Returns:
        snr (float): Signal-to-noise ratio.
    """
    # Compute FFT of the waveform
    frequencies, fft_waveform = compute_fft(time_samples, waveform)
    fft_waveform = np.abs(fft_waveform) 
    
    # Adjust FFT array to ASD array using spline interpolation
    spline_interp = interpolate.splrep(frequency_asd, amplitude_asd, s=0)
    amplitude_asd_interp = interpolate.splev(frequencies, spline_interp, der=0)
    psd = amplitude_asd_interp ** 2
    
    # Calculate SNR
    df = frequencies[1] - frequencies[0]
    snr_sq = np.sum(fft_waveform ** 2 / psd) * df
    snr = np.sqrt(snr_sq)
    
    return snr

def calculate_partial_snr(t_samples, waveform, freq_asd, amp_asd):
    """
    Calculate the partial signal-to-noise ratio (SNR) of a waveform.

    Parameters:
        t_samples (array): Time array.
        waveform (array): Waveform array.
        freq_asd (array): Frequency array for the ASD.
        amp_asd (array): Amplitude spectral density array.

    Returns:
        snr (float): Partial signal-to-noise ratio.
    """
    # Perform FFT of the waveform
    freq, fft_waveform = compute_fft(t_samples, waveform)
    fft_waveform = np.abs(fft_waveform)
    # Adjust FFT array to ASD array using spline interpolation
    spline_interp = interpolate.splrep(freq_asd, amp_asd, s=0)
    amp_asd_interp = interpolate.splev(freq, spline_interp, der=0)
    # Filter frequencies between 0.1 and 10 Hz
    freq_mask = (freq >= 0.1) & (freq <= 10)
    freq = freq[freq_mask]
    fft_waveform = fft_waveform[freq_mask]
    amp_asd_interp = amp_asd_interp[freq_mask]
    # Calculate PSD
    df = freq[1] - freq[0]
    psd = amp_asd_interp ** 2
    # Calculate SNR
    snr_sq = np.sum((fft_waveform ** 2) / psd) * df
    snr = np.sqrt(snr_sq)
    return snr

import os 
def plot_sensitivity_curves(ax=None):
    """
    Plot sensitivity curves for different GW detectors.

    Parameters:
        ax (matplotlib.axes.Axes, optional): Axes to plot on. If not provided, current axes will be used.

    Returns:
        list: List of plot objects for creating legends.
    """
    sensitivity_dir = 'Data/Sensitivity Curves/'
    sensitivity_files = {
        'ALIA':'ALIA.txt',
        'DO':'DeciHzobs.txt',
        'DECIGO': 'DECIGO1.txt',
        'BDECIGO': 'BDECIGO1.txt',
        'LILA': 'LILA.txt',
        'GLOC': 'GLOC_optimal.txt',
        'LGWAnb': 'LGWAnb.txt',
        'LGWAsi': 'LGWASi.txt',
        'Asharp': 'Asharp.txt',
        'CE': 'CE1_PSD.txt',
        'ET': 'ET_D.txt',
        'LIGO_O3_L1': 'aligo_O3actual_L1.txt',
        'atom':'ATOMifo.txt',
        'TianGO':'TianGO.txt',
        'BBO':'BBO.txt',
    }

    if ax is None:
        ax = plt.gca()

    pls = []
    for label, filename in sensitivity_files.items():
        file_path = os.path.join(sensitivity_dir, filename)
        data = np.genfromtxt(file_path)
        if label == 'LIGO_O3_L1':
            ax.loglog(data[:, 0], data[:, 1], lw=10, color='w')
            pl, = ax.loglog(data[:, 0], data[:, 1], lw=4, color='grey', label='LIGO O3')
        elif label == 'Asharp':
            ax.loglog(data[:, 0], data[:, 1], lw=10, color='w')
            pl, = ax.loglog(data[:, 0], data[:, 1], lw=4, color='black', label='A#')
        elif label == 'CE':
            ax.loglog(data[:, 0], np.sqrt(data[:, 1]), lw=10, color='w')
            pl, = ax.loglog(data[:, 0], np.sqrt(data[:, 1]), lw=4, color='darkorange', label='Cosmic Explorer (L = 40 km)')
        elif label == 'ET':
            ax.loglog(data[:, 0], data[:, 1], lw=10, color='w')
            pl, = ax.loglog(data[:, 0], data[:, 1], lw=4, color='blue', label='Einstein Telescope (Design D)')
        elif label == 'LGWAnb':
            ax.loglog(data[:, 0], np.sqrt(data[:, 1]), lw=6, color='w')
            pl, = ax.loglog(data[:, 0], np.sqrt(data[:, 1]), lw=3, color='tan', label='LGWA (Nb)')
        elif label == 'LGWAsi':
            ax.loglog(data[:, 0], np.sqrt(data[:, 1]), lw=6, color='w')
            pl, = ax.loglog(data[:, 0], np.sqrt(data[:, 1]), lw=3, color='sienna', label='LGWA (Si)')
        elif label == 'LILA':
            ax.loglog(data[:, 0], data[:, 1], lw=10, color='w')
            pl, = ax.loglog(data[:, 0], data[:, 1], lw=14, color='deeppink', label='LILA')
        elif label == 'ALIA':
            ax.loglog(data[:, 0], data[:, 1], lw=10, color='w')
            pl, = ax.loglog(data[:, 0], data[:, 1], lw=14, color='lightpink', label='ALIA')
        elif label == 'GLOC':
            ax.loglog(data[:, 0], data[:, 1], lw=10, color='w')
            pl, = ax.loglog(data[:, 0], data[:, 1], lw=4, color='deepskyblue', label='(Optimal) GLOC')
        elif label == 'DECIGO':
            ax.loglog(data[:, 0], data[:, 1], lw=10, color='w')
            pl, = ax.loglog(data[:, 0], data[:, 1], lw=4, color='navy', label='DECIGO')
        elif label == 'BDECIGO':
            ax.loglog(data[:, 0], data[:, 1], lw=10, color='w')
            pl, = ax.loglog(data[:, 0], data[:, 1], lw=4, color='teal', label='B-DECIGO')
        elif label == 'DO':
            ax.loglog(data[:, 0], data[:, 1], lw=10, color='w')
            pl, = ax.loglog(data[:, 0], data[:, 1], lw=4, color='lawngreen', label='Deci-Hz Obs')
        elif label == 'atom':
            ax.loglog(data[:, 0], data[:, 1], lw=10, color='w')
            pl, = ax.loglog(data[:, 0], data[:, 1], lw=4, color='tomato', label='Atomic Clock')
        elif label == 'TianGO':
            ax.loglog(data[:, 0], data[:, 1], lw=10, color='w')
            pl, = ax.loglog(data[:, 0], data[:, 1], lw=4, color='olive', label='TianGO')
        elif label == 'BBO':
            ax.loglog(data[:, 0], data[:, 1], lw=10, color='w')
            pl, = ax.loglog(data[:, 0], data[:, 1], lw=4, color='coral', label='BBO')
        else:
            ax.loglog(data[:, 0], data[:, 1], lw=10, color='w')
            pl, = ax.loglog(data[:, 0], data[:, 1], lw=10, label=label)
        pls.append(pl)

    # Set axis labels
    ax.set_xlabel('Frequency [Hz]', fontsize=44)
    ax.set_ylabel('Amplitude Spectral Density [Hz$^{-1/2}$]', fontsize=44)
    
    # Set tick sizes
    ax.tick_params(axis='both', which='major', labelsize=60, width=2)
    ax.tick_params(axis='both', which='minor', labelsize=60, width=1.5)
    
    # Set grid
    ax.grid(True, which='both', linestyle='--', linewidth=0.8)
    
    return pls
