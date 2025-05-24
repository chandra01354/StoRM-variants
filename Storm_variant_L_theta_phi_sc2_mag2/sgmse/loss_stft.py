import torch
import torch.nn.functional as F
import torchaudio
import torchaudio.transforms as T
import torchaudio.functional as F_audio  # For ISTFT
import librosa

def SCLoss(x_mag, y_mag):
    """Calculate forward propagation.
    Args:
        x_mag (Tensor): Magnitude spectrogram of predicted signal (B, #frames, #freq_bins).
        y_mag (Tensor): Magnitude spectrogram of groundtruth signal (B, #frames, #freq_bins).
    Returns:
        Tensor: Spectral convergence loss value.
    """
    return torch.norm(y_mag - x_mag, p="fro") / (torch.norm(y_mag, p="fro") + 1e-9)


'''
def stft(x, fs, framesz, hop):
    framesamp = int(framesz*fs)
    hopsamp = int(hop*fs)
    w = scipy.hanning(framesamp)
    X = scipy.array([scipy.fft(w*x[i:i+framesamp]) 
                     for i in range(0, len(x)-framesamp, hopsamp)])
    return X

def istft(X, fs, T, hop):
    x = scipy.zeros(T*fs)
    framesamp = X.shape[1]
    hopsamp = int(hop*fs)
    for n,i in enumerate(range(0, len(x)-framesamp, hopsamp)):
        x[i:i+framesamp] += scipy.real(scipy.ifft(X[n]))
'''
def LogSTFTMagLoss(x_mag, y_mag):
    """Calculate forward propagation.
    Args:
        x_mag (Tensor): Magnitude spectrogram of predicted signal (B, #frames, #freq_bins).
        y_mag (Tensor): Magnitude spectrogram of groundtruth signal (B, #frames, #freq_bins).
    Returns:
        Tensor: Log STFT magnitude loss value.
    """
    return F.l1_loss(torch.log(y_mag.squeeze(0) + 1e-9), torch.log(x_mag.squeeze(0) + 1e-9))



def SpectrogramToWaveform(spec, sample_rate=16000, n_fft=510, hop_length=256, win_length=510, device=None):
    """Convert a magnitude spectrogram back to waveform using inverse STFT."""
    # Create phase (zero phase, assuming no phase information)
    phase = torch.zeros_like(spec)  # zero phase approximation
    complex_spec = spec * torch.exp(1j * phase)  # complex spectrogram (magnitude only)

    # Ensure complex_spec is on the same device as the input tensor
    if device is None:
        device = spec.device  # Use spec's device if no device is specified
    complex_spec = complex_spec.to(device)

    # Inverse STFT requires real and imaginary parts separately
    real_part = complex_spec.real
    imag_part = complex_spec.imag

    # Combine real and imaginary parts to form the complex spectrogram
    complex_spec_real_imag = torch.stack((real_part, imag_part), dim=-1)

    # Compute the waveform using torch.istft
    # We need to pass the real and imaginary parts separately
    waveform = torch.istft(complex_spec, n_fft=n_fft, hop_length=hop_length, win_length=win_length, window=torch.hann_window(win_length).to(device))

    return waveform

def MFCCLoss(x_mag, y_mag, sample_rate=16000, n_fft=510, hop_length=256, win_length=510, n_mfcc=40):
    """Calculate MFCC loss between the predicted and ground truth spectrograms.
    Args:
        x_mag (Tensor): Magnitude spectrogram of predicted signal (B, #frames, #freq_bins).
        y_mag (Tensor): Magnitude spectrogram of groundtruth signal (B, #frames, #freq_bins).
        sample_rate (int): The sample rate of the audio.
        n_mfcc (int): Number of MFCC coefficients to compute.
    Returns:
        Tensor: MFCC loss value.
    """
    # Convert the magnitude spectrogram back to waveform using inverse STFT
    x_waveform = SpectrogramToWaveform(x_mag.squeeze(0), sample_rate, n_fft, hop_length, win_length)
    y_waveform = SpectrogramToWaveform(y_mag.squeeze(0), sample_rate, n_fft, hop_length, win_length)
    
    # Convert waveforms to NumPy arrays for librosa
    x_waveform_np = x_waveform.cpu().detach().numpy()
    y_waveform_np = y_waveform.cpu().detach().numpy()
    
    # Ensure proper shape for MFCC computation
    x_waveform_np = x_waveform_np.flatten() if x_waveform_np.ndim > 1 else x_waveform_np
    y_waveform_np = y_waveform_np.flatten() if y_waveform_np.ndim > 1 else y_waveform_np
    
    # Compute MFCC using librosa
    x_mfcc = librosa.feature.mfcc(y=x_waveform_np, sr=sample_rate, n_mfcc=n_mfcc)
    y_mfcc = librosa.feature.mfcc(y=y_waveform_np, sr=sample_rate, n_mfcc=n_mfcc)
    
    # Ensure that both MFCCs have the same shape
    if x_mfcc.shape != y_mfcc.shape:
        raise ValueError("MFCC dimensions mismatch between predicted and ground truth.")

    # Calculate the MSE loss between MFCCs for each sample in the batch
    mse_loss = F.mse_loss(torch.tensor(x_mfcc), torch.tensor(y_mfcc), reduction='none')  # 'none' keeps the loss per element
    
    # Return the mean of the per-element losses
    return mse_loss




def MinMaxNormalizeLoss(loss, eps=1e-9):
    """Normalize the MSE loss across the batch to range [0, 1] using min-max normalization."""
    min_loss = loss.min()
    max_loss = loss.max()
    return (loss - min_loss) / (max_loss - min_loss + eps)  # Normalize to [0, 1]


def STFTLoss(x, y):
    sc_loss = SCLoss(x, y)
    mag_loss = LogSTFTMagLoss(x, y)
    return sc_loss, mag_loss


def compute_mean_stft_loss(x_hat, x, sample_rate=16000, n_fft=510, hop_length=256, win_length=510):
    """
    Compute the mean STFT loss for a batch of inputs.

    Args:
        x_hat (torch.Tensor): Predicted batch tensor (B, 1, H, W).
        x (torch.Tensor): Ground truth batch tensor (B, 1, H, W).
        sample_rate (int): The sample rate of the audio.
        n_fft (int): FFT window size.
        hop_length (int): Hop length for STFT.
        win_length (int): Window length for STFT.
    
    Returns:
        tuple: Mean spectral convergence loss, mean magnitude loss, and mean normalized MFCC loss.
    """
    # Initialize variables to accumulate the losses
    total_sc_loss = 0.0
    total_mag_loss = 0.0
    total_mfcc_loss = []

    # Loop through each item in the batch
    for i in range(x_hat.size(0)):
        sc_loss, mag_loss = STFTLoss(x_hat[i].squeeze(0), x[i].squeeze(0))
        #mfcc_loss = MFCCLoss(x_hat[i].squeeze(0), x[i].squeeze(0), sample_rate, n_fft, hop_length, win_length)

        total_sc_loss += sc_loss
        total_mag_loss += mag_loss
        #total_mfcc_loss.append(mfcc_loss.mean())  # Store individual MFCC loss for each sample in the batch

    # Normalize MFCC loss for each sample in the batch
    #total_mfcc_loss = torch.stack(total_mfcc_loss)  # Stack into a tensor
    #normalized_mfcc_loss = MinMaxNormalizeLoss(total_mfcc_loss)  # Min-Max normalize the loss

    # Compute the mean of the losses
    mean_sc_loss = total_sc_loss / x_hat.size(0)
    mean_mag_loss = total_mag_loss / x_hat.size(0)
    
    # Compute mean of normalized MFCC loss
    #mean_mfcc_loss = normalized_mfcc_loss.mean()

    return mean_sc_loss, mean_mag_loss#, mean_mfcc_loss

def multi_resolution_stft_loss(x, y, sample_rate=16000,fft_sizes=[1024, 2048, 512], hop_sizes=[120, 240, 50], win_lengths=[600, 1200, 240], window="hann_window", factor_sc=0.1, factor_mag=0.1):
    """Calculate Multi-resolution STFT loss (spectral convergence + log STFT magnitude) across different resolutions.
    Args:
        x (Tensor): Predicted signal (B, T).
        y (Tensor): Groundtruth signal (B, T).
        fft_sizes (list): List of FFT sizes.
        hop_sizes (list): List of hop sizes.
        win_lengths (list): List of window lengths.
        window (str): Window function type.
        factor_sc (float): Weight for spectral convergence loss.
        factor_mag (float): Weight for log STFT magnitude loss.
    Returns:
        Tensor: Multi-resolution spectral convergence loss value.
        Tensor: Multi-resolution log STFT magnitude loss value.
    """
    sc_loss = 0.0
    mag_loss = 0.0
    for i in range(x.size(0)):

        x_waveform = SpectrogramToWaveform(x[i], sample_rate=16000, n_fft=510, hop_length=256, win_length=510)
        y_waveform = SpectrogramToWaveform(y[i], sample_rate=16000, n_fft=510, hop_length=256, win_length=510)
	    
	   
        for fft_size, hop_size, win_length in zip(fft_sizes, hop_sizes, win_lengths):
            sc_l, mag_l = stft_loss(x_waveform, y_waveform, fft_size, hop_size, win_length, window)
            sc_loss += sc_l
            mag_loss += mag_l
	    
	    # Average loss across the batch and resolutions
        sc_loss /= (len(fft_sizes) * x.size(0))
        mag_loss /= (len(fft_sizes) * x.size(0))

    return factor_sc * sc_loss, factor_mag * mag_loss

def stft_loss(x, y, fft_size, shift_size, win_length, window="hann_window"):
    """Calculate STFT loss (spectral convergence + log STFT magnitude).
    Args:
        x (Tensor): Predicted signal (B, T).
        y (Tensor): Groundtruth signal (B, T).
    Returns:
        Tensor: Spectral convergence loss value.
        Tensor: Log STFT magnitude loss value.
    """
    x_mag = stft(x, fft_size, shift_size, win_length, window)
    y_mag = stft(y, fft_size, shift_size, win_length, window)
    
    sc_loss = spectral_convergence_loss(x_mag, y_mag)
    mag_loss = log_stft_magnitude_loss(x_mag, y_mag)

    return sc_loss, mag_loss

def stft(x, fft_size, hop_size, win_length, window="hann_window"):
    """Perform STFT and convert to magnitude spectrogram.
    Args:
        x (Tensor): Input signal tensor (B, T).
        fft_size (int): FFT size.
        hop_size (int): Hop size.
        win_length (int): Window length.
        window (str): Window function type (e.g., 'hann_window').
    Returns:
        Tensor: Magnitude spectrogram (B, #frames, fft_size // 2 + 1).
    """
    # Generate the window tensor using the appropriate window function
    if window == "hann_window":
        window_tensor = torch.hann_window(win_length)
    else:
        raise ValueError(f"Unsupported window type: {window}")

    # Move the window to the same device as the input tensor `x`
    window_tensor = window_tensor.to(x.device)

    # Apply the STFT operation with the correct arguments
    x_stft = torch.stft(x, n_fft=fft_size, hop_length=hop_size, win_length=win_length, window=window_tensor, onesided=True, return_complex=True)  # Set return_complex=True

    #x_stft = torch.stft(x, n_fft=fft_size, hop_length=hop_size, win_length=win_length, window=window_tensor)
    
    # Extract real and imaginary parts
    real = x_stft.real
    imag = x_stft.imag
    
    #x_stft = torch.stft(x, fft_size, hop_size, win_length, window_tensor)
    #real = x_stft[..., 0]
    #imag = x_stft[..., 1]
    
    # NOTE(kan-bayashi): clamp is needed to avoid nan or inf
    #return torch.sqrt(torch.clamp(real ** 2 + imag ** 2, min=1e-7)).transpose(2, 1)

    # Compute magnitude and return
    return torch.sqrt(torch.clamp(real ** 2 + imag ** 2, min=1e-7))#.transpose(2, 1)

def spectral_convergence_loss(x_mag, y_mag):
    """Calculate Spectral Convergence Loss.
    Args:
        x_mag (Tensor): Magnitude spectrogram of predicted signal (B, #frames, #freq_bins).
        y_mag (Tensor): Magnitude spectrogram of groundtruth signal (B, #frames, #freq_bins).
    Returns:
        Tensor: Spectral convergence loss value.
    """
    return torch.norm(y_mag - x_mag, p="fro") / torch.norm(y_mag, p="fro")


def log_stft_magnitude_loss(x_mag, y_mag):
    """Calculate Log STFT Magnitude Loss.
    Args:
        x_mag (Tensor): Magnitude spectrogram of predicted signal (B, #frames, #freq_bins).
        y_mag (Tensor): Magnitude spectrogram of groundtruth signal (B, #frames, #freq_bins).
    Returns:
        Tensor: Log STFT magnitude loss value.
    """
    return F.l1_loss(torch.log(y_mag), torch.log(x_mag))

