import sre_compile
import torch
from torchaudio import load
from sgmse.util.other import si_sdr, pad_spec
from pesq import pesq
from tqdm import tqdm
from pystoi import stoi
import numpy as np
from SRMRpy.srmrpy import srmr
import soundfile as sf
from scipy.io import wavfile

# Settings
snr = 0.5
N = 50
corrector_steps = 1

# Plotting settings
MAX_VIS_SAMPLES = 10
n_fft = 512
hop_length = 128

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def srmrpy_eval_valid(predict, target):
    """ Note target_wav is not used in the srmr function !!!
        Show the unnormalized score for valid and test set.
    """
    return float(
        srmr(
            predict,
            fs=16000,
            n_cochlear_filters=23,
            low_freq=125,
            min_cf=4,
            max_cf=128,
            fast=True,
            norm=False,
        )[0]
    )

def dnsmos_eval_valid(predict, target):
    """ Note target_wav is not used in the dnsmos function !!!
        Show the unnormalized score for valid and test set.
    """
    pred_wav = predict

    pred_wav = pred_wav.numpy()
    pred_wav = pred_wav / max(abs(pred_wav))
    data = {"data": pred_wav.tolist()}
    input_data = json.dumps(data)
    while True:
        try:
            u = urlparse(SCORING_URI)
            resp = requests.post(
                urljoin("https://" + u.netloc, "score"),
                data=input_data,
                headers=headers,
            )
            score_dict = resp.json()
            score = float(score_dict["mos"])
            break
        except Exception as e:  # sometimes, access the dnsmos server too ofen may disable the service.
            print(e)
            time.sleep(10)  # wait for 10 secs
    return score

# Function to compute SSNR
'''
def compute_ssnr(clean_wav_path, enhanced_wav_path):
    """
    Computes the Normalized Segmental Signal-to-Noise Ratio (SSNR) for a clean and an enhanced speech file.
    
    Parameters:
    - clean_wav_path: Path to the clean speech WAV file
    - enhanced_wav_path: Path to the enhanced speech WAV file
    
    Returns:
    - normalized_ssnr: Normalized SSNR score for the provided clean and enhanced WAV files
    """
    # Set default frame length and frame shift
    frame_length = 256
    frame_shift = 128
    eps = 1e-10  # Small constant to avoid log(0) or division by zero

    # Load the waveform from the clean and enhanced files
    clean_speech, fs_clean = sf.read(clean_wav_path)
    enhanced_speech, fs_enhanced = sf.read(enhanced_wav_path)

    # Ensure the sample rates of the files match
    if fs_clean != fs_enhanced:
        raise ValueError("Sample rates of the clean and enhanced files must match.")
    
    # Compute SSNR for the provided files
    num_frames = (len(clean_speech) - frame_length) // frame_shift + 1
    ssnr_list = []
    
    for j in range(num_frames):
        start = j * frame_shift
        end = start + frame_length
        
        # Extract frames
        clean_frame = clean_speech[start:end]
        enhanced_frame = enhanced_speech[start:end]
        
        # Calculate noise frame (difference between clean and enhanced)
        noise_frame = clean_frame - enhanced_frame
        
        # Calculate signal and noise energy
        signal_energy = np.sum(clean_frame ** 2)
        noise_energy = np.sum(noise_frame ** 2)
        
        # Calculate segmental SNR for the frame if noise energy is above a threshold
        if signal_energy > eps and noise_energy > eps:
            snr_frame = 10 * np.log10(signal_energy / (noise_energy + eps))
            ssnr_list.append(snr_frame)
    
    # Calculate average SSNR over all frames
    average_ssnr = np.mean(ssnr_list) if ssnr_list else 0.0
    
    # Normalize SSNR to range [0, 1]
    normalized_ssnr = (average_ssnr + 10) / 45
    
    return normalized_ssnr
'''

def compute_ssnr(clean_signal, noisy_signal, frame_length=256):
    """Calculates the Segmental Signal-to-Noise Ratio (SSNR).

    Args:
        clean_signal (numpy.ndarray): The clean (reference) audio signal.
        noisy_signal (numpy.ndarray): The noisy audio signal.
        frame_length (int): Length of each frame to compute segmental SNR.

    Returns:
        float: The computed SSNR value.
    """
    # Ensure both signals have the same length
    min_len = min(len(clean_signal), len(noisy_signal))
    clean_signal = clean_signal[:min_len]
    noisy_signal = noisy_signal[:min_len]

    ssnr_values = []
    for i in range(0, len(clean_signal), frame_length):
        clean_frame = clean_signal[i:i + frame_length]
        noisy_frame = noisy_signal[i:i + frame_length]

        # Avoid zero-division error
        if np.sum(clean_frame ** 2) == 0:
            continue

        # Calculate SNR for the current frame
        snr = 10 * np.log10(np.sum(clean_frame ** 2) / np.sum((clean_frame - noisy_frame) ** 2))

        # Clip the SNR to a reasonable range (-10 to 35 dB)
        snr = np.clip(snr, -10, 35)
        ssnr_values.append(snr)

    # Calculate the mean SSNR value
    if len(ssnr_values) > 0:
        ssnr = np.mean(ssnr_values)
    else:
        ssnr = float('-inf')  # In case there are no valid frames

    return ssnr


def evaluate_model(model, num_eval_files, spec=False, audio=False, discriminative=False):

	model.eval()
	_pesq, _si_sdr, _estoi, _ssnr, _srmr, _dnsmos = 0., 0., 0., 0., 0., 0.
	if spec:
		noisy_spec_list, estimate_spec_list, clean_spec_list = [], [], []
	if audio:
		noisy_audio_list, estimate_audio_list, clean_audio_list = [], [], []

	for i in range(num_eval_files):
		# Load wavs
		x, y = model.data_module.valid_set.__getitem__(i, raw=True) #d,t
		norm_factor = y.abs().max().item()
		x_hat = model.enhance(y)

		if x_hat.ndim == 1:
			x_hat = x_hat.unsqueeze(0)
			
		if x.ndim == 1:
			x = x.unsqueeze(0).cpu().numpy()
			x_hat = x_hat.unsqueeze(0).cpu().numpy()
			y = y.unsqueeze(0).cpu().numpy()
		else: #eval only first channel
			x = x[0].unsqueeze(0).cpu().numpy()
			x_hat = x_hat[0].unsqueeze(0).cpu().numpy()
			y = y[0].unsqueeze(0).cpu().numpy()

		_si_sdr += si_sdr(x[0], x_hat[0])
		_pesq += pesq(16000, x[0], x_hat[0], 'wb') 
		_estoi += stoi(x[0], x_hat[0], 16000, extended=True)
		_ssnr += compute_ssnr(x[0], x_hat[0])
		_srmr += srmrpy_eval_valid(x_hat[0], x[0])
		_dnsmos += srmrpy_eval_valid(x_hat[0], x[0])
		
		y, x_hat, x = torch.from_numpy(y), torch.from_numpy(x_hat), torch.from_numpy(x)
		if spec and i < MAX_VIS_SAMPLES:
			y_stft, x_hat_stft, x_stft = model._stft(y[0]), model._stft(x_hat[0]), model._stft(x[0])
			noisy_spec_list.append(y_stft)
			estimate_spec_list.append(x_hat_stft)
			clean_spec_list.append(x_stft)

		if audio and i < MAX_VIS_SAMPLES:
			noisy_audio_list.append(y[0])
			estimate_audio_list.append(x_hat[0])
			clean_audio_list.append(x[0])

	if spec:
		if audio:
			return _pesq/num_eval_files, _si_sdr/num_eval_files, _estoi/num_eval_files, _ssnr/num_eval_files, _srmr/num_eval_files, _dnsmos/num_eval_files, [noisy_spec_list, estimate_spec_list, clean_spec_list], [noisy_audio_list, estimate_audio_list, clean_audio_list]
		else:
			return _pesq/num_eval_files, _si_sdr/num_eval_files, _estoi/num_eval_files, _ssnr/num_eval_files, _srmr/num_eval_files, _dnsmos/num_eval_files, [noisy_spec_list, estimate_spec_list, clean_spec_list], None
	elif audio and not spec:
			return _pesq/num_eval_files, _si_sdr/num_eval_files, _estoi/num_eval_files, _ssnr/num_eval_files, _srmr/num_eval_files, _dnsmos/num_eval_files, None, [noisy_audio_list, estimate_audio_list, clean_audio_list]
	else:
		return _pesq/num_eval_files, _si_sdr/num_eval_files, _estoi/num_eval_files, _ssnr/num_eval_files, _srmr/num_eval_files, _dnsmos/num_eval_files, None, None

