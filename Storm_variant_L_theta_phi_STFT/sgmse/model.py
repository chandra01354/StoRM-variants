from math import ceil
import warnings
import logging #CAIR
import matplotlib.pyplot as plt

import torch
import pytorch_lightning as pl
from torch_ema import ExponentialMovingAverage
import wandb
import time
import os
import numpy as np

from scipy.signal import istft  # Import iSTFT function from SciPy

from pesq import pesq
from pystoi import stoi
from joblib import Parallel, delayed
from SRMRpy.srmrpy import srmr
from sgmse import sampling
from sgmse.sdes import SDERegistry
from sgmse.backbones import BackboneRegistry
from sgmse.util.inference import evaluate_model
from sgmse.util.graphics import visualize_example, visualize_one
from sgmse.util.other import pad_spec, si_sdr_torch
from sgmse.loss_stft import compute_mean_stft_loss
VIS_EPOCHS = 5 
import soundfile as sf
import io


torch.autograd.set_detect_anomaly(True)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu") #CAIR

def softmax(x):
    """
    Compute the softmax of a single real number.
    
    :param x: Input scalar (real number).
    :return: Softmax value (scalar between 0 and 1).
    """
    return 1 / (1 + np.exp(-x))  # This is the logistic sigmoid function

# Function for calculating dynamic weight bsed on PESQ
def pesq_weighting_function(pesq_score, k=5, c=3.0):
    """
    Calculate the dynamic weight based on the PESQ score.
    
    :param pesq_score: The PESQ score of the current prediction.
    :param k: Steepness of the transition.
    :param c: Midpoint PESQ score where the weight starts to shift.
    :return: Dynamic weight based on PESQ score.
    """
    return 1 / (1 + np.exp(-k * (pesq_score - c)))

# Normalised SSNR
# Function to compute SSNR
def compute_ssnr(clean_tensor, enhanced_tensor, sample_rate=16000):
    """
    Computes the Normalized Segmental Signal-to-Noise Ratio (SSNR) for each sample in the batch.
    
    Parameters:
    - clean_tensor: PyTorch tensor of clean speech signals, shape [batch_size, 1, 256, 256]
    - enhanced_tensor: PyTorch tensor of enhanced speech signals, shape [batch_size, 1, 256, 256]
    - frame_length: Length of each frame (in samples)
    - frame_shift: Shift between consecutive frames (in samples)
    - sample_rate: Sample rate of the audio signals
    
    Returns:
    - normalized_ssnr_tensor: PyTorch tensor of normalized SSNR scores for each sample in the batch
    """
    frame_length=256 
    frame_shift=128
    # List to store SSNR scores for each sample
    ssnr_scores = []
    
    for i in range(clean_tensor.shape[0]):  # Iterate over batch dimension
        # Convert tensors to waveforms
        clean_speech = tensor_to_wav_buffer(clean_tensor[i, 0], sample_rate)
        enhanced_speech = tensor_to_wav_buffer(enhanced_tensor[i, 0], sample_rate)
        
####################################################################################
        # Load the waveform from the buffer
        clean_speech, _ = sf.read(clean_speech)
        enhanced_speech, _ = sf.read(enhanced_speech)
        # Modified to check if the error goes away
####################################################################################

        # Compute SSNR for the current sample
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
            
            # Calculate segmental SNR for the frame
            signal_energy = np.sum(clean_frame ** 2)
            noise_energy = np.sum(noise_frame ** 2)
            
            eps = 1e-8 # Given to remove the warning of RuntimeWarning
            if signal_energy > eps and noise_energy > eps:  
                snr_frame = 10 * np.log10(signal_energy / (noise_energy + eps))
                ssnr_list.append(snr_frame)

            #if noise_energy > 0:  # Avoid log of zero or negative
                #snr_frame = 10 * np.log10(signal_energy / (noise_energy + eps)) 
                #ssnr_list.append(snr_frame)
        
        # Calculate average SSNR over all frames for the current sample
        average_ssnr = np.mean(ssnr_list) if ssnr_list else 0.0
        
        # Normalize SSNR
        normalized_ssnr = (average_ssnr + 10) / 45  # Normalized to range [0, 1]
        ssnr_scores.append(normalized_ssnr)
    
    # Convert the list of SSNR scores to a PyTorch tensor
    normalized_ssnr_tensor = torch.tensor(ssnr_scores).to(clean_tensor.device)
    
    return normalized_ssnr_tensor


# Normalised signal
def normalize_signal(signal):
    """Normalize the signal to have zero mean and unit variance."""
    if isinstance(signal, torch.Tensor):
        signal = signal.detach().cpu().numpy()  # Detach from computation graph and move to CPU
    # Normalize using NumPy operations
    signal = signal - np.mean(signal)
    signal = signal / np.std(signal)
    return signal

# Function for calculating dynamic weight bsed on PESQ
def pesq_weighting_function(pesq_score, k=5, c=3.0):
    """
    Calculate the dynamic weight based on the PESQ score.
    
    :param pesq_score: The PESQ score of the current prediction.
    :param k: Steepness of the transition.
    :param c: Midpoint PESQ score where the weight starts to shift.
    :return: Dynamic weight based on PESQ score.
    """
    return 1 / (1 + np.exp(-k * (pesq_score - c)))

def dnsmos_eval(predict, target):
    """ Note target_wav is not used in the dnsmos function !!!
        Normalize the score to 0~1 for training.
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
            score = float(
                sigmoid(score_dict["mos"])
            )  # normalize the score to 0~1
            break
        except Exception as e:  # sometimes, access the dnsmos server too ofen may disable the service.
            print(e)
            time.sleep(10)  # wait for 10 secs
    return score


def sigmoid(x):
    return 1 / (1 + np.exp(-x))



def load_wav_from_buffer(buffer):
    """ Load WAV data from an in-memory buffer into a NumPy array. """
    buffer.seek(0)
    audio_np, _ = sf.read(buffer)
    return audio_np


def load_wav_from_buffer(buffer):
    """ Load WAV data from an in-memory buffer into a NumPy array. """
    buffer.seek(0)
    audio_np, _ = sf.read(buffer)
    return audio_np


def tensor_to_wav_buffer(audio_tensor, sample_rate=16000, n_fft=510, hop_length=256, win_length=510):
    """
    Convert a tensor representing STFT frames to an in-memory WAV file buffer.
    
    Args:
        audio_tensor (torch.Tensor): Tensor with shape [256, 256] representing STFT frames.
        sample_rate (int): Sampling rate of the audio.
        n_fft (int): Number of FFT components.
        hop_length (int): Number of samples between successive frames.
        win_length (int): Window length.
    
    Returns:
        io.BytesIO: In-memory buffer containing the WAV file.
    """
    # Convert tensor to NumPy array
    stft_np = audio_tensor.detach().cpu().numpy()

    # Ensure the array is 2D
    if stft_np.ndim != 2:
        raise ValueError("Input tensor must be 2D representing STFT frames")

    # Handle complex numbers if present in the STFT representation
    if np.iscomplexobj(stft_np):
        stft_np = np.real(stft_np)  # Use the real part, or np.abs(stft_np) for magnitude

    # Convert the STFT back to the time-domain signal using iSTFT
    _, audio_np = istft(stft_np, fs=sample_rate, nperseg=win_length, noverlap=hop_length)

    # Ensure the data type is supported
    if audio_np.dtype != np.float32:
        audio_np = audio_np.astype(np.float32)
    
    # Create an in-memory buffer
    buffer = io.BytesIO()
    
    # Write NumPy array to the buffer as WAV file
    sf.write(buffer, audio_np, sample_rate, format='WAV')
    
    # Seek to the beginning of the buffer
    buffer.seek(0)
    
    return buffer


def srmrpy_eval(predict, target):
    """ Evaluate SRMR score and normalize it. """
    
    def calculate_srmr_from_buffer(buffer):
        """ Calculate SRMR score from an in-memory WAV buffer. """
        try:
            # Load the WAV file from the buffer
            audio_np = load_wav_from_buffer(buffer)
            
            # Calculate SRMR score
            srmr_score = srmr(
                audio_np,
                fs=16000,
                n_cochlear_filters=23,
                low_freq=125,
                min_cf=4,
                max_cf=128,
                fast=True,
                norm=False,
            )[0]
        except ValueError as e:
            print(f"Error encountered during SRMR calculation: {e}")
            return None
        
        return srmr_score
    
    # Calculate SRMR for each sample in the batch
    srmr_scores = []
    for i in range(predict.shape[0]):  # Iterate over batch dimension
        predict_sample = predict[i, 0]  # Select batch item and channel
        
        # Convert tensor to an in-memory WAV buffer
        buffer = tensor_to_wav_buffer(predict_sample)
        
        # Calculate SRMR from the buffer
        srmr_score = calculate_srmr_from_buffer(buffer)
        
        if srmr_score is not None:
            srmr_scores.append(srmr_score)
    
    # Aggregate the scores if needed (e.g., mean of all samples)
    if srmr_scores:
        avg_srmr_score = np.mean(srmr_scores)
    else:
        avg_srmr_score = 0.0
    
    # Normalize the score using sigmoid and convert to float
    normalized_score = float(sigmoid(0.1 * avg_srmr_score))
    
    # Convert back to tensor if needed
    normalized_score_tensor = torch.tensor(normalized_score).to(predict.device)
    
    return normalized_score_tensor

def stoi_loss(clean_tensor, noisy_tensor, sr=16000):
    """Calculate STOI Loss for tensors."""
    batch_size = clean_tensor.shape[0]  # Assuming batch size is the first dimension
    stoi_scores = []

    for i in range(batch_size):
        # Convert each tensor in the batch to an in-memory WAV buffer
        clean_wav_buffer = tensor_to_wav_buffer(clean_tensor[i, 0], sample_rate=sr)
        noisy_wav_buffer = tensor_to_wav_buffer(noisy_tensor[i, 0], sample_rate=sr)

        # Load the waveform from the buffer
        clean_wav, _ = sf.read(clean_wav_buffer)
        noisy_wav, _ = sf.read(noisy_wav_buffer)

        try:
            # Compute STOI score
            stoi_score = stoi(clean_wav, noisy_wav, sr, extended=True)
        except Exception as e:
            # Handle errors if necessary
            print(f"Error computing STOI: {e}")
            stoi_score = 0  # STOI ranges from 0 to 1, so 0 is a logical default
        
        stoi_scores.append(stoi_score)

    # Calculate the average STOI score for the batch
    avg_stoi_score = np.mean(stoi_scores)
    
    return avg_stoi_score

'''
# Function for calculating PESQ Loss
def pesq_loss(clean, noisy, sr=16000):
    try:
        pesq_score = pesq(sr, clean, noisy, 'wb')
    except:
        # error can happen due to silent period
        pesq_score = -1
    return pesq_score
'''
def pesq_loss(clean_tensor, noisy_tensor, sr=16000):
    """Calculate PESQ Loss for tensors."""
    batch_size = clean_tensor.shape[0]  # Assuming batch size is the first dimension
    pesq_scores = []

    for i in range(batch_size):
        # Convert each tensor in the batch to an in-memory WAV buffer
        clean_wav_buffer = tensor_to_wav_buffer(clean_tensor[i, 0], sample_rate=sr)
        noisy_wav_buffer = tensor_to_wav_buffer(noisy_tensor[i, 0], sample_rate=sr)

        # Load the waveform from the buffer
        clean_wav, _ = sf.read(clean_wav_buffer)
        noisy_wav, _ = sf.read(noisy_wav_buffer)

        try:
            # Compute PESQ score
            pesq_score = pesq(sr, clean_wav, noisy_wav, 'wb')
        except Exception as e:
            # Handle errors, possibly due to silent periods or other issues
            #print(f"Error computing PESQ: {e}")
            pesq_score = -1
        
        pesq_scores.append(pesq_score)

    # Calculate the average PESQ score for the batch
    avg_pesq_score = np.mean(pesq_scores)
    
    return avg_pesq_score

# Batch Processing for PESQ Calculation
def batch_pesq(clean, noisy):

    pesq_score = Parallel(n_jobs=-1)(delayed(pesq_loss)(c, n) for c, n in zip(clean, noisy))
    pesq_score = np.array(pesq_score)
    if -1 in pesq_score:
        return None
    pesq_score = (pesq_score - 1) / 3.5
    return torch.FloatTensor(pesq_score).to('cuda')

def non_uniform_timesteps(self, x, p=3, q=3):
    # Step 1: Generate uniform samples in [0, 1]
    uniform_samples = torch.rand(x.shape[0], device=x.device)

    # Step 2: Apply the power transformation to get sparser samples near t_eps and finer near T
    # p controls the coarseness near t_eps, q controls the fineness near T
    if p != q:
        transformed_samples = uniform_samples ** (1/p) * (1 - uniform_samples ** (1/q))
    else:
        transformed_samples = uniform_samples ** p

    # Step 3: Scale to [t_eps, T]
    timesteps = self.t_eps + (self.sde.T - self.t_eps) * transformed_samples
    return timesteps


# Configure Logging
logging.basicConfig(filename='training_logs.txt', level=logging.INFO,
                    format='%(asctime)s:%(levelname)s:%(message)s')

class ScoreModel(pl.LightningModule):
    def __init__(self,
        backbone: str = "ncsnpp", sde: str = "ouvesde",
        lr: float = 1e-4, ema_decay: float = 0.999,
        t_eps: float = 3e-2, transform: str = 'none', nolog: bool = False, device=device,
        num_eval_files: int = 50, loss_type: str = 'mse', data_module_cls = None, **kwargs
    ):
        """
        Create a new ScoreModel.

        Args:
            backbone: The underlying backbone DNN that serves as a score-based model.
                Must have an output dimensionality equal to the input dimensionality.
            sde: The SDE to use for the diffusion.
            lr: The learning rate of the optimizer. (1e-4 by default).
            ema_decay: The decay constant of the parameter EMA (0.999 by default).
            t_eps: The minimum time to practically run for to avoid issues very close to zero (1e-5 by default).
            reduce_mean: If `True`, average the loss across data dimensions.
                Otherwise sum the loss across data dimensions.
        """
        super().__init__()
        # Initialize Backbone DNN
        dnn_cls = BackboneRegistry.get_by_name(backbone)
        kwargs.update(input_channels=4)
        self.dnn = dnn_cls(**kwargs)
        # Initialize SDE
        sde_cls = SDERegistry.get_by_name(sde)
        self.sde = sde_cls(**kwargs)
        # Store hyperparams and save them
        self.lr = lr
        self.ema_decay = ema_decay
        self.ema = ExponentialMovingAverage(self.parameters(), decay=self.ema_decay)
        self._error_loading_ema = False
        self.t_eps = t_eps
        self.loss_type = loss_type
        self.num_eval_files = num_eval_files

        self.save_hyperparameters(ignore=['nolog'])
        self.data_module = data_module_cls(**kwargs, gpu=kwargs.get('gpus', 0) > 0)
        self._reduce_op = lambda *args, **kwargs: 0.5 * torch.sum(*args, **kwargs)
        self.nolog = nolog

    @staticmethod
    def add_argparse_args(parser):
        parser.add_argument("--lr", type=float, default=1e-4, help="The learning rate")
        parser.add_argument("--ema_decay", type=float, default=0.999, help="The parameter EMA decay constant (0.999 by default)")
        parser.add_argument("--t_eps", type=float, default=0.03, help="The minimum time (3e-2 by default)")
        parser.add_argument("--num_eval_files", type=int, default=10, help="Number of files for speech enhancement performance evaluation during training.")
        parser.add_argument("--loss_type", type=str, default="mse", choices=("mse", "mae", "gaussian_entropy", "kristina", "sisdr", "time_mse"), help="The type of loss function to use.")
        parser.add_argument("--spatial_channels", type=int, default=1)
        return parser

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)
        return optimizer

    def optimizer_step(self, *args, **kwargs):
        # Method overridden so that the EMA params are updated after each optimizer step
        super().optimizer_step(*args, **kwargs)
        self.ema.update(self.parameters())

    # on_load_checkpoint / on_save_checkpoint needed for EMA storing/loading
    def on_load_checkpoint(self, checkpoint):
        ema = checkpoint.get('ema', None)
        if ema is not None:
            self.ema.load_state_dict(checkpoint['ema'])
        else:
            self._error_loading_ema = True
            warnings.warn("EMA state_dict not found in checkpoint!")

    def on_save_checkpoint(self, checkpoint):
        checkpoint['ema'] = self.ema.state_dict()

    def train(self, mode, no_ema=False):
        res = super().train(mode)  # call the standard `train` method with the given mode
        if not self._error_loading_ema:
            if mode == False and not no_ema:
                # eval
                self.ema.store(self.parameters())        # store current params in EMA
                self.ema.copy_to(self.parameters())      # copy EMA parameters over current params for evaluation
            else:
                # train
                if self.ema.collected_params is not None:
                    self.ema.restore(self.parameters())  # restore the EMA weights (if stored)
        return res

    def eval(self, no_ema=False):
        return self.train(False, no_ema=no_ema)

    def _loss(self, err, err_time=None, err_mag=None):
        if self.loss_type == 'mse':
            losses = torch.square(err.abs())
            loss = torch.mean(0.5*torch.sum(losses.reshape(losses.shape[0], -1), dim=-1))

        elif self.loss_type == 'mae':
            losses = err.abs()
            loss = torch.mean(0.5*torch.sum(losses.reshape(losses.shape[0], -1), dim=-1))

        return loss

    def _weighted_mean(self, x, w):
        return torch.mean(x * w)

    def _raw_dnn_output(self, x, t, y):
        dnn_input = torch.cat([x, y], dim=1) #b,2*d,f,t
        return self.dnn(dnn_input, t)

    def forward(self, x, t, y, **kwargs):
        score = -self._raw_dnn_output(x, t, y)
        std = self.sde._std(t, y=y)
        if std.ndim < y.ndim:
            std = std.view(*std.size(), *((1,)*(y.ndim - std.ndim)))
        return score

    def _step(self, batch, batch_idx):
        if len(batch) == 2:
            x, y = batch
        elif len(batch) == 3:
            assert "bwe" in self.data_module.task, "Received metadata for a task which is not BWE"
            x, y, scale_factors = batch
        t = torch.rand(x.shape[0], device=x.device) * (self.sde.T - self.t_eps) + self.t_eps
######################################################################################################################
        '''
        # Step 1: Generate uniform samples in [0, 1]
        uniform_samples = torch.rand(x.shape[0], device=x.device)
        
        # Step 2: Apply exponential transformation
        # Use the inverse CDF (quantile function) for the exponential distribution
        lambda_=2 #param lambda_: Rate parameter for the exponential distribution (larger lambda means more concentration towards T).
        transformed_samples = -torch.log(1 - uniform_samples) / lambda_
        
        # Normalize the samples to [0, 1] range
        normalized_samples = transformed_samples / (transformed_samples.max() + 1e-6)
        
        # Step 3: Scale to [t_eps, T]
        t = self.t_eps + (self.sde.T - self.t_eps) * normalized_samples
        '''
######################################################################################################################
        mean, std = self.sde.marginal_prob(x, t, y)
        z = torch.randn_like(x)  # i.i.d. normal distributed with var=0.5 ---> problem: this cannot work for FreqOUVE, because is standard, and tries to match a score with a sigma which is not standard
        if std.ndim < y.ndim:
            std = std.view(*std.size(), *((1,)*(y.ndim - std.ndim)))
        sigmas = std
        perturbed_data = mean + sigmas * z
        score = self(perturbed_data, t, y)
        err = score * sigmas + z
        loss = self._loss(err)
        return loss

    def training_step(self, batch, batch_idx):
        loss = self._step(batch, batch_idx)
        self.log('train_loss', loss, on_step=True, on_epoch=True, batch_size=self.data_module.batch_size)
        #logging.info(f"Train loss: {loss.item()} at step {batch_idx}") #CAIR
        return loss

    def validation_step(self, batch, batch_idx, discriminative=False, sr=16000):
        loss = self._step(batch, batch_idx)
        self.log('valid_loss', loss, on_step=False, on_epoch=True, batch_size=self.data_module.batch_size)
        #logging.info(f"Validation loss: {loss.item()} at step {batch_idx}") #CAIR

        # Evaluate speech enhancement performance
        if batch_idx == 0 and self.num_eval_files != 0:
            pesq_est, si_sdr_est, estoi_est, ssnr_est, srmr_est, spec, audio = evaluate_model(self, self.num_eval_files, spec=not self.current_epoch%VIS_EPOCHS, audio=not self.current_epoch%VIS_EPOCHS, discriminative=discriminative)
            print(f"PESQ at epoch {self.current_epoch} : {pesq_est:.2f}")
            print(f"SISDR at epoch {self.current_epoch} : {si_sdr_est:.1f}")
            print(f"ESTOI at epoch {self.current_epoch} : {estoi_est:.2f}")
            print(f"SRMR at epoch {self.current_epoch} : {srmr_est:.2f}")
            print(f"SSNR at epoch {self.current_epoch} : {ssnr_est:.2f}")
            print('__________________________________________________________________')
            
            self.log('ValidationPESQ', pesq_est, on_step=False, on_epoch=True) #CAIR
            self.log('ValidationSISDR', si_sdr_est, on_step=False, on_epoch=True) #CAIR
            self.log('ValidationESTOI', estoi_est, on_step=False, on_epoch=True) #CAIR
            self.log('ValidationSRMR', srmr_est, on_step=False, on_epoch=True) #CAIR
            self.log('ValidationSSNR', ssnr_est, on_step=False, on_epoch=True) # CAIR

            #logging.info(f"PESQ at epoch {self.current_epoch} : {pesq_est:.2f}")
            #logging.info(f"SISDR at epoch {self.current_epoch} : {si_sdr_est:.1f}")
            #logging.info(f"ESTOI at epoch {self.current_epoch} : {estoi_est:.2f}")

            if audio is not None:
                y_list, x_hat_list, x_list = audio
                
                for idx, (y, x_hat, x) in enumerate(zip(y_list, x_hat_list, x_list)):
                    if self.current_epoch == 0:
                        # Normalize y and add batch dimension
                        #audio_tensor = (y / torch.max(torch.abs(y))).unsqueeze(0)  # Add batch dimension
                        #print('y.shape = ', y.shape)
                        #print('audio_tensor.shape = ', audio_tensor.shape)

                        # Verify the tensor shape: should be (1, time) for single-channel audio
                        #if audio_tensor.shape[1] != 1:
                            #raise ValueError("The number of channels in the audio tensor should be 1 for single-channel audio.")

                        self.logger.experiment.add_audio(f"Epoch={self.current_epoch} Mix/{idx}", (y / torch.max(torch.abs(y))), global_step=None, sample_rate=sr) #CAIR
                        self.logger.experiment.add_audio(f"Epoch={self.current_epoch} Clean/{idx}", (x / torch.max(torch.abs(x))), global_step=None, sample_rate=sr)
                    self.logger.experiment.add_audio(f"Epoch={self.current_epoch} Estimate/{idx}", (x_hat / torch.max(torch.abs(x_hat))), global_step=None, sample_rate=sr)
            
            if spec is not None:
                figures = []
                y_stft_list, x_hat_stft_list, x_stft_list = spec
                for idx, (y_stft, x_hat_stft, x_stft) in enumerate(zip(y_stft_list, x_hat_stft_list, x_stft_list)):
                    figures.append(
                        visualize_example(
                        torch.abs(y_stft), 
                        torch.abs(x_hat_stft), 
                        torch.abs(x_stft), return_fig=True))
                self.logger.experiment.add_figure(f"Epoch={self.current_epoch}/Spec", figures)

        return loss

    def to(self, *args, **kwargs):
        self.ema.to(*args, **kwargs)
        return super().to(*args, **kwargs)

    def get_pc_sampler(self, predictor_name, corrector_name, y, N=None, minibatch=None, scale_factor=None, **kwargs):
        N = self.sde.N if N is None else N
        sde = self.sde.copy()
        sde.N = N

        kwargs = {"eps": self.t_eps, **kwargs}
        if minibatch is None:
            return sampling.get_pc_sampler(predictor_name, corrector_name, sde=sde, score_fn=self, y=y, **kwargs)
        else:
            M = y.shape[0]
            def batched_sampling_fn():
                samples, ns = [], []
                for i in range(int(ceil(M / minibatch))):
                    y_mini = y[i*minibatch:(i+1)*minibatch]
                    sampler = sampling.get_pc_sampler(predictor_name, corrector_name, sde=sde, score_fn=self, y=y_mini, **kwargs)
                    sample, n = sampler()
                    samples.append(sample)
                    ns.append(n)
                samples = torch.cat(samples, dim=0)
                return samples, ns
            return batched_sampling_fn

    def get_ode_sampler(self, y, N=None, minibatch=1, **kwargs):
        N = self.sde.N if N is None else N
        sde = self.sde.copy()
        sde.N = N

        kwargs = {"eps": self.t_eps, **kwargs}
        if minibatch is None:
            return sampling.get_ode_sampler(sde, self, y=y, **kwargs)
        else:
            M = y.shape[0]
            def batched_sampling_fn():
                samples, ns = [], []
                for i in range(int(ceil(M / minibatch))):
                    y_mini = y[i*minibatch:(i+1)*minibatch]
                    sampler = sampling.get_ode_sampler(sde, self, y=y_mini, **kwargs)
                    sample, n = sampler()
                    samples.append(sample)
                    ns.append(n)
                samples = torch.cat(samples, dim=0)
                return samples, ns
            return batched_sampling_fn

    def train_dataloader(self):
        return self.data_module.train_dataloader()

    def val_dataloader(self):
        return self.data_module.val_dataloader()

    def test_dataloader(self):
        return self.data_module.test_dataloader()

    def setup(self, stage=None):
        return self.data_module.setup(stage=stage)

    def to_audio(self, spec, length=None):
        return self._istft(self._backward_transform(spec), length)

    def _forward_transform(self, spec):
        return self.data_module.spec_fwd(spec)

    def _backward_transform(self, spec):
        return self.data_module.spec_back(spec)

    def _stft(self, sig):
        return self.data_module.stft(sig)

    def _istft(self, spec, length=None):
        return self.data_module.istft(spec, length)

    def enhance(self, y, sampler_type="pc", predictor="reverse_diffusion",
        corrector="ald", N=50, corrector_steps=1, snr=0.5, timeit=False,
        scale_factor = None, return_stft=False,
        **kwargs
    ):
        """
        One-call speech enhancement of noisy speech `y`, for convenience.
        """
        start = time.time()
        T_orig = y.size(1)
        norm_factor = y.abs().max().item()
        y = y / norm_factor
        #Y = torch.unsqueeze(self._forward_transform(self._stft(y.cuda())), 0)  
        Y = torch.unsqueeze(self._forward_transform(self._stft(y.to(device))), 0)  #CAIR
        #Y = torch.unsqueeze(self._forward_transform(self._stft(y.cpu())), 0)
        Y = pad_spec(Y)
        if sampler_type == "pc":
            sampler = self.get_pc_sampler(predictor, corrector, Y, N=N,
                corrector_steps=corrector_steps, snr=snr, intermediate=False,
                scale_factor=scale_factor,
                **kwargs)
        elif sampler_type == "ode":
            sampler = self.get_ode_sampler(Y, N=N, **kwargs)
        else:
            print("{} is not a valid sampler type!".format(sampler_type))
        sample, nfe = sampler()

        if return_stft:
            return sample.squeeze(), Y.squeeze(), T_orig, norm_factor

        x_hat = self.to_audio(sample.squeeze(), T_orig)
        x_hat = x_hat * norm_factor
        x_hat = x_hat.squeeze().cpu()
        end = time.time()
        if timeit:
            sr = 16000
            rtf = (end-start)/(len(x_hat)/sr)
            return x_hat, nfe, rtf
        else:
            return x_hat









class DiscriminativeModel(ScoreModel):

    def forward(self, y):
        if self.dnn.FORCE_STFT_OUT:
            y = self._istft(self._backward_transform(y.clone().squeeze(1)))
        t = torch.ones(y.shape[0], device=y.device)
        x_hat = self.dnn(y, t)
        return x_hat

    def _loss(self, x, xhat):
        if self.dnn.FORCE_STFT_OUT:
            x = self._istft(self._backward_transform(x.clone().squeeze(1)))

        if self.loss_type == 'mse':
            losses = torch.square((x - xhat).abs())
            loss = torch.mean(0.5*torch.sum(losses.reshape(losses.shape[0], -1), dim=-1))

        elif self.loss_type == 'mae':
            losses = (x - xhat).abs()
            loss = torch.mean(0.5*torch.sum(losses.reshape(losses.shape[0], -1), dim=-1))

        elif self.loss_type == "sisdr":
            loss = - torch.mean(torch.stack([si_sdr_torch(x[i], xhat[i]) for i in range(x.size(0))]))
        return loss

    def _step(self, batch, batch_idx):
        X, Y = batch
        Xhat = self(Y)
        loss = self._loss(X, Xhat)
        return loss

    def enhance(self, y, **ignored_kwargs):
        with torch.no_grad():
            norm_factor = y.abs().max().item()
            T_orig = y.size(1)

            if self.data_module.return_time:
                #Y = torch.unsqueeze((y/norm_factor).cuda(), 0) #1,D=1,T
                #Y = torch.unsqueeze((y/norm_factor).cpu(), 0) #1,D=1,T 
                Y = torch.unsqueeze((y/norm_factor).to(device), 0) #1,D=1,T   #CAIR
            else:
                #Y = torch.unsqueeze(self._forward_transform(self._stft((y/norm_factor).cuda())), 0) #1,D,F,T
                #Y = torch.unsqueeze(self._forward_transform(self._stft((y/norm_factor).cpu())), 0) #1,D,F,T
                Y = torch.unsqueeze(self._forward_transform(self._stft((y/norm_factor).to(device))), 0) #1,D,F,T y. #CAIR
                Y = pad_spec(Y)
            X_hat = self(Y)
            if self.dnn.FORCE_STFT_OUT:
                X_hat = self._forward_transform(self._stft(X_hat)).unsqueeze(1)

            if self.data_module.return_time:
                x_hat = X_hat.squeeze()
            else:
                x_hat = self.to_audio(X_hat.squeeze(), T_orig)

            return (x_hat * norm_factor).squeeze()
                    
    def validation_step(self, batch, batch_idx):
        return super().validation_step(batch, batch_idx, discriminative=True)
    

















class StochasticRegenerationModel(pl.LightningModule):
    def __init__(self,
        backbone_denoiser: str, backbone_score: str, sde: str,
        lr: float = 1e-4, ema_decay: float = 0.999,
        t_eps: float = 3e-2, nolog: bool = False, num_eval_files: int = 50,
        loss_type_denoiser: str = "none", loss_type_score: str = 'mse', data_module_cls = None, 
        mode = "regen-joint-training", condition = "both",
        **kwargs
    ):
        """
        Create a new ScoreModel.

        Args:
            backbone: The underlying backbone DNN that serves as a score-based model.
                Must have an output dimensionality equal to the input dimensionality.
            sde: The SDE to use for the diffusion.
            lr: The learning rate of the optimizer. (1e-4 by default).
            ema_decay: The decay constant of the parameter EMA (0.999 by default).
            t_eps: The minimum time to practically run for to avoid issues very close to zero (1e-5 by default).
            reduce_mean: If `True`, average the loss across data dimensions.
                Otherwise sum the loss across data dimensions.
        """
        super().__init__()
        # Initialize Backbone DNN
        kwargs_denoiser = kwargs
        kwargs_denoiser.update(input_channels=2)
        kwargs_denoiser.update(discriminative=True)
        self.denoiser_net = BackboneRegistry.get_by_name(backbone_denoiser)(**kwargs) if backbone_denoiser != "none" else None

        kwargs.update(input_channels=(6 if condition == "both" else 4))
        kwargs_denoiser.update(discriminative=False)
        self.score_net = BackboneRegistry.get_by_name(backbone_score)(**kwargs) if backbone_score != "none" else None

        # Initialize SDE
        sde_cls = SDERegistry.get_by_name(sde)
        self.sde = sde_cls(**kwargs)
        self.t_eps = t_eps

        # Store hyperparams and save them
        self.lr = lr
        self.ema_decay = ema_decay
        self.ema = ExponentialMovingAverage(self.parameters(), decay=self.ema_decay)
        self._error_loading_ema = False

        self.loss_type_denoiser = loss_type_denoiser
        self.loss_type_score = loss_type_score
        if "weighting_denoiser_to_score" in kwargs.keys():
            self.weighting_denoiser_to_score = kwargs["weighting_denoiser_to_score"]
        else:
            self.weighting_denoiser_to_score = .5
        self.condition = condition
        self.mode = mode
        self.configure_losses()

        self.num_eval_files = num_eval_files
        self.save_hyperparameters(ignore=['nolog'])
        self.data_module = data_module_cls(**kwargs, gpu=kwargs.get('gpus', 0) > 0)
        self._reduce_op = lambda *args, **kwargs: 0.5 * torch.sum(*args, **kwargs)
        self.nolog = nolog

    @staticmethod
    def add_argparse_args(parser):
        parser.add_argument("--lr", type=float, default=1e-4, help="The learning rate")
        parser.add_argument("--ema_decay", type=float, default=0.999, help="The parameter EMA decay constant (0.999 by default)")
        parser.add_argument("--t_eps", type=float, default=0.03, help="The minimum time (3e-2 by default)")
        parser.add_argument("--num_eval_files", type=int, default=10, help="Number of files for speech enhancement performance evaluation during training.")
        parser.add_argument("--loss_type_denoiser", type=str, default="mse", choices=("none", "mse", "mae", "sisdr", "mse_cplx+mag", "mse_time+mag"), help="The type of loss function to use.")
        parser.add_argument("--loss_type_score", type=str, default="mse", choices=("none", "mse", "mae"), help="The type of loss function to use.")
        parser.add_argument("--weighting_denoiser_to_score", type=float, default=0.5, help="a, as in L = a * L_denoiser + (1-a) * .")
        parser.add_argument("--condition", default="both", choices=["noisy", "post_denoiser", "both"])
        parser.add_argument("--spatial_channels", type=int, default=1)
        return parser

    def configure_losses(self):
        # Score Loss
        if self.loss_type_score == "mse":
            self.loss_fn_score = lambda err: self._reduce_op(torch.square(torch.abs(err)))
        elif self.loss_type_score == "mae":
            self.loss_fn_score = lambda err: self._reduce_op(torch.abs(err))
        elif self.loss_type_score == "none":
            raise NotImplementedError
            self.loss_fn_score = None
        else:
            raise NotImplementedError
        
        # Denoiser Loss
        if self.loss_type_denoiser == "mse":
            self.loss_fn_denoiser = lambda x, y: self._reduce_op(torch.square(torch.abs(x - y)))
        elif self.loss_type_denoiser == "mae":
            self.loss_fn_denoiser = lambda x, y: self._reduce_op(torch.abs(x - y))
        elif self.loss_type_denoiser == "none":
            self.loss_fn_denoiser = None
        else:
            raise NotImplementedError

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)
        return optimizer

    def optimizer_step(self, *args, **kwargs):
        # Method overridden so that the EMA params are updated after each optimizer step
        super().optimizer_step(*args, **kwargs)
        self.ema.update(self.parameters())

    def load_denoiser_model(self, checkpoint):
        self.denoiser_net = DiscriminativeModel.load_from_checkpoint(checkpoint).dnn
        if self.mode == "regen-freeze-denoiser":
            for param in self.denoiser_net.parameters():
                param.requires_grad = False

    def load_score_model(self, checkpoint):
        self.score_net = ScoreModel.load_from_checkpoint(checkpoint).dnn

    # on_load_checkpoint / on_save_checkpoint needed for EMA storing/loading
    def on_load_checkpoint(self, checkpoint):
        ema = checkpoint.get('ema', None)
        if ema is not None:
            self.ema.load_state_dict(checkpoint['ema'])
        else:
            self._error_loading_ema = True
            warnings.warn("EMA state_dict not found in checkpoint!")

    def on_save_checkpoint(self, checkpoint):
        checkpoint['ema'] = self.ema.state_dict()

    def train(self, mode, no_ema=False):
        res = super().train(mode)  # call the standard `train` method with the given mode
        if not self._error_loading_ema:
            if mode == False and not no_ema:
                # eval
                self.ema.store(self.parameters())        # store current params in EMA
                self.ema.copy_to(self.parameters())      # copy EMA parameters over current params for evaluation
            else:
                # train
                if self.ema.collected_params is not None:
                    self.ema.restore(self.parameters())  # restore the EMA weights (if stored)
        return res

    def eval(self, no_ema=False):
        return self.train(False, no_ema=no_ema)

    def _loss(self, err, y_denoised, x):
        loss_score = self.loss_fn_score(err) if self.loss_type_score != "none" else None
        loss_denoiser = self.loss_fn_denoiser(y_denoised, x) if self.loss_type_denoiser != "none" else None
	#PESQ Calculation
        #print(f"Shape of x: {x.shape}")
        #print(f"Shape of y_denoised: {y_denoised.shape}")	
        # PESQ Calculation
        #pesq_score = pesq_loss(x, y_denoised)
        #pesq_ls = (pesq_score - 1) / 3.5
        #print("pesq_ls=", pesq_ls)
        '''
        # Compute STOI (normalized by nature of the input normalization)
        stoi_score = stoi_loss(x, y_denoised, sr=16000)
   
        # Compute normalized SSNR
        ssnr_score = compute_ssnr(x, y_denoised, sample_rate=16000)
        ssnr_score = torch.mean(ssnr_score)

        # Compute normalized srmr
        srmr_score=srmrpy_eval(y_denoised, x)
        '''
        pesq_score = pesq_loss(x, y_denoised)
        pesq_ls = (pesq_score - 1) / 3.5

      
        if loss_score is not None:
            if loss_denoiser is not None:

###########################################################################################################################################

                #loss = loss_denoiser + loss_score
                #loss = (loss_denoiser + loss_score) * ((1 - pesq_ls) ** 2)* ((1 - stoi_score) ** 2)* ((1 - normalized_ssnr) ** 2)
                
                #print(f"x shape: {x.shape}")
                #print(f"y_denoised shape: {y_denoised.shape}")
                #print("pesq", type(pesq_ls))
                #print("srmr", type(srmr_score))
                
                #loss = (loss_denoiser + loss_score) #*((1 - srmr_score) ** 2)
                sc_loss, mag_loss = compute_mean_stft_loss(y_denoised, x)
                
                stft_loss = sc_loss + mag_loss
                #weight=pesq_weighting_function(pesq_score)
                
                weight = torch.sigmoid(stft_loss)
                #print("sigmoid(weigth_stft_loss)=", weight)
                
                #loss = (loss_denoiser + loss_score) + (0.1*((sc_loss))**1) + (0.1*((mag_loss))**1)
                loss = (weight)*(loss_denoiser + loss_score)

###########################################################################################################################################

            else:
                loss = loss_score
        else:
            loss = loss_denoiser
        return loss, loss_score, loss_denoiser

    def _weighted_mean(self, x, w):
        return torch.mean(x * w)

    def forward_score(self, x, t, score_conditioning, sde_input, **kwargs):
        # Move them all to the same device (CAIR)
        device = x.device #CAIR
        sde_input = sde_input.to(device) #CAIR
        score_conditioning = [sc.to(device) for sc in score_conditioning] #CAIR
        dnn_input = torch.cat([x] + score_conditioning, dim=1) #b,n_input*d,f,t
        score = -self.score_net(dnn_input, t)
        t = t.to(device) #CAIR
        std = self.sde._std(t, y=sde_input)
        if std.ndim < sde_input.ndim:
            std = std.view(*std.size(), *((1,)*(sde_input.ndim - std.ndim)))
        return score

    def forward_denoiser(self, y, **kwargs):
        y = y.cuda()
        x_hat = self.denoiser_net(y)
        #print(" m ready to return") #CAIR
        return x_hat

    def _step(self, batch, batch_idx):
        x, y = batch
        #print('X.shape = ', x.shape) #CAIR
        #print('Y.shape = ', y.shape) #CAIR

        # Denoising step
        with torch.set_grad_enabled(self.mode != "regen-freeze-denoiser"):
            y_denoised = self.forward_denoiser(y)

        # Score step

        sde_target = x
        sde_input = y_denoised

        #print('y_denoised.shape = ', sde_input.shape)
        #print('x.shape = ',x.shape)
        # Forward process
        t = torch.rand(x.shape[0], device=x.device) * (self.sde.T - self.t_eps) + self.t_eps
######################################################################################################################
        '''
        # Step 1: Generate uniform samples in [0, 1]
        uniform_samples = torch.rand(x.shape[0], device=x.device)
        
        # Step 2: Apply exponential transformation
        # Use the inverse CDF (quantile function) for the exponential distribution
        lambda_=2 #param lambda_: Rate parameter for the exponential distribution (larger lambda means more concentration towards T).
        transformed_samples = -torch.log(1 - uniform_samples) / lambda_
        
        # Normalize the samples to [0, 1] range
        normalized_samples = transformed_samples / (transformed_samples.max() + 1e-6)
        
        # Step 3: Scale to [t_eps, T]
        t = self.t_eps + (self.sde.T - self.t_eps) * normalized_samples
        '''
######################################################################################################################
        #t = non_uniform_timesteps(self, sde_target, p=3, q=3)
        #print('self.sde.T = ', self.sde.T) #CAIR
        #print('self.t_eps = ', self.t_eps) #CAIR
        #print('t.shape = ', t.shape) #CAIR
        mean, std = self.sde.marginal_prob(sde_target, t, sde_input)
        #print('Mean.shape = ', mean.shape) #CAIR
        #print('STD = ', std) #CAIR
        z = torch.randn_like(x)  # i.i.d. normal distributed with var=0.5
        if std.ndim < y.ndim:
            std = std.view(*std.size(), *((1,)*(y.ndim - std.ndim)))
        sigmas = std
        perturbed_data = mean + sigmas * z

        # Score estimation
        if self.condition == "noisy":
            score_conditioning = [y]
        elif self.condition == "post_denoiser":
            score_conditioning = [y_denoised]
        elif self.condition == "both":
            score_conditioning = [y, y_denoised]
        else:
            raise NotImplementedError(f"Don't know the conditioning you have wished for: {self.condition}")

        score = self.forward_score(perturbed_data, t, score_conditioning, sde_input)
        err = score * sigmas + z

        loss, loss_score, loss_denoiser = self._loss(err, y_denoised, x)

        return loss, loss_score, loss_denoiser

    def training_step(self, batch, batch_idx):
        loss, loss_score, loss_denoiser = self._step(batch, batch_idx)
        self.log('train_loss', loss, on_step=True, on_epoch=True, batch_size=self.data_module.batch_size)
        self.log('train_loss_score', loss_score, on_step=True, on_epoch=True, batch_size=self.data_module.batch_size)
        if loss_denoiser is not None:
            self.log('train_loss_denoiser', loss_denoiser, on_step=True, on_epoch=True, batch_size=self.data_module.batch_size)
        return loss

    def validation_step(self, batch, batch_idx, discriminative=False, sr=16000):
        loss, loss_score, loss_denoiser = self._step(batch, batch_idx)
        self.log('valid_loss', loss, on_step=False, on_epoch=True, batch_size=self.data_module.batch_size)
        self.log('valid_loss_score', loss_score, on_step=False, on_epoch=True, batch_size=self.data_module.batch_size)
        if loss_denoiser is not None:
            self.log('valid_loss_denoiser', loss_denoiser, on_step=False, on_epoch=True, batch_size=self.data_module.batch_size)

        # Evaluate speech enhancement performance
        if batch_idx == 0 and self.num_eval_files != 0:
            pesq_est, si_sdr_est, estoi_est, ssnr_est, srmr_est, dnsmos_est, spec, audio = evaluate_model(self, self.num_eval_files, spec=not self.current_epoch%VIS_EPOCHS, audio=not self.current_epoch%VIS_EPOCHS, discriminative=discriminative)
            #srmrpy_eval_valid
            print(f"PESQ at epoch {self.current_epoch} : {pesq_est:.2f}")
            print(f"SISDR at epoch {self.current_epoch} : {si_sdr_est:.1f}")
            print(f"ESTOI at epoch {self.current_epoch} : {estoi_est:.2f}")
            print(f"SRMR at epoch {self.current_epoch} : {srmr_est:.2f}")
            print(f"SSNR at epoch {self.current_epoch} : {ssnr_est:.2f}")
            print('__________________________________________________________________')
            
            self.log('ValidationPESQ', pesq_est, on_step=False, on_epoch=True) #CAIR
            self.log('ValidationSISDR', si_sdr_est, on_step=False, on_epoch=True) #CAIR
            self.log('ValidationESTOI', estoi_est, on_step=False, on_epoch=True) #CAIR
            self.log('ValidationSRMR', srmr_est, on_step=False, on_epoch=True) #CAIR
            self.log('ValidationSSNR', ssnr_est, on_step=False, on_epoch=True) # CAIR
            #logging.info(f"Validation PESQ at epoch {self.current_epoch} : {pesq_est:.2f}")
            #logging.info(f"Validation SISDR at epoch {self.current_epoch} : {si_sdr_est:.1f}")
            #logging.info(f"Validation ESTOI at epoch {self.current_epoch} : {estoi_est:.2f}")

            if audio is not None:
                y_list, x_hat_list, x_list = audio
                for idx, (y, x_hat, x) in enumerate(zip(y_list, x_hat_list, x_list)):
                    if self.current_epoch == 0:
                        y = (y / torch.max(torch.abs(y)))
                        x = (x / torch.max(torch.abs(x)))
                        x_hat = (x_hat / torch.max(torch.abs(x_hat)))

                        # Ensure the tensor has shape [1, length]
                        #y = y.unsqueeze(0) if y.dim() == 1 else y
                        #x_hat = x_hat.unsqueeze(0) if x_hat.dim() == 1 else x_hat
                        #x = x.unsqueeze(0) if x.dim() == 1 else x

                        # Printing Unsqueeze values
                        #print('Y.unsqueeze.shape = ', y.unsqueeze(0).shape)
                        #print('X.unsqueeze = ', x.unsqueeze(0))
                        #print('X_hat.unsqueeze = ', x_hat.unsqueeze(0))

                        self.logger.experiment.add_audio(f"Epoch={self.current_epoch} Mix/{idx}", y, global_step=None, sample_rate=sr)
                        self.logger.experiment.add_audio(f"Epoch={self.current_epoch} Clean/{idx}", x, global_step=None, sample_rate=sr)
                    self.logger.experiment.add_audio(f"Epoch={self.current_epoch} Estimate/{idx}", x_hat, global_step=None, sample_rate=sr)

            if spec is not None:
                figures = []
                y_stft_list, x_hat_stft_list, x_stft_list = spec
                for idx, (y_stft, x_hat_stft, x_stft) in enumerate(zip(y_stft_list, x_hat_stft_list, x_stft_list)):
                    figures.append(
                        visualize_example(
                        torch.abs(y_stft), 
                        torch.abs(x_hat_stft), 
                        torch.abs(x_stft), return_fig=True))
                self.logger.experiment.add_figure(f"Epoch={self.current_epoch}/Spec", figures)

        return loss

    def to(self, *args, **kwargs):
        self.ema.to(*args, **kwargs)
        return super().to(*args, **kwargs)

    def get_pc_sampler(self, predictor_name, corrector_name, y, N=None, minibatch=None, scale_factor=None, conditioning=None, **kwargs):
        N = self.sde.N if N is None else N
        sde = self.sde.copy()
        sde.N = N

        kwargs = {"eps": self.t_eps, **kwargs}
        if minibatch is None:
            return sampling.get_pc_sampler(predictor_name, corrector_name, sde=sde, score_fn=self.forward_score, y=y, conditioning=conditioning, **kwargs)
        else:
            M = y.shape[0]
            def batched_sampling_fn():
                samples, ns = [], []
                for i in range(int(ceil(M / minibatch))):
                    y_mini = y[i*minibatch:(i+1)*minibatch]
                    sampler = sampling.get_pc_sampler(predictor_name, corrector_name, sde=sde, score_fn=self.forward_score, y=y_mini, conditioning=conditioning, **kwargs)
                    sample, n = sampler()
                    samples.append(sample)
                    ns.append(n)
                samples = torch.cat(samples, dim=0)
                return samples, ns
            return batched_sampling_fn

    def get_ode_sampler(self, y, N=None, minibatch=1, **kwargs):
        N = self.sde.N if N is None else N
        sde = self.sde.copy()
        sde.N = N

        kwargs = {"eps": self.t_eps, **kwargs}
        if minibatch is None:
            return sampling.get_ode_sampler(sde, self, y=y, **kwargs)
        else:
            M = y.shape[0]
            def batched_sampling_fn():
                samples, ns = [], []
                for i in range(int(ceil(M / minibatch))):
                    y_mini = y[i*minibatch:(i+1)*minibatch]
                    sampler = sampling.get_ode_sampler(sde, self, y=y_mini, **kwargs)
                    sample, n = sampler()
                    samples.append(sample)
                    ns.append(n)
                samples = torch.cat(samples, dim=0)
                return samples, ns
            return batched_sampling_fn

    def train_dataloader(self):
        return self.data_module.train_dataloader()

    def val_dataloader(self):
        return self.data_module.val_dataloader()

    def test_dataloader(self):
        return self.data_module.test_dataloader()

    def setup(self, stage=None):
        return self.data_module.setup(stage=stage)

    def to_audio(self, spec, length=None):
        return self._istft(self._backward_transform(spec), length)

    def _forward_transform(self, spec):
        return self.data_module.spec_fwd(spec)

    def _backward_transform(self, spec):
        return self.data_module.spec_back(spec)

    def _stft(self, sig):
        return self.data_module.stft(sig)

    def _istft(self, spec, length=None):
        return self.data_module.istft(spec, length)

    def enhance(self, y, sampler_type="pc", predictor="reverse_diffusion",
        corrector="none", N=30, corrector_steps=1, snr=0.5, timeit=False,
        scale_factor = None, return_stft=False, denoiser_only=False,
        device=None, **kwargs
    ):


        if device is None:
          device = torch.device("cpu")
    
        # Move input tensor y to the specified device
        y = y.to(device)
        
        """
        One-call speech enhancement of noisy speech `y`, for convenience.
        """
        
        start = time.time()
        T_orig = y.size(1)
        norm_factor = y.abs().max().item()
        y = y / norm_factor
        #Y = torch.unsqueeze(self._forward_transform(self._stft(y.cuda())), 0)
        #Y = torch.unsqueeze(self._forward_transform(self._stft(y.cpu())), 0) 
        Y = torch.unsqueeze(self._forward_transform(self._stft(y.to(device))), 0)    #CAIR
        Y = pad_spec(Y)
        with torch.no_grad():
            
            if self.denoiser_net is not None:
                Y_denoised = self.forward_denoiser(Y)
            else:
                Y_denoised = None

            if self.score_net is not None and not denoiser_only:
                # Conditioning
                if self.condition == "noisy":
                    score_conditioning = [Y]
                elif self.condition == "post_denoiser":
                    score_conditioning = [Y_denoised]
                elif self.condition == "both":
                    score_conditioning = [Y, Y_denoised]
                else:
                    raise NotImplementedError(f"Don't know the conditioning you have wished for: {self.condition}")

                # Reverse process
                if sampler_type == "pc":
                    sampler = self.get_pc_sampler(predictor, corrector, Y_denoised, N=N,
                        corrector_steps=corrector_steps, snr=snr, intermediate=False,
                        scale_factor=scale_factor, conditioning=score_conditioning,
                        **kwargs)
                elif sampler_type == "ode":
                    sampler = self.get_ode_sampler(Y_denoised, N=N, 
                        conditioning=score_conditioning, 
                        **kwargs)
                else:
                    print("{} is not a valid sampler type!".format(sampler_type))
                sample, nfe = sampler()
            else:
                sample = Y_denoised

            if return_stft:
                return sample.squeeze(), Y.squeeze(), T_orig, norm_factor

        x_hat = self.to_audio(sample.squeeze(), T_orig)
        x_hat = x_hat * norm_factor
        x_hat = x_hat.squeeze().cpu()
        end = time.time()
        if timeit:
            sr = 16000
            rtf = (end-start)/(len(x_hat)/sr)
            return x_hat, nfe, rtf
        else:
            return x_hat

