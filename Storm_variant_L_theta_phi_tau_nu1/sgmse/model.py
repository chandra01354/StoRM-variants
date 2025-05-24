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

from pesq import pesq
from joblib import Parallel, delayed

from sgmse import sampling
from sgmse.sdes import SDERegistry
from sgmse.backbones import BackboneRegistry
from sgmse.util.inference import evaluate_model
from sgmse.util.graphics import visualize_example, visualize_one
from sgmse.util.other import pad_spec, si_sdr_torch
VIS_EPOCHS = 5 

torch.autograd.set_detect_anomaly(True)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu") #CAIR

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

# Function for calculating PESQ Loss
def pesq_loss(clean, noisy, sr=16000):
    try:
        pesq_score = pesq(sr, clean, noisy, 'wb')
    except:
        # error can happen due to silent period
        pesq_score = -1
    return pesq_score

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
        #t = torch.rand(x.shape[0], device=x.device) * (self.sde.T - self.t_eps) + self.t_eps
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
        lambda_param = 5.0  # Controls the skewness of the exponential distribution
        # Generate uniform samples in the range (t_eps, sde_T)
        t = torch.rand(x.shape[0], device=x.device) * (self.sde.T - self.t_eps) + self.t_eps

        # Normalize uniform samples to range (0, 1) before applying the exponential transformation
        t_normalized = (t - self.t_eps) / (self.sde.T - self.t_eps)

        # Transform to exponential distribution skewed towards 1
        exponential_samples = 1 - torch.exp(-lambda_param * t_normalized)

        # Rescale back to the original range (t_eps, sde_T)
        t = exponential_samples * (self.sde.T - self.t_eps) + self.t_eps
        
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
            pesq_est, si_sdr_est, estoi_est, spec, audio = evaluate_model(self, self.num_eval_files, spec=not self.current_epoch%VIS_EPOCHS, audio=not self.current_epoch%VIS_EPOCHS, discriminative=discriminative)
            print(f"PESQ at epoch {self.current_epoch} : {pesq_est:.2f}")
            print(f"SISDR at epoch {self.current_epoch} : {si_sdr_est:.1f}")
            print(f"ESTOI at epoch {self.current_epoch} : {estoi_est:.2f}")
            print(f"Loss at epoch {self.current_epoch} : {loss:.3f}")
            print('__________________________________________________________________')
            
            self.log('ValidationPESQ', pesq_est, on_step=False, on_epoch=True) #CAIR
            self.log('ValidationSISDR', si_sdr_est, on_step=False, on_epoch=True) #CAIR
            self.log('ValidationESTOI', estoi_est, on_step=False, on_epoch=True) #CAIR

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

        # PESQ Calculation
        pesq_score = pesq_loss(x, y_denoised)
        pesq_ls = (pesq_score - 1) / 3.5

        if loss_score is not None:
            if loss_denoiser is not None:
                #loss = self.weighting_denoiser_to_score * loss_denoiser + (1 - self.weighting_denoiser_to_score) * loss_score
                #loss = 0.9 * (self.weighting_denoiser_to_score * loss_denoiser + (1 - self.weighting_denoiser_to_score) * loss_score) + 0.1 * pesq_ls
                #loss = 1 * (self.weighting_denoiser_to_score * loss_denoiser + (1 - self.weighting_denoiser_to_score) * loss_score) + 0.1 * pesq_ls
                #loss = 1 * (self.weighting_denoiser_to_score * loss_denoiser + (1 - self.weighting_denoiser_to_score) * loss_score) + (0.1 * (1 - pesq_ls))
                #loss = 1 * (self.weighting_denoiser_to_score * loss_denoiser + (1 - self.weighting_denoiser_to_score) * loss_score) + 0.5 * pesq_ls
                #loss = 1 * (self.weighting_denoiser_to_score * loss_denoiser + (1 - self.weighting_denoiser_to_score) * loss_score) + ((1 - pesq_ls) ** 2)
                #loss = (loss_denoiser + loss_score) * ((1 - pesq_ls) ** 2)
                loss = loss_denoiser + loss_score
                #weight = pesq_weighting_function(pesq_score, k=5, c=3.0)
                #loss = weight * loss - (1-weight) * ((pesq_ls) ** 2)   #may be we can take square weight**2, (1-weight)**2
                #loss = (loss) / ((pesq_ls) ** 2)   #may be we can take square weight**2, (1-weight)**2
                #epsilon = 1e-9  # Small constant to prevent log of zero
                #loss = torch.log(loss + epsilon)
                #loss = loss*weight
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

        #print('y_denoised.shape = ',sde_input.shape)
        # Forward process
        #t = torch.rand(x.shape[0], device=x.device) * (self.sde.T - self.t_eps) + self.t_eps
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
        lambda_param = 5.0  # Controls the skewness of the exponential distribution
        # Generate uniform samples in the range (t_eps, sde_T)
        t = torch.rand(x.shape[0], device=x.device) * (self.sde.T - self.t_eps) + self.t_eps

        # Normalize uniform samples to range (0, 1) before applying the exponential transformation
        t_normalized = (t - self.t_eps) / (self.sde.T - self.t_eps)

        # Transform to exponential distribution skewed towards 1
        exponential_samples = 1 - torch.exp(-lambda_param * t_normalized)

        # Rescale back to the original range (t_eps, sde_T)
        t = exponential_samples * (self.sde.T - self.t_eps) + self.t_eps
        
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
            pesq_est, si_sdr_est, estoi_est, spec, audio = evaluate_model(self, self.num_eval_files, spec=not self.current_epoch%VIS_EPOCHS, audio=not self.current_epoch%VIS_EPOCHS, discriminative=discriminative)
            print(f"PESQ at epoch {self.current_epoch} : {pesq_est:.2f}")
            print(f"SISDR at epoch {self.current_epoch} : {si_sdr_est:.1f}")
            print(f"ESTOI at epoch {self.current_epoch} : {estoi_est:.2f}")
            print(f"Loss at epoch {self.current_epoch} : {loss:.3f}")
            print('__________________________________________________________________')
            
            self.log('ValidationPESQ', pesq_est, on_step=False, on_epoch=True) #CAIR
            self.log('ValidationSISDR', si_sdr_est, on_step=False, on_epoch=True) #CAIR
            self.log('ValidationESTOI', estoi_est, on_step=False, on_epoch=True) #CAIR

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

