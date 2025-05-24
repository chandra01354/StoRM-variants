import torch
import torch.nn.functional as F


def SCLoss(x_mag, y_mag):
    """Calculate forward propagation.
    Args:
        x_mag (Tensor): Magnitude spectrogram of predicted signal (B, #frames, #freq_bins).
        y_mag (Tensor): Magnitude spectrogram of groundtruth signal (B, #frames, #freq_bins).
    Returns:
        Tensor: Spectral convergence loss value.
        
    """
    return torch.norm(y_mag - x_mag, p="fro") / (torch.norm(y_mag, p="fro") + 1e-9)


def LogSTFTMagLoss(x_mag, y_mag):
    """Calculate forward propagation.
    Args:
        x_mag (Tensor): Magnitude spectrogram of predicted signal (B, #frames, #freq_bins).
        y_mag (Tensor): Magnitude spectrogram of groundtruth signal (B, #frames, #freq_bins).
    Returns:
        Tensor: Log STFT magnitude loss value.
    """
    return F.l1_loss(torch.log(y_mag.squeeze(0) + 1e-9), torch.log(x_mag.squeeze(0) + 1e-9))

# def LogSTFTMagLoss(x_mag, y_mag, eps=torch.finfo(torch.float32).eps):
#     """Calculate forward propagation.
#     Args:
#         x_mag (Tensor): Magnitude spectrogram of predicted signal (B, #frames, #freq_bins).
#         y_mag (Tensor): Magnitude spectrogram of groundtruth signal (B, #frames, #freq_bins).
#     Returns:
#         Tensor: Log STFT magnitude loss value.
#     """
#     # Clamp magnitudes after adding eps to ensure all values are positive
#     x_mag = torch.clamp(x_mag + eps, min=eps)
#     y_mag = torch.clamp(y_mag + eps, min=eps)
    
#     # Calculate the log magnitude loss
#     return F.l1_loss(torch.log(y_mag), torch.log(x_mag))


def STFTLoss(x, y):
    sc_loss = SCLoss(x, y)
    mag_loss = LogSTFTMagLoss(x, y)

    return sc_loss, mag_loss

def compute_mean_stft_loss(x_hat, x):
    """
    Compute the mean STFT loss for a batch of inputs.

    Args:
        x_hat (torch.Tensor): Predicted batch tensor (B, 1, H, W).
        x (torch.Tensor): Ground truth batch tensor (B, 1, H, W).

    Returns:
        tuple: Mean spectral convergence loss and mean magnitude loss.
    """
    # Initialize variables to accumulate the losses
    total_sc_loss = 0.0
    total_mag_loss = 0.0

    # Loop through each item in the batch
    for i in range(x_hat.size(0)):
        sc_loss, mag_loss = STFTLoss(x_hat[i].squeeze(0), x[i].squeeze(0))
        total_sc_loss += sc_loss
        total_mag_loss += mag_loss

    # Compute the mean of the losses
    mean_sc_loss = total_sc_loss / x_hat.size(0)
    mean_mag_loss = total_mag_loss / x_hat.size(0)

    return mean_sc_loss, mean_mag_loss
