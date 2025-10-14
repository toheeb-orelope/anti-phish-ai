import torch
from phishin_train_cnn import LightningCNN
from phishin_train_ffnn import LightningFFNN

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Load models
cnn = LightningCNN.load_from_checkpoint(
    "models/cnn_lightning.ckpt", map_location=DEVICE
)
ffnn = LightningFFNN.load_from_checkpoint(
    "models/ffnn_lightning.ckpt", map_location=DEVICE
)

cnn.eval()
ffnn.eval()

# Test dummy input (a fake URL encoded as float array)
test_url = "https://paypal-login-security-check.com/account"
s = str(test_url)[:200].ljust(200)
cnn_input = torch.tensor(
    [[ord(c) / 128 for c in s]], dtype=torch.float32, device=DEVICE
)
ffnn_input = torch.tensor(
    [[ord(c) / 128 for c in s]], dtype=torch.float32, device=DEVICE
)

# Forward pass
with torch.no_grad():
    cnn_out = torch.sigmoid(cnn(cnn_input))
    ffnn_out = torch.sigmoid(ffnn(ffnn_input))

print(f"✅ CNN output:  {cnn_out.item():.4f}")
print(f"✅ FFNN output: {ffnn_out.item():.4f}")
