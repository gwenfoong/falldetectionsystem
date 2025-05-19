import torch
import cv2
from PIL import Image
from torchvision import transforms

# ── Device helper ────────────────────────────────────────────────────────────
def get_device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ── Transforms ───────────────────────────────────────────────────────────────
frame_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225]),
])

def preprocess_frame(frame: 'np.ndarray') -> torch.Tensor:
    """
    Convert a BGR OpenCV frame -> normalized torch tensor [1,C,H,W].
    """
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    img = Image.fromarray(rgb)
    return frame_transform(img).unsqueeze(0)

def preprocess_sequence(frames: list) -> torch.Tensor:
    """
    Given a list of BGR frames, returns a tensor [1, seq, C, H, W].
    """
    tensors = []
    for frame in frames:
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        img = Image.fromarray(rgb)
        tensors.append(frame_transform(img))
    seq = torch.stack(tensors, dim=0).unsqueeze(0)
    return seq

# ── Model‐loading ────────────────────────────────────────────────────────────
from pipeline.models.my_model import MyBaselineModel
from pipeline.models.temporal_model import MyTemporalModel

def load_frame_model(checkpoint_path: str) -> torch.nn.Module:
    device = get_device()
    model  = MyBaselineModel(num_classes=2)
    state  = torch.load(checkpoint_path, map_location=device)
    # strip or add "model." prefix if needed
    new_state = {}
    for k, v in state.items():
        if k.startswith("model."):
            new_state[k.replace("model.", "")] = v
        else:
            new_state[k] = v
    model.load_state_dict(new_state, strict=False)
    model.to(device).eval()
    return model

def load_temporal_model(checkpoint_path: str) -> torch.nn.Module:
    device = get_device()
    model  = MyTemporalModel(num_classes=2, hidden_size=256)
    state  = torch.load(checkpoint_path, map_location=device)
    # if your checkpoint keys are nested differently, adjust here
    model.load_state_dict(state, strict=False)
    model.to(device).eval()
    return model
