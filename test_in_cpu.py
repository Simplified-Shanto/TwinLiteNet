import torch
import numpy as np
import shutil
from tqdm.autonotebook import tqdm
import os
import os
import torch
from model import TwinLite as net
import cv2


def select_device(device='', batch_size=0, newline=True):
    # device = None or 'cpu' or 0 or '0' or '0,1,2,3'
    s = f'TwinLiteNet 🚀 torch-{torch.__version__} '
    device = str(device).strip().lower().replace('cuda:', '').replace('none', '')  # to string, 'cuda:0' to '0'
    cpu = device == 'cpu'
    mps = device == 'mps'  # Apple Metal Performance Shaders (MPS)
    if cpu or mps:
        os.environ['CUDA_VISIBLE_DEVICES'] = '-1'  # force torch.cuda.is_available() = False
    elif device:  # non-cpu device requested
        os.environ['CUDA_VISIBLE_DEVICES'] = device  # set environment variable - must be before assert is_available()
        assert torch.cuda.is_available() and torch.cuda.device_count() >= len(device.replace(',', '')), \
            f"Invalid CUDA '--device {device}' requested, use '--device cpu' or pass valid CUDA device(s)"

    if not cpu and not mps and torch.cuda.is_available():  # prefer GPU if available
        devices = device.split(',') if device else '0'  # range(torch.cuda.device_count())  # i.e. 0,1,6,7
        n = len(devices)  # device count
        if n > 1 and batch_size > 0:  # check batch_size is divisible by device_count
            assert batch_size % n == 0, f'batch-size {batch_size} not multiple of GPU count {n}'
        space = ' ' * (len(s) + 1)
        for i, d in enumerate(devices):
            p = torch.cuda.get_device_properties(i)
            s += f"{'' if i == 0 else space}CUDA:{d} ({p.name}, {p.total_memory / (1 << 20):.0f}MiB)\n"  # bytes to MB
        arg = 'cuda:0'
    elif mps and getattr(torch, 'has_mps', False) and torch.backends.mps.is_available():  # prefer MPS if available
        s += 'MPS\n'
        arg = 'mps'
    else:  # revert to CPU
        s += 'CPU\n'
        arg = 'cpu'

    if not newline:
        s = s.rstrip()
    print(s)
    return torch.device(arg)


def Run(model, img, device):
    img = cv2.resize(img, (640, 360))
    img_rs = img.copy()

    img = img[:, :, ::-1].transpose(2, 0, 1)
    img = np.ascontiguousarray(img)
    img = torch.from_numpy(img)
    img = torch.unsqueeze(img, 0)  # add a batch dimension
    img = img.to(device).float() / 255.0
    with torch.no_grad():
        img_out = model(img)
    x0 = img_out[0]
    x1 = img_out[1]

    _, da_predict = torch.max(x0, 1)
    _, ll_predict = torch.max(x1, 1)

    DA = da_predict.byte().cpu().data.numpy()[0] * 255
    LL = ll_predict.byte().cpu().data.numpy()[0] * 255
    img_rs[DA > 100] = [255, 0, 0]
    img_rs[LL > 100] = [0, 255, 0]

    return img_rs


# Select device (CPU/GPU)
device = select_device('cpu')  # Force CPU usage

model = net.TwinLiteNet()

# Load state dict and handle DataParallel prefix
state_dict = torch.load('pretrained/best.pth', map_location=device)

# Remove 'module.' prefix if present (from DataParallel training)
if list(state_dict.keys())[0].startswith('module.'):
    state_dict = {k[7:]: v for k, v in state_dict.items()}  # Remove 'module.' prefix

model.load_state_dict(state_dict)
model = model.to(device)
model.eval()

image_list = os.listdir('images')
if os.path.exists('results'):
    shutil.rmtree('results')
os.mkdir('results')



# for i, imgName in enumerate(image_list):
#     img = cv2.imread(os.path.join('images', imgName))
#     img = Run(model, img, device)
#     print("Running on ", imgName)
#     cv2.imwrite(os.path.join('results', imgName), img)
#     cv2.imshow("Frame", img)
#
#     key = cv2.waitKey(1)
#     if (key& 0xFF)==ord("q"):
#         break



cap = cv2.VideoCapture(r"test_source\testvideotrim.mp4")
count = 0
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break
    count+=1
    img = Run(model, frame, device)
    cv2.imwrite(os.path.join('results', f"{count}.png"), img)
    cv2.imshow("Frame", img)
    print("Processed frame ", count)
    key = cv2.waitKey(1)
    if (key& 0xFF)==ord("q"):
        break