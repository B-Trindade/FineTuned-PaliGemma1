import torch
if torch.cuda.is_available():
    print(f"Success! PyTorch can see your GPU.")
    print(f"Device name: {torch.cuda.get_device_name(0)}")
else:
    print("Failure. PyTorch still cannot see your GPU.")
    torch.zeros(1).cuda()