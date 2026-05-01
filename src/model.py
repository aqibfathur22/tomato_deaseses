import torch
import torch.nn as nn
from torchvision import models
from torchvision.models import MobileNet_V3_Small_Weights
from torchinfo import summary

from .config import LEARNING_RATE, NUM_CLASSES, FREEZE_UNTIL

def build_model(class_weights: torch.Tensor = None, device: torch.device = None):

    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    print(f"  Device yang digunakan : {device}")

    # Model
    model = models.mobilenet_v3_small(weights=MobileNet_V3_Small_Weights.IMAGENET1K_V1)

    # freezing  
    for i, layer in enumerate(model.features):
        if i < FREEZE_UNTIL:
            for param in layer.parameters():
                param.requires_grad = False

    in_features = model.classifier[0].in_features  

    model.classifier = nn.Sequential(
        # Linear activations
        nn.Linear(in_features, 576),
        # Non-linear activations
        nn.Hardswish(),
        # Dropout
        nn.Dropout(p=0.4),
        # Output 10 class
        nn.Linear(576, NUM_CLASSES),
    )

    # Hardware placement
    model = model.to(device)

    if class_weights is not None:
        class_weights = class_weights.to(device)

    # loss Function
    criterion = nn.CrossEntropyLoss(
        weight          = class_weights, 
        # label_smoothing = 0.15            
    )

    # Optimizer
    optimizer = torch.optim.AdamW(
        params       = filter(lambda p: p.requires_grad, model.parameters()),
        lr           = LEARNING_RATE,  
        weight_decay = 1e-4            
    )

    return model, optimizer, criterion

def get_gradcam_target_layer(model):

    return model.features[-1]

def count_parameters(model):

    print("\n  MODEL SUMMARY ( MobileNetV3-Small )")
    result = summary(
        model,
        input_size = (1, 3, 224, 224),
        row_settings = ["var_names"],
        verbose = 0
    )

    print(result)

    print("  Jumlah Layer")
    print(f"   - Top-level (children) : {len(list(model.children()))}")
    print(f"   - Features             : {len(list(model.features))}")
    print(f"   - Classifier           : {len(list(model.classifier))}")
    print(f"   - Modules              : {len(list(model.modules()))}")

def run_model(class_weights: torch.Tensor = None):
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    model, optimizer, criterion = build_model(
        class_weights = class_weights,
        device        = device
    )

    count_parameters(model)

    # target_layer = get_gradcam_target_layer(model)
    # print(f"  GradCAM target layer  : {target_layer.__class__.__name__}"
    #     f" — model.features[-1]")

    return model, optimizer, criterion, device