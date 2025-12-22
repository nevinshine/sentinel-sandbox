import torch
import torch.nn as nn
from models.efd_lut import EFDLUT

class DWNClassifier(nn.Module):
    """
    Standard Weightless Architecture (WiSARD / ULEEN style).
    
    Instead of a shared feature extractor + Linear Head, we use
    class-specific Discriminators.
    
    - Class 0 (Normal): Has its own independent LUTs.
    - Class 1 (Attack): Has its own independent LUTs.
    
    Prediction = argmax(Response_0, Response_1)
    """
    def __init__(self, num_inputs, tuple_size=4, num_classes=2):
        super().__init__()
        self.num_classes = num_classes
        
        # Create one independent EFD-LUT bank per class
        # This isolates the memory of "Normal" from "Attack"
        self.discriminators = nn.ModuleList([
            EFDLUT(num_inputs, tuple_size) for _ in range(num_classes)
        ])

    def forward(self, x):
        """
        Args:
            x: Binary Input [Batch, Num_Inputs]
        Returns:
            scores: [Batch, Num_Classes] -> Raw summation scores
        """
        class_responses = []
        
        for discriminator in self.discriminators:
            # Each discriminator outputs a raw sum of active LUT entries [Batch]
            response = discriminator(x) 
            class_responses.append(response)
            
        # Stack into a single tensor [Batch, Num_Classes]
        # Column 0 = Normal Score, Column 1 = Attack Score
        return torch.stack(class_responses, dim=1)