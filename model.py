import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
import math

# Gradient Reversal Layer (GRL) with dynamic lambda scheduling.
class GradientReversalLayer(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, lambda_adv):
        ctx.lambda_adv = lambda_adv
        # Forward pass is identity.
        return x.view_as(x)
    
    @staticmethod
    def backward(ctx, grad_output):
        # Multiply gradients by -lambda_adv during backpropagation.
        return -ctx.lambda_adv * grad_output, None

def grad_reverse(x, lambda_adv=1.0):
    return GradientReversalLayer.apply(x, lambda_adv)

def compute_grl_lambda(p, gamma=10):
    """
    Compute dynamic lambda using a logistic schedule.
    Args:
        p (float): training progress between 0 and 1.
        gamma (float): controls the speed of increase (default 10).
    Returns:
        lambda_adv (float): value computed as 2/(1+exp(-gamma*p)) - 1.
    """
    return 2.0 / (1.0 + math.exp(-gamma * p)) - 1.0

class DomainClassifier(nn.Module):
    """
    A simple domain classifier with one hidden layer.
    Architecture: FC -> BN -> ReLU -> Dropout -> FC.
    """
    def __init__(self, input_dim=2048, hidden_dim=512, dropout_rate=0.5):
        super(DomainClassifier, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.bn1 = nn.BatchNorm1d(hidden_dim)
        self.dropout = nn.Dropout(dropout_rate)
        self.fc2 = nn.Linear(hidden_dim, 2)  # 2 output classes: source and target.
    
    def forward(self, x):
        x = self.fc1(x)
        x = self.bn1(x)
        x = F.relu(x)
        x = self.dropout(x)
        x = self.fc2(x)
        return x

class ProjectionHead(nn.Module):
    """
    Projects intermediate features into a lower-dimensional embedding space (e.g., 256-d)
    for use in the InfoNCE contrastive loss.
    """
    def __init__(self, input_channels=2048, output_dim=256):
        super(ProjectionHead, self).__init__()
        self.conv1x1 = nn.Conv2d(input_channels, output_dim, kernel_size=1)
        self.relu = nn.ReLU()
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
    
    def forward(self, x):
        x = self.conv1x1(x)
        x = self.relu(x)
        x = self.pool(x)
        x = x.view(x.size(0), -1)  # Flatten to [batch, output_dim]
        return x

class ACADA(nn.Module):
    """
    ACADA Model: Implements domain-adaptive object detection by combining:
      - A shared ResNet50 backbone,
      - A global branch for adversarial domain alignment (with GRL and a domain classifier),
      - A local branch for contrastive learning using a projection head.
    
    The forward pass returns:
      - global_feat_flat: Global features from the backbone (for detection and supervised loss),
      - domain_pred: Domain predictions after applying GRL (for adversarial loss),
      - local_emb: Local embeddings (for contrastive InfoNCE loss).
    
    The GRL lambda (λ) is dynamically updated via update_lambda().
    """
    def __init__(self, num_classes=21, lambda_adv=1.0):
        super(ACADA, self).__init__()
        self.lambda_adv = lambda_adv  # Initial lambda value.
        
        # Shared backbone: Pretrained ResNet50, excluding final pooling and FC layers.
        backbone = models.resnet50(pretrained=True)
        self.feature_extractor = nn.Sequential(*list(backbone.children())[:-2])
        
        # Global Branch: Global average pooling + domain classifier.
        self.global_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.domain_classifier = DomainClassifier(input_dim=2048, hidden_dim=512, dropout_rate=0.5)
        
        # Local Branch: Projection head for instance-level embeddings.
        self.projection_head = ProjectionHead(input_channels=2048, output_dim=256)
    
    def update_lambda(self, p, gamma=10):
        """
        Update the GRL lambda based on training progress.
        Args:
            p (float): Training progress from 0 to 1.
            gamma (float): Hyperparameter for scheduling (default 10).
        Returns:
            new_lambda (float): The updated lambda value.
        """
        new_lambda = compute_grl_lambda(p, gamma)
        self.lambda_adv = new_lambda
        return new_lambda
    
    def forward(self, x):
        """
        Forward pass of the ACADA model.
        Args:
            x (Tensor): Input image tensor of shape [batch, 3, H, W].
        Returns:
            global_feat_flat (Tensor): Global features [batch, 2048].
            domain_pred (Tensor): Domain predictions [batch, 2].
            local_emb (Tensor): Local embeddings [batch, 256].
        """
        # Shared backbone forward pass.
        features = self.feature_extractor(x)  # [batch, 2048, H', W']
        
        # Global Branch: Global average pooling and flatten.
        global_feat = self.global_pool(features)  # [batch, 2048, 1, 1]
        global_feat_flat = global_feat.view(global_feat.size(0), -1)  # [batch, 2048]
        
        # Apply GRL with current lambda.
        reversed_feat = grad_reverse(global_feat_flat, self.lambda_adv)
        domain_pred = self.domain_classifier(reversed_feat)
        
        # Local Branch: Compute local embeddings.
        local_emb = self.projection_head(features)  # [batch, 256]
        
        return global_feat_flat, domain_pred, local_emb

# Example usage for testing:
if __name__ == "__main__":
    model = ACADA(lambda_adv=0.5)  # Initialize with a starting lambda value.
    model.train()
    dummy_input = torch.randn(4, 3, 512, 512)  # Dummy input batch of 4 images.
    global_feat, domain_pred, local_emb = model(dummy_input)
    print("Global feature shape:", global_feat.shape)  # Expected: [4, 2048]
    print("Domain prediction shape:", domain_pred.shape)  # Expected: [4, 2]
    print("Local embedding shape:", local_emb.shape)  # Expected: [4, 256]
    
    # Example of updating lambda using training progress p (0.0 to 1.0)
    progress = 0.3  # For example, 30% of training is done.
    new_lambda = model.update_lambda(progress, gamma=10)
    print(f"Updated GRL lambda (p={progress}): {new_lambda:.4f}")
