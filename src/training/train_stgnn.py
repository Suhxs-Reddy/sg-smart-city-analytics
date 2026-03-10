"""
Singapore Smart City - Level 3 Trainer
Spatio-Temporal Graph Neural Network (ST-GNN) Physics-Informed Trainer

This module utilizes PyTorch Lightning to orchestrate Distributed Data Parallel (DDP) 
training across multi-node GPU clusters for predicting continuous traffic states.
"""

import argparse

import pytorch_lightning as pl
import torch
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
from pytorch_lightning.loggers import WandbLogger

from src.models.physics_loss import PhysicsInformedLoss
from src.models.stgnn import PINodeSTGNN


class SmartCityODEPredictor(pl.LightningModule):
    """
    Lightning Wrapper for the Physics-Informed Neural ODE architecture.
    Handles the training loop, optimizers, and MLOps metrics logging.
    """
    def __init__(self, learning_rate: float = 1e-3, physics_weight: float = 0.2):
        super(SmartCityODEPredictor, self).__init__()
        self.save_hyperparameters()

        self.model = PINodeSTGNN(num_node_features=12, hidden_dim=64)
        self.criterion = PhysicsInformedLoss(physics_weight=physics_weight)

        # Define continuous time horizons to evaluate (e.g. +15m, +30m, +60m)
        self.register_buffer("eval_times", torch.tensor([1.0, 2.0, 4.0]).float())

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.hparams.learning_rate, weight_decay=1e-4)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=50)
        return {"optimizer": optimizer, "lr_scheduler": scheduler}

    def training_step(self, batch, batch_idx):
        """Standard PyTorch Lightning training step."""
        x, y, edge_idx, edge_wt = batch.x, batch.y, batch.edge_index, batch.edge_weight

        # Get initial condition at t=0
        x_t0 = x[:, :, -1, :]
        current_state = x_t0[:, :, 0] # Vehicle count

        # Predict continuous futures
        pred = self.model(x_t0, edge_idx, edge_wt, self.eval_times)

        # Compute Physics + MSE Loss
        loss, mse_loss, phys_loss = self.criterion(pred, y, current_state, edge_idx)

        self.log('train/loss', loss, prog_bar=True, sync_dist=True)
        self.log('train/mse', mse_loss, sync_dist=True)
        self.log('train/physics_residual', phys_loss, prog_bar=True, sync_dist=True)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y, edge_idx, edge_wt = batch.x, batch.y, batch.edge_index, batch.edge_weight
        x_t0 = x[:, :, -1, :]
        current_state = x_t0[:, :, 0]

        pred = self.model(x_t0, edge_idx, edge_wt, self.eval_times)
        loss, mse_loss, phys_loss = self.criterion(pred, y, current_state, edge_idx)

        self.log('val/loss', loss, sync_dist=True)
        self.log('val/mse', mse_loss, sync_dist=True)

def train(args):
    """
    Main training execution function. Designed to be called by orchestration tools (e.g., KubeFlow).
    """
    print("🚀 Initializing Singapore Smart City Level 3 ST-GNN Trainer...")

    # 1. Setup MLOps Tracking
    logger = WandbLogger(project="sg-smart-city", name="PI-NODE-STGNN") if args.wandb else None

    # 2. Callbacks
    callbacks = [
        ModelCheckpoint(dirpath="models/level3/", monitor="val/loss", mode="min", save_top_k=1),
        EarlyStopping(monitor="val/loss", patience=10, mode="min")
    ]

    # 3. Model
    model = SmartCityODEPredictor(learning_rate=args.lr, physics_weight=args.physics_weight)

    # 4. Trainer (Handles Mixed Precision and DDP automatically)
    trainer = pl.Trainer(
        max_epochs=args.epochs,
        accelerator="auto",
        devices="auto",
        logger=logger,
        callbacks=callbacks,
        precision="16-mixed" if args.fp16 else 32
    )

    # Note: In a real run, we would initialize the PyG DataLoader here using `src.ingestion.dataset`
    # trainer.fit(model, train_dataloaders=train_loader, val_dataloaders=val_loader)
    print("✅ Model compiled. (Data Loaders mocked for architecture setup).")
    return model

if __name__ == "__main__":
    parser = argparse.ArgumentParser("PI-NODE-STGNN Trainer")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--physics_weight", type=float, default=0.2)
    parser.add_argument("--wandb", action="store_true", help="Enable Weights & Biases logging")
    parser.add_argument("--fp16", action="store_true", help="Use Mixed Precision Training")

    args = parser.parse_args()
    # train(args)   # Disabled for local environment safety
