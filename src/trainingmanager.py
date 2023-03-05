import logging
import os
from typing import Tuple, Optional
import shutil
from pathlib import Path

import torch
import torch.nn as nn
from torch.nn.modules.loss import _Loss
from torch.optim import Optimizer
from torch.utils.data import Dataset
import torch.profiler
#from torch.profiler import tensorboard_trace_handler

from tqdm import tqdm
import numpy as np


log = logging.getLogger(__name__)
log.setLevel(logging.INFO)

class TrainingManager:
    def __init__(
        self,
        model: nn.Module,
        data: Dataset,
        loss_fn: _Loss,
        optimizer: Optimizer,
        working_dir: str,
        dev: str,
        amp: bool,
    ):
        self._model = model
        self._training_data = data.training_loader
        self._validation_data = data.validation_loader
        self._loss_fn = loss_fn
        self._optimizer = optimizer
        self._working_dir = working_dir
        self._dev = dev
        self.amp = amp

        if self.amp:
            self._scaler = torch.cuda.amp.GradScaler()
        else:
            self._scaler = None

    def save_model(self, epoch: int, val_loss: float, basefolder: Optional[str] = None) -> str:
        basefolder = basefolder if basefolder else self._working_dir
        # Additional information
        if not os.path.exists(basefolder):
            os.mkdir(basefolder)

        PATH = f"{basefolder}/model_{epoch:03d}.pt"

        torch.save({
                    'epoch': epoch,
                    'model_state_dict': self._model.state_dict(),
                    'optimizer_state_dict': self._optimizer.state_dict(),
                    'loss': val_loss,
                    }, PATH)
        return PATH
    
    def load_model(self, path: str) -> Tuple[int, float]:
        checkpoint = torch.load(path)
        self._model.load_state_dict(checkpoint['model_state_dict'])
        self._optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        epoch = checkpoint['epoch']
        loss = checkpoint['loss']

        return (epoch, loss)
            
    def train(self, max_epoch: int = 0, early_stopping: bool = True, save_all_epochs=False) -> None:
        """Train model for max_epoch epochs or early stopping kicks in.
            Restore best model at the end.        

        Args:
            max_epoch (int, optional): _description_. Defaults to 0.
            early_stopping (bool, optional): _description_. Defaults to True.
            save_all_epochs (bool, optional): _description_. Defaults to False.
        """
        max_epoch = 1000 if max_epoch == 0 else max_epoch

        best_validation_loss = np.inf
        best_epoch_path = None
        patience = 5
        for epoch in range(max_epoch):
            log.info(f"Starting training epoch {epoch} ...")
            trn_loss, val_loss = self.train_epoch()
            log.info(f"training loss {trn_loss}, validation loss {val_loss}")

            if val_loss < best_validation_loss or save_all_epochs:
                opath = self.save_model(epoch, val_loss)
                log.info(f"Storing model checkpoint in {opath} ...")
            
                if val_loss < best_validation_loss:
                    best_epoch_path = opath
                    best_validation_loss = val_loss
                    patience = 5

            patience -= 1
            if val_loss >= best_validation_loss and patience <= 0:
                log.info("No training improvement! Will stop training ...")
                break
        
        if best_epoch_path:
            best_epoch_cnt, best_val_loss = self.load_model(best_epoch_path)
            log.info(f"Finished Training with best epoch = {best_epoch_cnt}, val_loss = {best_val_loss}")
            output_model_path = Path(best_epoch_path) / "model_best.pt"
            log.info(f"Storing best model in {output_model_path} ...")
            shutil.copy(best_epoch_path, output_model_path)
        else:
            log.warning("No model finished training!")
        
        log.info("Finished training ...")



    def train_epoch(self) -> Tuple[float, float]:
        """Train the model for one epoch and evaluate it on the validation set.

        Returns:
            Tuple[float, float]: (training loss, validation loss) 
        """

        #with torch.profiler.profile(
        #    schedule=torch.profiler.schedule(
        #        wait=1,
        #        warmup=1,
        #        active=6,
        #        repeat=1),
        #    on_trace_ready=torch.profiler.tensorboard_trace_handler('./tensorboard'),
        #    with_stack=True
        #) as profiler:
        self._model.train()
        loss = 0
        with tqdm(self._training_data, desc="TRN") as loop:
            for batch_idx, (data, targets) in enumerate(loop):
                log.debug(f"Training batch {batch_idx}/{len(self._training_data)} ...")
                data = data.to(device=self._dev)
                targets = targets.float().unsqueeze(1).to(device=self._dev)

                # If automatic mixed precision is available, use it with the GradScaler to increase performance
                if self.amp:
                    # Forward
                    with torch.amp.autocast(
                        device_type=self._dev, dtype=torch.bfloat16
                    ):
                        predictions = self._model(data)
                        loss = self._loss_fn(predictions, targets)

                    # backward
                    self._optimizer.zero_grad(set_to_none=True)
                    self._scaler.scale(loss).backward()
                    self._scaler.step(self._optimizer)
                    self._scaler.update()
                else:
                    # Forward
                    predictions = self._model(data)
                    loss = self._loss_fn(predictions, targets)

                    # backward
                    self._optimizer.zero_grad(set_to_none=True)
                    loss.backward()
                    self._optimizer.step()

                # update tqdm loop
                loop.set_postfix(loss=loss.item())
                #profiler.step()

        logging.debug("Running validation ...")
        val_loss = 0
        self._model.eval()
        with torch.no_grad():
            with tqdm(self._validation_data, desc="VAL") as loop:
                for batch_idx, (data, targets) in enumerate(loop):
                    self._optimizer.zero_grad()
                    data = data.to(device=self._dev)
                    targets = targets.float().unsqueeze(1).to(device=self._dev)

                    # forward
                    predictions = self._model(data)
                    val_loss += self._loss_fn(predictions, targets)

                    loop.set_postfix(loss=val_loss.item()/(batch_idx+1))

        val_loss = val_loss / len(self._validation_data)
        return loss, val_loss
