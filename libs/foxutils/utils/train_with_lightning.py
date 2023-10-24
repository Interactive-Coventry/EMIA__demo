from .core_utils import SEED, models_dir

from os.path import join as pathjoin
from os.path import isfile, splitext

# PyTorch
import torch
from torch import Tensor
from torch import nn
import torch.optim as optim
import torch.nn.functional as F
import torchmetrics

# Torchvision
import torchvision
import lightning.pytorch as pl
from lightning.pytorch.callbacks import EarlyStopping, LearningRateMonitor, ModelCheckpoint
from lightning.pytorch.loggers import TensorBoardLogger

# Kornia
from kornia import image_to_tensor, tensor_to_image
from kornia import augmentation

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

import warnings

warnings.filterwarnings("ignore", ".*Consider increasing the value of the `num_workers` argument*")

# Setting the seed
pl.seed_everything(SEED)

# Ensure that all operations are deterministic on GPU (if used) for reproducibility
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

# You are using a CUDA device ('NVIDIA GeForce RTX 3060 Laptop GPU') that has Tensor Cores. To properly utilize them,
# you should set `torch.set_float32_matmul_precision('medium' | 'high')` which will trade-off precision for
# performance. For more details, read https://pytorch.org/docs/stable/generated/torch.set_float32_matmul_precision
# .html#torch.set_float32_matmul_precision
torch.set_float32_matmul_precision('medium')


#################################################################################

def get_lightning_log_dir(model_name):
    lightning_log_dir = pathjoin(models_dir, model_name)
    return lightning_log_dir


def get_lightning_checkpoint_path(lightning_log_dir, checkpoint_version):
    checkpoint_path = lightning_log_dir.replace('\\', '/') + "/lightning_logs/version_" + str(checkpoint_version) \
                      + "/checkpoints/"
    return checkpoint_path


def get_lightning_checkpoint_file(lightning_log_dir, checkpoint_version, checkpoint_file):
    checkpoint_path = get_lightning_checkpoint_path(lightning_log_dir, checkpoint_version)
    if '.' not in checkpoint_file:
        checkpoint_file = checkpoint_file + '.ckpt'
    pretrained_filename = pathjoin(checkpoint_path, checkpoint_file)
    return pretrained_filename


def pl_load_trained_model(target_model_class, weight_path, **model_params):
    weight_path_file, weight_path_ext = splitext(weight_path)
    assert weight_path_ext == '.pts', True
    print(f'Model is loaded from location: {weight_path}')
    target_model = target_model_class(**model_params)
    target_model.load_state_dict(torch.load(weight_path))
    target_model.eval()
    return target_model


# PyTorch Lightning >= 2.0
def pl_load_trained_model_from_checkpoint(target_model_class, checkpoint_path, **model_params):
    checkpoint_path_file, checkpoint_path_ext = splitext(checkpoint_path)
    assert checkpoint_path_ext == '.ckpt', True
    print(f'Model is loaded from checkpoint: {checkpoint_path}')
    target_model = target_model_class.load_from_checkpoint(checkpoint_path=checkpoint_path, **model_params)
    checkpoint = torch.load(checkpoint_path, map_location=lambda storage, loc: storage)
    hyperparams = checkpoint["hyper_parameters"]
    print(f'Loaded hyperparameters: {hyperparams}')
    target_model.eval()
    return target_model


#################################################################################
# LightningModule classes

def get_binary_label(model, x):
    outputs = torch.squeeze(model(x))
    preds = torch.round(outputs)
    return outputs, preds


def get_label_probs(model, x):
    outputs = model(x)
    return outputs, outputs


def get_argmax_label(model, x):
    outputs = model(x)
    _, preds = torch.max(outputs, 1)
    return outputs, preds


def get_sgd_optimizer(self):
    optimizer = torch.optim.SGD(self.parameters(), lr=0.001)
    return optimizer


def get_conf_matrix_fig(cm, class_mapping=None):
    df_cm = pd.DataFrame(cm)
    if class_mapping:
        inv_map = {v: k for k, v in class_mapping.items()}
        df_cm.rename(columns=inv_map, index=inv_map, inplace=True)

    plt.figure(figsize=(10, 7))
    fig_ = sns.heatmap(df_cm, annot=True, cmap='Spectral', fmt='d').get_figure()
    plt.close(fig_)
    return fig_


# Define a custom weight initialization function
def custom_weights_init(layer):
    if isinstance(layer, nn.Linear):
        layer.reset_parameters()
        # nn.init.xavier_uniform_(layer.weight)
        # layer.bias.data.fill_(0.01)

    if isinstance(layer, nn.Conv2d):
        layer.reset_parameters()


class DataAugmentation(nn.Module):
    """Module to perform data augmentation using Kornia on torch tensors."""

    def get_default_transforms(self, p):
        default_transforms = nn.Sequential(
            augmentation.RandomHorizontalFlip(p=p),
            augmentation.RandomRotation(degrees=10.0, p=p, keepdim=False),
            # augmentation.RandomBrightness(p=p),
            augmentation.RandomPerspective(0.1, p=p, keepdim=False),
            # augmentation.RandomThinPlateSpline(scale=0.1, p=p),
        )
        return default_transforms

    def get_color_transforms(self):
        transform = augmentation.ColorJitter(0.5, 0.5, 0.5, 0.5)
        return transform

    def keep_original_dim(self, x: Tensor, orig_dim: tuple) -> Tensor:
        new_dim = x.shape[-2:]
        if self.keep_orig_dim:
            transform = nn.Sequential(
                augmentation.CenterCrop((new_dim[0] - 30, new_dim[1] - 30), p=1, keepdim=True),
                augmentation.Resize(orig_dim, p=1, keepdim=True),
            )
            return transform(x)
        else:
            return x

    def __init__(self, transforms=None, p=0.5, keep_orig_dim=False) -> None:
        super().__init__()
        self.keep_orig_dim = keep_orig_dim
        if transforms is None:
            self.transforms = self.get_default_transforms(p)
        else:
            self.transforms = transforms

    @torch.no_grad()  # disable gradients for effiency
    def forward(self, x: Tensor) -> Tensor:
        orig_dim = x.shape[-2:]
        x_out = self.transforms(x)  # BxCxHxW
        x_out = self.keep_original_dim(x_out, orig_dim)
        return x_out


class PredictionModel(pl.LightningModule):
    def __init__(self, model_class, task="binary", num_classes=2, num_labels=2, forward_function=get_binary_label,
                 loss_fun=nn.BCELoss, configure_optimizers_fun=None, average='micro', has_augmentation=False,
                 aug_transforms=None, aug_p=0.5, class_mapping=None, **model_hyperparameters):
        super().__init__()
        self.save_hyperparameters(ignore=['loss_fun', 'model_class'])
        self.class_mapping = class_mapping
        self.has_augmentation = has_augmentation
        self.transform = DataAugmentation(transforms=aug_transforms, p=aug_p, keep_orig_dim=True)
        self.forward_function = forward_function
        self.loss_fun = loss_fun  # F.binary_cross_entropy()
        if len(model_hyperparameters) == 0:
            self.model = model_class
        else:
            self.model = model_class(**model_hyperparameters)

        if configure_optimizers_fun is None:
            self.automatic_optimization = True
        else:
            self.automatic_optimization = False
            self.configure_optimizers = configure_optimizers_fun

        self.average = average
        self.train_precision = torchmetrics.Precision(task=task, num_classes=num_classes, top_k=1,
                                                      num_labels=num_labels, average=average)
        self.valid_precision = torchmetrics.Precision(task=task, num_classes=num_classes, top_k=1,
                                                      num_labels=num_labels, average=average)
        self.train_recall = torchmetrics.Recall(task=task, num_classes=num_classes, top_k=1,
                                                num_labels=num_labels, average=average)
        self.valid_recall = torchmetrics.Recall(task=task, num_classes=num_classes, top_k=1,
                                                num_labels=num_labels, average=average)
        self.train_acc = torchmetrics.Accuracy(task=task, num_classes=num_classes, top_k=1,
                                               num_labels=num_labels, average=average)
        self.valid_acc = torchmetrics.Accuracy(task=task, num_classes=num_classes, top_k=1,
                                               num_labels=num_labels, average=average)
        self.train_cm = torchmetrics.ConfusionMatrix(num_classes=num_classes)
        self.valid_cm = torchmetrics.ConfusionMatrix(num_classes=num_classes)

        self.training_step_preds = []
        self.training_step_labels = []

        self.validation_step_preds = []
        self.validation_step_labels = []

    def forward(self, x):
        _, preds = self.forward_function(self.model, x)
        return preds

    def show_batch(self, dataloader, win_size=(10, 10), transform=None):
        def _to_vis(data):
            if transform is not None:
                data = [transform(x) for x in data]
            return tensor_to_image(torchvision.utils.make_grid(data, nrow=8))

        imgs, labels = next(iter(dataloader))
        if self.has_augmentation:
            imgs_aug = self.transform(imgs)  # apply transforms
        else:
            imgs_aug = imgs

        # use matplotlib to visualize
        plt.figure(figsize=win_size)
        plt.imshow(_to_vis(imgs))
        plt.figure(figsize=win_size)
        plt.imshow(_to_vis(imgs_aug))

    def on_after_batch_transfer(self, batch, dataloader_idx):
        x, y = batch
        if self.trainer.training and self.has_augmentation:
            x = self.transform(x)  # => we perform GPU/Batched data augmentation
        return x, y

    def training_step(self, batch, batch_idx):
        # training_step defines the train loop.
        feature_vectors, labels = batch

        # zero the parameter gradients
        optimizer = self.optimizers()
        optimizer.zero_grad()

        with torch.set_grad_enabled(True):
            outputs, preds = self.forward_function(self.model, feature_vectors)

            loss = self.loss_fun(outputs, labels)
            self.log('train_loss', loss, prog_bar=True, on_step=False, on_epoch=True, logger=True)
            self.logger.experiment.add_scalars('loss', {'train': loss}, self.global_step)

            # backward + optimize only if in training phase
            self.manual_backward(loss)
            optimizer.step()

            self.training_step_preds.append(preds)
            self.training_step_labels.append(labels)

        return loss

    def on_train_epoch_end(self):
        preds = torch.cat(self.training_step_preds)
        labels = torch.cat(self.training_step_labels)

        acc = self.train_acc(preds, labels)
        self.logger.experiment.add_scalars('acc', {'train': acc}, self.global_step)
        self.log('train_acc', acc, prog_bar=True, on_step=False, on_epoch=True, logger=True)

        rc = self.train_recall(preds, labels)
        self.logger.experiment.add_scalars('recall', {'train': rc}, self.global_step)
        self.log('train_recall', rc, prog_bar=True, on_step=False, on_epoch=True, logger=True)

        pc = self.train_precision(preds, labels)
        self.logger.experiment.add_scalars('precision', {'train': pc}, self.global_step)
        self.log('train_precision', pc, prog_bar=True, on_step=False, on_epoch=True, logger=True)

        cm = self.train_cm(preds, labels)
        computed_confusion = cm.detach().cpu().numpy().astype(int)
        fig_ = get_conf_matrix_fig(computed_confusion, self.class_mapping)
        self.logger.experiment.add_figure("Train Confusion Matrix", fig_, self.current_epoch)

        # Free memory
        self.training_step_preds.clear()
        self.training_step_labels.clear()

        sch = self.lr_schedulers()
        sch.step()

    def validation_step(self, batch, batch_idx):
        feature_vectors, labels = batch
        outputs, preds = self.forward_function(self.model, feature_vectors)

        loss = self.loss_fun(outputs, labels)
        self.log('val_loss', loss, prog_bar=True, on_step=False, on_epoch=True, logger=True)
        self.logger.experiment.add_scalars('loss', {'valid': loss}, self.global_step)

        self.validation_step_preds.append(preds)
        self.validation_step_labels.append(labels)

        return loss

    def on_validation_epoch_end(self):
        preds = torch.cat(self.validation_step_preds)
        labels = torch.cat(self.validation_step_labels)

        acc = self.valid_acc(preds, labels)
        self.logger.experiment.add_scalars('acc', {'valid': acc}, self.global_step)
        self.log('val_acc', acc, prog_bar=True, on_step=False, on_epoch=True, logger=True)

        rc = self.valid_recall(preds, labels)
        self.logger.experiment.add_scalars('recall', {'valid': rc}, self.global_step)
        self.log('val_recall', rc, prog_bar=True, on_step=False, on_epoch=True, logger=True)

        pc = self.valid_precision(preds, labels)
        self.logger.experiment.add_scalars('precision', {'valid': pc}, self.global_step)
        self.log('val_precision', pc, prog_bar=True, on_step=False, on_epoch=True, logger=True)

        cm = self.valid_cm(preds, labels)
        computed_confusion = cm.detach().cpu().numpy().astype(int)
        fig_ = get_conf_matrix_fig(computed_confusion, self.class_mapping)
        self.logger.experiment.add_figure("Valid Confusion Matrix", fig_, self.current_epoch)

        # Free memory
        self.validation_step_preds.clear()
        self.validation_step_labels.clear()

    def configure_optimizers(self):
        return self.configure_optimizers_fun()


class ImageModel(pl.LightningModule):
    def __init__(self, num_input_channels, width, height, model_class, **model_hyperparameters):
        super().__init__()
        self.save_hyperparameters()
        self.model = model_class(**model_hyperparameters, width=width, height=height)
        self.example_input_array = torch.zeros(2, num_input_channels, width, height)

    def forward(self, x):
        outputs = self.model(x)
        return outputs

    @staticmethod
    def _get_reconstruction_loss(batch):
        """
        Given a batch of images, this function returns the reconstruction loss (MSE in our case)
        """
        x, labels = batch
        loss = F.mse_loss(x, labels, reduction="none")
        loss = loss.sum(dim=[1, 2, 3]).mean(dim=[0])
        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-3)
        # Using a scheduler is optional but can be helpful.
        # The scheduler reduces the LR if the validation performance hasn't improved for the last N epochs
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer,
                                                               mode='min',
                                                               factor=0.2,
                                                               patience=20,
                                                               min_lr=5e-5)
        return {"optimizer": optimizer, "lr_scheduler": scheduler}

    def training_step(self, batch, batch_idx):
        loss = self._get_reconstruction_loss(batch)
        self.log("train_loss", loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        return loss

    def validation_step(self, batch, batch_idx):
        loss = self._get_reconstruction_loss(batch)
        self.log("val_loss", loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)

    def test_step(self, batch, batch_idx):
        loss = self._get_reconstruction_loss(batch)
        self.log("test_loss", loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)


#################################################################################
# Lightning Training

def train_predictive_model(target_model_class, lightning_log_dir, data_generators, epochs=2, early_stopping_patience=5,
                           **model_params):
    target_model = target_model_class(**model_params)

    lr_logger = LearningRateMonitor("epoch")
    logger = TensorBoardLogger(lightning_log_dir)

    callbacks = [lr_logger,
                 PrintLossCallback(every_n_epochs=5),
                 ModelCheckpoint(monitor='val_acc', save_top_k=1, save_weights_only=False, mode='max')
                 ]

    if early_stopping_patience is not None:
        print(f'Early stop callback with patience {early_stopping_patience} enabled')
        early_stop_callback = EarlyStopping(monitor="val_acc", min_delta=1e-6, patience=early_stopping_patience,
                                            verbose=True, mode="max")
        callbacks.append(early_stop_callback)

    trainer = pl.Trainer(
        log_every_n_steps=4,
        max_epochs=epochs,
        accelerator='gpu',
        devices=1,
        enable_model_summary=True,
        callbacks=callbacks,
        logger=logger)

    trainer.fit(
        target_model,
        train_dataloaders=data_generators["train"],
        val_dataloaders=data_generators["valid"])

    checkpoint_path = trainer.checkpoint_callback.best_model_path
    target_model = pl_load_trained_model_from_checkpoint(target_model_class, checkpoint_path, **model_params)

    return target_model, trainer


class PrintLossCallback(pl.Callback):
    def __init__(self, every_n_epochs=20):
        super().__init__()
        self.every_n_epochs = every_n_epochs

    def on_epoch_end(self, trainer, pl_module):
        metrics = trainer.callback_metrics
        print(f"Epoch: {trainer.current_epoch}, "
              f"Train Loss: {metrics['train_loss']:.4f}"
              f"Validation Loss: {metrics['val_loss']:.4f}")


class GenerateCallbackForImageReconstruction(pl.Callback):

    def __init__(self, input_imgs, every_n_epochs=20):
        super().__init__()
        self.input_imgs = input_imgs  # Images to reconstruct during training
        # Only save those images every N epochs (otherwise tensorboard gets quite large)
        self.every_n_epochs = every_n_epochs

    def on_train_epoch_end(self, trainer, pl_module):
        if trainer.current_epoch % self.every_n_epochs == 0:
            # Reconstruct images
            input_imgs = self.input_imgs.to(pl_module.device)
            with torch.no_grad():
                pl_module.eval()
                reconst_imgs = pl_module(input_imgs)
                pl_module.train()
            # Plot and add to tensorboard
            imgs = torch.stack([input_imgs, reconst_imgs], dim=1).flatten(0, 1)
            grid = torchvision.utils.make_grid(imgs, nrow=2, normalize=True, range=(-1, 1))
            trainer.logger.experiment.add_image("Reconstructions", grid, global_step=trainer.global_step)


def train_image_reconstruction_model(target_model_class, lightning_log_dir, data_generators, epochs=2,
                                     has_checkpoint=False, checkpoint_path=None, pretrained_filename=None,
                                     callback_data=None, **model_params):
    target_model = target_model_class(**model_params)

    lr_logger = LearningRateMonitor("epoch")
    logger = TensorBoardLogger(lightning_log_dir)
    early_stop_callback = EarlyStopping(monitor="val_mse_loss", min_delta=1e-4, patience=5, verbose=True, mode="min")

    # Create a PyTorch Lightning trainer with the generation callback
    trainer = pl.Trainer(
        max_epochs=epochs,
        accelerator='gpu',
        devices=1,
        enable_model_summary=True,
        gradient_clip_val=0.1,
        callbacks=[lr_logger,
                   early_stop_callback,
                   ModelCheckpoint(monitor='val_acc', save_top_k=1, save_weights_only=False, mode='max'),
                   GenerateCallbackForImageReconstruction(callback_data, every_n_epochs=1)],
        logger=logger)

    trainer.logger._log_graph = True  # If True, we plot the computation graph in tensorboard
    trainer.logger._default_hp_metric = None  # Optional logging argument that we don't need

    # Check whether pretrained model exists. If yes, load it and skip training
    if has_checkpoint and isfile(pretrained_filename):
        print("Found pretrained model, loading...")

        target_model = target_model.load_from_checkpoint(pretrained_filename)
        # target_model = pl_load_trained_model_from_checkpoint(target_model_class, checkpoint_path, **model_params)

    else:
        trainer.fit(target_model, data_generators["train"], data_generators["valid"])

        checkpoint_path = trainer.checkpoint_callback.best_model_path
        print('Loading model from the best path: ', checkpoint_path)
        target_model = pl_load_trained_model_from_checkpoint(target_model_class, checkpoint_path, **model_params)

    # Test best model on validation and test set
    val_result = trainer.test(target_model, data_generators["valid"], verbose=False)
    test_result = trainer.test(target_model, data_generators["test"], verbose=False)
    result = {"test": test_result, "val": val_result}
    return target_model, result, trainer