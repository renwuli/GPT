import torch
import os
import random
import numpy as np


class BaseTrainer(object):
    def __init__(self, opts):
        self.opts = opts
        self.global_step = 0
        self.epoch = 0
        self.logger = None
        self.model = None
        self.loss = None
        self.optimizer = None
        self.lr_scheduler = None
        self.train_dataset = None
        self.val_dataset = None
        self.test_dataset = None

        self.train_loader = None
        self.val_loader = None
        self.test_loader = None

        self.prepare_summary_writer(opts)
        self.build_model(opts)
        self.prepare_dataset(opts)
        self.create_optimizers(opts)
        self.build_loss(opts)
        self.seed_torch(opts)

    def prepare_summary_writer(self, opts):
        raise NotImplementedError

    def build_model(self, opts):
        raise NotImplementedError

    def build_loss(self, opts):
        raise NotImplementedError

    def create_optimizers(self, opts):
        raise NotImplementedError

    def prepare_dataset(self, opts):
        raise NotImplementedError

    def tocuda(self, vars):
        for i in range(len(vars)):
            vars[i] = vars[i].cuda()
        return vars

    def train(self):
        self.logger.write(">>> Starting training ...")
        start_epoch = 0
        if self.opts.ckpt_path is not None:
            start_epoch = self.load(self.opts.ckpt_path) + 1

        for epoch in range(start_epoch, self.opts.train_epochs):
            self.logger.write(f">>> Training epoch: {epoch} ...")
            self.epoch = epoch
            self.train_epoch()

            self.lr_scheduler.step()

            if epoch % self.opts.save_epochs == 0:
                self.save(epoch)
            if epoch % self.opts.eval_epochs == 0:
                self.logger.write(f">>> Evaulating epoch: {epoch} ...")
                self.evaluate()

            self.reset_dataset()

        self.logger.close()
    
    def reset_dataset(self):
        pass

    def train_epoch(self):
        pass

    @torch.no_grad()
    def evaluate(self):
        raise NotImplementedError

    @torch.no_grad()
    def test(self):
        pass

    def save(self, epoch):
        ckpt_dir = os.path.join(self.logger.save_dir, "checkpoints")
        if not os.path.exists(ckpt_dir):
            os.makedirs(ckpt_dir)
        ckpt_path = os.path.join(ckpt_dir, f"{epoch}.ckpt")

        self.logger.write(f">>> Saving checkpoint into {ckpt_path} ...")

        torch.save(
            {
                'epoch': epoch,
                'model_state_dict': self.model.state_dict(),
                'optimizer_state_dict': self.optimizer.state_dict(),
            }, ckpt_path)
        self.logger.write(">>> Save Done")

    def load(self, ckpt_path):
        self.logger.write(f">>> Loading checkpoint from {ckpt_path} ...")

        checkpoints = torch.load(ckpt_path)
        self.logger.write(">>> Done")

        self.model.load_state_dict(checkpoints['model_state_dict'])
        self.optimizer.load_state_dict(checkpoints['optimizer_state_dict'])

        self.logger.write(">>> Load Done")
        return checkpoints['epoch']

    def seed_torch(self, opts):
        random.seed(opts.seed)
        os.environ['PYTHONHASHSEED'] = str(opts.seed)
        np.random.seed(opts.seed)
        torch.manual_seed(opts.seed)
        torch.cuda.manual_seed(opts.seed)
        torch.cuda.manual_seed_all(opts.seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.enabled = False