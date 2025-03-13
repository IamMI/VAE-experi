# This file is built by MI which responds to the following snippet:
# Build VAE and train it

# =============== Ordinary ================== #
from diffusers import AutoencoderKL
import torch
import torch.utils
import torch.nn.functional as F
from tqdm import tqdm
import numpy as np
import time
import os
import sys
current_work_dir = os.path.dirname(os.path.dirname(__file__))
sys.path.append(current_work_dir)
# =============== Artificial Package ================ #
from Datasets import ImageNetdatasets
from CVAE import ConditionalVAE
# =============== tensorboard ======================= #
from torch.utils.tensorboard import SummaryWriter
# =============== DDP ==================== #
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.distributed as dist
import torch.multiprocessing as mp

def setup(rank, world_size):
    """
    DDP initialize, come from pytorch document
    """
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'

    # initialize the process group
    dist.init_process_group("nccl", rank=rank, world_size=world_size)

def cleanup():
    """
    DDP destroy, come from pytorch document
    """
    dist.destroy_process_group()


class VAE_MI:
    def __init__(self, 
                 index=148, noise_schedule="linear", diffusion_steps=1000):
        """
        Initialize the VAE model
        :param index: Index of insert timestep
        :param noise_schedule: Schedule of betas
        :param diffusion_step: Number of total timestep
        """
        self.model = ConditionalVAE(num_classes=1000, embedding_dim=32*32)
        
        # Encoder-Decoder VAE
        self.vae = AutoencoderKL.from_pretrained(f"stabilityai/sd-vae-ft-mse")
        # Noise
        betas = np.linspace(0.0001, 0.02, 1000, dtype=np.float64)
        alphas = 1.0 - betas
        alphas_cumprod = np.cumprod(alphas, axis=0)
        self.coeffs = torch.tensor(alphas_cumprod[index])
        
        
    def loss_func(self, recon_x, x, mu, logvar):
        """
        Loss function
        """
        recon_loss = F.mse_loss(recon_x, x, reduction='sum')
        kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
        return recon_loss + kl_loss        
        
    def train_init(self, batch_size=64, num_epoch=1500):
        """
        Initialize train step
        """
        # Port check
        if os.system('lsof -i:12355') != 256:
            print("Port 12355 is busy! If available, you could shut down port using 'kill -9 PID' command.")
            exit()
        
        # GPUs count
        print('='*40)
        print(f"Found {torch.cuda.device_count()} GPUs!")
        # Optimizer
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=1e-3)
        
        # Load datasets
        self.train_dataset = ImageNetdatasets().load_dataset("train")
        self.eval_dataset = ImageNetdatasets().load_dataset("val")
        print(f"Train dataset's size: {len(self.train_dataset)}")
        print(f"Eval dataset's size: {len(self.eval_dataset)}")
        
        # Meta data
        self.batch_size = batch_size
        self.num_epoch = num_epoch
        print(f"Batch size: {self.batch_size}")
        print(f"Number of epochs: {self.num_epoch}")
        
        print("\n")
        print("We use subset of ImageNet-1k to train our model!! Contained data follow:")
        for key in self.train_dataset.classes:
            print(f"Folder: {key}, label: {self.train_dataset.class_to_idx[key]}")
        
        print('='*40)
        
    def train(self, rank, world_size):
        """
        Train the VAE model
        :param rank: DDP params
        :param world_size: DDP params
        """
        
        print(f"Running DDP of model on rank {rank}.")
        setup(rank, world_size)
    
        # Dataloader
        train_sampler =torch.utils.data.distributed.DistributedSampler(self.train_dataset)
        self.train_loader = torch.utils.data.DataLoader(self.train_dataset, batch_size=self.batch_size, sampler=train_sampler, drop_last=True)
        if rank == 0:
            # Process 0 responds for eval and record
            eval_loader = torch.utils.data.DataLoader(self.eval_dataset, batch_size=self.batch_size, shuffle=False)
            # Create results folder
            t = time.localtime()
            result_name = str(t.tm_mon) + '-' + str(t.tm_mday) + '-' + str(t.tm_hour) + '-' + str(t.tm_min)
            folder = os.path.join(current_work_dir, 'results', result_name)
            os.makedirs(folder, exist_ok=True)
            os.makedirs(os.path.join(folder, 'checkpoints'), exist_ok=True)
            os.makedirs(os.path.join(folder, 'logs'), exist_ok=True)
            # Tensorboard
            tb_writer = SummaryWriter(log_dir=os.path.join(folder, 'logs'))
            
        # DDP training
        device = torch.device("cuda", rank)
        
        self.vae = self.vae.to(device)
        model = self.model.to(device)
        ddp_model = DDP(model, device_ids=[rank], output_device=rank)

        for epoch in range(self.num_epoch):
            dist.barrier()
            ddp_model.module.train()
            
            total_loss = 0
            self.train_loader.sampler.set_epoch(epoch)
            
            start_time = time.time()
            for (data, labels) in tqdm(self.train_loader, disable=(rank != 0), desc=f"Epoch {epoch+1}/{self.num_epoch}"):
                data, labels = data.to(device), labels.to(device)
                # Add noise to the image
                with torch.no_grad():
                    x = self.vae.encode(data).latent_dist.sample().mul_(0.18215).squeeze(0)
                    noise = torch.randn_like(x).to(device)
                    data = torch.sqrt(self.coeffs) * x + torch.sqrt(1.0 - self.coeffs) * noise
                
                self.optimizer.zero_grad()
                recon_x, posterior, z = ddp_model(data, labels)  
                
                # Compute loss
                loss = self.loss_func(recon_x, data, posterior.mean, posterior.logvar)
                # Backward
                loss.backward()
                self.optimizer.step()
                total_loss += loss.item()                
            end_time = time.time()
            
            
            if rank==0:
                # Caculate average train loss
                avg_loss = total_loss / len(self.train_loader.dataset)
                print(f"Epoch [{epoch+1}/{self.num_epoch}], Average Loss: {avg_loss:.6f}, Timeout: {(end_time-start_time)} s")
                
                # Caculate average eval loss
                ddp_model.module.eval()
                eval_loss = 0
                for (data, labels) in eval_loader:
                    data, labels = data.to(device), labels.to(device)
                    with torch.no_grad():
                        x = self.vae.encode(data).latent_dist.sample().mul_(0.18215).squeeze(0)
                        noise = torch.randn_like(x).to(device)
                        data = torch.sqrt(self.coeffs) * x + torch.sqrt(1.0 - self.coeffs) * noise
                
                    recon_x, posterior, z = ddp_model.module(data, labels)
                    loss = self.loss_func(recon_x, data, posterior.mean, posterior.logvar)
                    eval_loss += loss.item()
                avg_eval_loss = eval_loss / len(eval_loader.dataset)
                print(f"Epoch [{epoch+1}/{self.num_epoch}], Average Eval Loss: {avg_eval_loss:.6f}")
                
                # Record 
                tb_writer.add_scalar('Train loss', avg_loss, global_step=epoch)
                tb_writer.add_scalar('Eval loss', avg_eval_loss, global_step=epoch)
                
                if (epoch+1)%20==0:
                    # Save model
                    print("Begin to save model.")
                    checkpoints = {
                        'model_state_dict': ddp_model.module.state_dict(),
                        'optimizer_state_dict': self.optimizer.state_dict(),
                        'epoch': epoch + 1, 
                        'train_loss': avg_loss,
                        'eval_loss': avg_eval_loss,
                    }
                    path = os.path.join(folder, "checkpoints", "CVAE-MI-{:03d}.pt".format(epoch))
                    torch.save(checkpoints, path)
            
            dist.barrier()
            
        cleanup()
        if rank==0:
            tb_writer.close()

if __name__ == "__main__":
    vae = VAE_MI()
    vae.train_init()
    mp.spawn(vae.train,
             args=(torch.cuda.device_count(),),
             nprocs=torch.cuda.device_count(),
             join=True)
    print("Done.")