import torch
import torch.nn as nn
import torch.nn.functional as F
import einops


class InputEmbeddings(nn.Module):

  def __init__(self, patches_size:int,latent_size:int, batch_size:int):
    super().__init__()
    # self.input_data    = input_data
    self.patche_size         = patches_size
    self.latent_size         = latent_size
    self.batch_size          = batch_size
    self.input_size          = self.patche_size * self.patche_size*3

    self.class_token         = nn.Parameter(torch.randn(1, 1, self.latent_size))
    self.positional_encoding = nn.Parameter(torch.rand(1, 1, self.latent_size))
    self.layer_norm          = nn.LayerNorm(latent_size)
  
  def forward(self, x):
    x = einops.rearrange(x, 'b c (h h1) (w w1) -> b c (h h1 w w1)', h1 = self.patche_size, w1 =self.patche_size)
    p, n, m = x.shape
    self.liner_projetction   = nn.Linear(m,self.latent_size).to(torch.float32).to('cuda') 
    x = self.liner_projetction(x)
    pos_embedding = einops.repeat(self.positional_encoding, '1 1 h -> 1 m h', m=n+1).to('cuda')
    class_token   = einops.repeat(self.class_token, '1 1 h -> b 1 h', b=p).to('cuda')
    pos_embedding = einops.repeat(self.positional_encoding, '1 1 h -> b 1 h', b=p).to('cuda')
    x  = torch.cat((x, class_token), dim=1)
    x += pos_embedding  
    x  = self.layer_norm(x)
    return x

class EncoderBlock(nn.Module):
  
  def __init__(self, laten_size:int, num_head:int, embdin_dim:int, dropout:int=0.1):
    super().__init__()
    self.laten_size = laten_size
    self.num_head   = num_head
    self.embdin_dim = embdin_dim
    self.droupout   = dropout
    self.attn_blck  = nn.MultiheadAttention(self.embdin_dim, self.num_head, self.droupout)
    self.mlp        = nn.Sequential(
                            nn.LayerNorm(self.laten_size),
                            nn.GELU(),
                            nn.Dropout(self.droupout),
                            nn.Linear(self.laten_size, self.laten_size),
                            nn.Dropout(self.droupout)
                      )
    self.layer_norm = nn.LayerNorm(self.laten_size)
  
  def forward(self,x):
    x = x.to(torch.float32)
    attn = self.attn_blck(x,x,x)[0]
    attn = x + attn
    attn_2 = self.layer_norm(attn)
    x = self.mlp(attn_2)
    x = x + attn
    return x.to('cuda')


class ViTModel(nn.Module):

  def __init__(self, patch_size:int, number_block:int, batch_size:int, embeddin_dim:int, num_head:int, latent_space:int, num_class:int, dropout:int):

    super().__init__()
    self.number_block = number_block
    self.latent_space = latent_space
    self.num_class    = num_class
    self.dropout      = dropout
    self.dim_emb      = embeddin_dim
    self.num_head     = num_head
    self.batch_size   = batch_size
    self.patch_size   = patch_size
    self.encoder      = EncoderBlock(self.latent_space, self.num_head, self.dim_emb, self.dropout)
    self.input_embg   = InputEmbeddings(self.patch_size, self.latent_space, self.batch_size)
    self.mlp = nn.Sequential(
      nn.LayerNorm(self.latent_space),
      nn.Linear(self.latent_space, self.latent_space),
      nn.Linear(self.latent_space, self.num_class)
    )

  def forward(self, x):
    x =x.to(torch.float32)
    
    x = self.input_embg(x)
    for _ in range(1, self.number_block):
      x =  self.encoder(x)
    x = x[:,0]
    # x =  self.mlp(x)
    return x
