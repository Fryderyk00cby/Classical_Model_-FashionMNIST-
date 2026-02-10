import torch
from torch import nn
from functools import partial
from collections import OrderedDict

def drop_path(x,drop_prob:float=0.,training=False):
    if drop_prob == 0. or not training:
        return x
    keep_prob = 1 - drop_prob
    shape = (x.shape[0],) + (1,) * (x.ndim - 1) #生成一个形状为(batch_size,1,1,...,1)的张量，用于在drop_path中进行广播操作。
    random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device) #生成一个随机张量，值在[keep_prob,1+keep_prob)之间。
    random_tensor.floor_() #对随机张量进行向下取整操作，使其值变为0或1。
    output = x.div(keep_prob) * random_tensor #将输入张量除以keep_prob，并乘以随机张量，实现drop_path的功能。
    return output
class DropPath(nn.Module):
    def __init__(self,drop_prob=None):
        super(DropPath,self).__init__()
        self.drop_prob = drop_prob
    def forward(self,x):
        return drop_path(x,self.drop_prob,self.training)
    

def _init_vit_weights(m):
    if isinstance(m,nn.Linear):
        nn.init.trunc_normal_(m.weight,std=0.02) #线性层权重初始化，使用截断正态分布
        if isinstance(m.bias,nn.Parameter):
            nn.init.zeros_(m.bias) #线性层偏置初始化，使用全零
    elif isinstance(m,nn.Conv2d):
        nn.init.kaiming_normal_(m.weight,mode='fan_out') #卷积层权重初始化，使用Kaiming正态分布
        if isinstance(m.bias,nn.Parameter):
            nn.init.zeros_(m.bias) #卷积层偏置初始化，使用全零
    elif isinstance(m,nn.LayerNorm):
        nn.init.zeros_(m.bias) #LayerNorm层偏置初始化，使用全零
        nn.init.ones_(m.weight) #LayerNorm层权重初始化，使用全一

class PatchEmded(nn.Module):
    def __init__(self,img_size=224,patch_size=16,in_chans=3,embed_dim=768,norm_layer=None):
        #img_size:输入图像的大小
        #patch_size:每个patch的大小
        super().__init__()
        #将输入图像的大小和patch的大小转换为元组形式，以便后续计算。
        img_size = (img_size,img_size)
        patch_size = (patch_size,patch_size)
        self.img_size = img_size
        self.patch_size = patch_size
        self.grid_size = (img_size[0]//patch_size[0],img_size[1]//patch_size[1])
        self.num_patches = self.grid_size[0]*self.grid_size[1]
        #通过一个卷积层将输入图像划分为patches，并将每个patch映射到一个embed_dim维的向量空间中。
        self.proj = nn.Conv2d(in_chans,embed_dim,kernel_size=patch_size,stride=patch_size)
        self.norm = norm_layer(embed_dim) if norm_layer else nn.Identity()
        #nn.Identity()是一个占位符模块，不会对输入进行任何操作，直接返回输入本身。
    def forward(self,x):
        B,C,H,W = x.shape
        assert H == self.img_size[0] and W == self.img_size[1],f"Input image size ({H}*{W}) doesn't match model ({self.img_size[0]}*{self.img_size[1]})."
        #B,3,224,224 --> B,768,14,14 --> B,768,196 --> B,196,768
        x = self.proj(x).flatten(2).transpose(1,2)
        x = self.norm(x) #若有norm_layer则进行归一化处理，否则直接返回输入
        return x

class Attention(nn.Module):
    def __init__(self,
                 dim,#输入token的维度 768  
                 num_heads=8,#多头注意力机制中的头数
                 qkv_bias=False,#是否在qkv线性变换中添加偏置项
                 qk_scale=None,#缩放因子，默认为None表示使用sqrt(d_k)作为缩放因子
                 attn_drop_rate=0.,#注意力权重的dropout率
                 proj_drop_rate=0.#,#输出的dropout率
                 ):
        super().__init__()
        self.num_heads = num_heads #多头注意力机制中的头数
        head_dim = dim // num_heads #每个头的维度
        self.scale = qk_scale or head_dim ** -0.5 #缩放因子，默认为None表示使用sqrt(d_k)作为缩放因子
        self.qkv = nn.Linear(dim,dim*3,bias=qkv_bias) #线性变换层，用于生成查询、键和值的向量
        #(这是个大矩阵,将Wq,Wk,Wv合并成一个矩阵,输入是dim维,输出是3*dim维)
        self.attn_drop = nn.Dropout(attn_drop_rate) #注意力权重的dropout层
        self.proj_drop = nn.Dropout(proj_drop_rate) #输出的dropout层
        self.proj = nn.Linear(dim,dim) #线性变换层，用于将多头注意力的输出映射回输入维度
    def forward(self,x):
        B,N,C = x.shape #batch,num_patch+1,embed_dim 这个1是cls_token
        qkv = self.qkv(x).reshape(B,N,3,self.num_heads,C//self.num_heads).permute(2,0,3,1,4) 
        #qkv的形状为(B,N,3*dim) --> (B,N,3,num_heads,head_dim) --> (3,B,num_heads,N,head_dim)
        #现在这里我们有Q,K,V三个矩阵了,每个矩阵的形状都是(B,num_heads,N,head_dim)
        q,k,v = qkv[0],qkv[1],qkv[2] #分别取出Q,K,V矩阵
        attn = (q @ k.transpose(-2,-1)) * self.scale #计算注意力分数
        #形状为(B,num_heads,N,N)
        attn = attn.softmax(dim=-1) #对最后一个维度进行softmax操作，得到注意力权重
        attn = self.attn_drop(attn) #对注意力权重进行drop
        x = (attn @ v).transpose(1,2).reshape(B,N,C) #计算加权平均值
        #形状为(B,num_heads,N,head_dim) --> (B,N,num_heads,head_dim) --> (B,N,dim)
        x = self.proj(x) #通过线性变换层将多头注意力的输出映射回输入维度
        x = self.proj_drop(x) #对输出进行drop
        return x

class MLP(nn.Module):
    def __init__(self,in_features,hidden_features=None,out_features=None,act_layer=nn.GELU,drop=0.):
        #in_features:输入特征的维度 hidden_features:隐藏层特征的维度(通常为4倍于in_features) 
        # #out_features:输出特征的维度（通常与in_features相同） act_layer:激活函数 drop:dropout率
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features,hidden_features) #第一层线性变换
        self.act = act_layer() #激活函数
        self.fc2 = nn.Linear(hidden_features,out_features) #第二层线性变换
        self.drop = nn.Dropout(drop) #dropout层
    def forward(self,x):
        x = self.fc1(x) #通过第一层线性变换
        x = self.act(x) #通过激活函数
        x = self.drop(x) #通过dropout层
        x = self.fc2(x) #通过第二层线性变换
        x = self.drop(x) #通过dropout层
        return x
    
class Block(nn.Module):
    def __init__(self,
                 dim,#输入token的维度 768
                 num_heads=8,#多头注意力机制中的头数
                 mlp_ratio=4.,#MLP中隐藏层特征的维度与输入特征维度的比例
                 qkv_bias=False,#是否在qkv线性变换中添加偏置项
                 qk_scale=None,#缩放因子，默认为None表示使用sqrt(d_k)作为缩放因子
                 drop_rate=0.,#多头注意力最后使用的dropout率
                 attn_drop_rate=0.,#注意力权重的dropout率
                 drop_path_rate=0.,#stochastic depth的dropout率(直接跳过一整个block)
                 act_layer=nn.GELU,#激活函数
                 norm_layer=nn.LayerNorm#归一化层
                 ):
        super().__init__()
        self.norm1 = norm_layer(dim) #第一层归一化
        self.attn = Attention(dim,num_heads,qkv_bias=qkv_bias,qk_scale=qk_scale,
                              attn_drop_rate=attn_drop_rate,proj_drop_rate=drop_rate) #多头注意力层
        self.drop_path = DropPath(drop_path_rate) if drop_path_rate > 0. else nn.Identity() #stochastic depth层
        self.norm2 = norm_layer(dim) #第二层归一化
        mlp_hidden_dim = int(dim*mlp_ratio) #MLP中隐藏层特征的维度
        self.mlp = MLP(in_features=dim,hidden_features=mlp_hidden_dim,act_layer=act_layer,drop=drop_rate) #MLP层
    def forward(self,x):
        x = x + self.drop_path(self.attn(self.norm1(x))) #残差连接和stochastic depth
        x = x + self.drop_path(self.mlp(self.norm2(x))) #残差连接和stochastic depth
        return x

class VisionTransformer(nn.Module):
    def __init__(self,
                 img_size=224, #输入图像的大小
                 patch_size=16, #每个patch的大小
                 in_chans=3, #输入图像的通道数
                 num_classes=1000, #分类任务中的类别数
                 embed_dim=768, #每个patch的嵌入维度
                 depth=12, #Transformer块的数量
                 num_heads=8, #多头注意力机制中的头数
                 mlp_ratio=4., #MLP中隐藏层特征的维度与输入特征维度的比例
                 qkv_bias=False, #是否在qkv线性变换中添加偏置项
                 qk_scale=None, #缩放因子，默认为None表示使用sqrt(d_k)作为缩放因子
                 representation_size=None, #表示层的维度，如果为None则不使用表示层
                 distilled = False, #是否使用蒸馏token
                    drop_rate=0., #Transformer块中使用的dropout率
                    attn_drop_rate=0., #注意力权重的dropout率
                    drop_path_rate=0., #stochastic depth的dropout率(直接跳过一整个block)
                    embed_layer=PatchEmded, #用于生成patch嵌入的层
                    norm_layer=None, #归一化层
                    act_layer=None #激活函数
                    ):
        super().__init__()
        self.num_classes = num_classes
        self.num_features = self.embed_dim = embed_dim #特征维度等于嵌入维度
        self.num_tokens = 2 if distilled else 1 #如果使用蒸馏token，则总token数为2（cls_token和dist_token），否则为1（cls_token）
        norm_layer = norm_layer or partial(nn.LayerNorm,eps=1e-6) #默认使用LayerNorm作为归一化层，eps参数用于数值稳定性
        act_layer = act_layer or nn.GELU #默认使用GELU作为激活函数
        self.patch_embed = embed_layer(img_size=img_size,patch_size=patch_size,in_chans=in_chans,
                                       embed_dim=embed_dim,norm_layer=norm_layer) #用于生成patch嵌入的层
        num_patches = self.patch_embed.num_patches #计算patch的数量
        #分类token和蒸馏token的参数初始化 大小为(1,1,768)
        self.cls_token = nn.Parameter(torch.zeros(1,1,embed_dim)) #分类token的参数
        self.dist_token = nn.Parameter(torch.zeros(1,1,embed_dim)) if distilled else None #蒸馏token的参数
        #位置编码参数初始化 大小为(1,197 or 198,768)
        self.pos_embed = nn.Parameter(torch.zeros(1,num_patches + self.num_tokens,embed_dim)) #位置编码的参数
        self.pos_drop = nn.Dropout(p=drop_rate) #位置编码的dropout层
        #构建等差数列，表示每个block的drop_path_rate 越靠后drop越高
        dpr = [x.item() for x in torch.linspace(0,drop_path_rate,depth)] #stochastic depth的dropout率列表
        self.blocks = nn.Sequential(*[
            Block(
                dim=embed_dim,
                num_heads=num_heads,
                mlp_ratio=mlp_ratio,
                qkv_bias=qkv_bias,
                qk_scale=qk_scale,
                drop_rate=drop_rate,
                attn_drop_rate=attn_drop_rate,
                drop_path_rate=dpr[i],
                act_layer=act_layer,
                norm_layer=norm_layer
            ) for i in range(depth)
        ])
        self.norm = norm_layer(embed_dim) #Transformer块后的归一化层
        #分类头的定义，如果representation_size为None，则直接使用线性层将特征映射到类别数；
        #否则先通过一个全连接层和Tanh激活函数将特征映射到representation_size维，再通过一个线性层映射到类别数。
        if representation_size and not distilled:
            self.has_logits = True
            self.num_features = representation_size
            self.pre_logits = nn.Sequential(OrderedDict([
                ('fc',nn.Linear(embed_dim,representation_size)),
                ('act',nn.Tanh())
            ]))
        else:
            self.has_logits = False
            self.pre_logits = nn.Identity()
        #分类头和蒸馏头的定义，如果num_classes大于0，则使用线性层将特征映射到类别数；否则使用Identity层直接返回输入特征。
        self.head = nn.Linear(self.num_features,num_classes) if num_classes > 0 else nn.Identity() #分类头
        self.head_dist = None
        if distilled:
            self.head_dist = nn.Linear(self.num_features,num_classes) if num_classes > 0 else nn.Identity() #蒸馏头
        #参数初始化
        nn.init.trunc_normal_(self.pos_embed,std=0.02) #位置编码参数初始化，使用截断正态分布
        if self.dist_token is not None:
            nn.init.trunc_normal_(self.dist_token,std=0.02) #蒸馏token参数初始化，使用截断正态分布
        nn.init.trunc_normal_(self.cls_token,std=0.02) #分类token参数初始化，使用截断正态分布
        self.apply(_init_vit_weights) #对模型的所有参数进行初始化，使用_init_vit_weights函数
    def forward_features(self,x):
        #B,3,224,224 --> B,196,768
        x = self.patch_embed(x) #通过patch embedding层将输入图像转换为patch嵌入
        #1,1,768 --> B,1,768
        cls_token = self.cls_token.expand(x.shape[0],-1,-1) #将分类token扩展到与batch_size相同的维度
        #如果使用蒸馏token，则将分类token、蒸馏token和patch嵌入拼接在一起，形成输入序列；
        # 否则只将分类token和patch嵌入拼接在一起。
        if self.dist_token is None:
            x = torch.cat((cls_token,x),dim=1) #将分类token和patch嵌入拼接在一起，形成输入序列
        else:
            dist_token = self.dist_token.expand(x.shape[0],-1,-1) #将蒸馏token扩展到与batch_size相同的维度
            x = torch.cat((cls_token,dist_token,x),dim=1) #将分类token、蒸馏token和patch嵌入拼接在一起，形成输入序列
        x = self.pos_drop(x + self.pos_embed) #将位置编码添加到输入序列中，并通过dropout层进行处理
        x = self.blocks(x) #通过Transformer块进行处理
        x = self.norm(x) #通过归一化层进行处理
        if self.dist_token is None:
            return self.pre_logits(x[:,0]) #返回分类token对应的特征
        else:
            return x[:,0],x[:,1] #返回分类token和蒸馏token对应的特征
    def forward(self,x):
        x = self.forward_features(x) #获取特征
        #如果使用蒸馏token，则将分类头和蒸馏头分别应用于分类token和蒸馏token对应的特征，并返回它们的输出；
        # 否则只将分类头应用于分类token对应的特征，并返回输出。
        if self.head_dist is not None:
            x,x_dist = self.head(x[0]),self.head_dist(x[1])
            #如果模型处于训练模式且不是在TorchScript中进行脚本化，则返回分类头和蒸馏头的输出；否则只返回分类头的输出。
            if self.training and not torch.jit.is_scripting():
                return x,x_dist
        else:
            x = self.head(x)#将分类头应用于分类token对应的特征，并返回输出
        return x
    
def vit_patch16_224(num_classes:int = 1000,
                    pretrained = False):
    model = VisionTransformer(
        img_size=224,
        patch_size=16,
        in_chans=3,
        num_classes=num_classes,
        embed_dim=768,
        depth=12,
        num_heads=8,
        mlp_ratio=4.,
        qkv_bias=True,
        norm_layer=partial(nn.LayerNorm,eps=1e-6),
    )
    return model


