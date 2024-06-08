import torch
from torch.nn import Linear, BatchNorm1d, ReLU
import numpy as np
import sparsemax


def initialize_non_glu(module, input_dim, output_dim):
    gain_value = np.sqrt((input_dim + output_dim) / np.sqrt(4 * input_dim))
    torch.nn.init.xavier_normal_(module.weight, gain=gain_value)
    # torch.nn.init.zeros_(module.bias)
    return


def initialize_glu(module, input_dim, output_dim):
    gain_value = np.sqrt((input_dim + output_dim) / np.sqrt(input_dim))
    torch.nn.init.xavier_normal_(module.weight, gain=gain_value)
    # torch.nn.init.zeros_(module.bias)
    return


class TabNet(torch.nn.Module):
    def __init__(
        self,
        input_dim,
        output_dim,
        model_seg=None,
        n_d=8,
        n_a=8,
        n_steps=3,
        gamma=1.3,
        cat_idxs=[],
        cat_dims=[],
        cat_emb_dim=1,
        n_independent=2,
        n_shared=2,
        epsilon=1e-15,
        virtual_batch_size=128,
        momentum=0.02,
        mask_type="sparsemax",
        group_attention_matrix=[],
    ):
        """
        Defines TabNet network

        Parameters
        ----------
        input_dim : int
            Initial number of features
        output_dim : int
            Dimension of network output
            examples : one for regression, 2 for binary classification etc...
        n_d : int
            Dimension of the prediction  layer (usually between 4 and 64)
        n_a : int
            Dimension of the attention  layer (usually between 4 and 64)
        n_steps : int
            Number of successive steps in the network (usually between 3 and 10)
        gamma : float
            Float above 1, scaling factor for attention updates (usually between 1.0 to 2.0)
        cat_idxs : list of int
            Index of each categorical column in the dataset
        cat_dims : list of int
            Number of categories in each categorical column
        cat_emb_dim : int or list of int
            Size of the embedding of categorical features
            if int, all categorical features will have same embedding size
            if list of int, every corresponding feature will have specific size
        n_independent : int
            Number of independent GLU layer in each GLU block (default 2)
        n_shared : int
            Number of independent GLU layer in each GLU block (default 2)
        epsilon : float
            Avoid log(0), this should be kept very low
        virtual_batch_size : int
            Batch size for Ghost Batch Normalization
        momentum : float
            Float value between 0 and 1 which will be used for momentum in all batch norm
        mask_type : str
            Either "sparsemax" or "entmax" : this is the masking function to use
        group_attention_matrix : torch matrix
            Matrix of size (n_groups, input_dim), m_ij = importance within group i of feature j
        """
        super(TabNet, self).__init__()
        self.cat_idxs = cat_idxs or []
        self.cat_dims = cat_dims or []
        self.cat_emb_dim = cat_emb_dim
        self.model_seg = model_seg
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.n_d = n_d
        self.n_a = n_a
        self.n_steps = n_steps
        self.gamma = gamma
        self.epsilon = epsilon
        self.n_independent = n_independent
        self.n_shared = n_shared
        self.mask_type = mask_type

        if self.n_steps <= 0:
            raise ValueError("n_steps should be a positive integer.")
        if self.n_independent == 0 and self.n_shared == 0:
            raise ValueError("n_shared and n_independent can't be both zero.")

        self.virtual_batch_size = virtual_batch_size
        self.embedder = EmbeddingGenerator(input_dim,
                                           cat_dims,
                                           cat_idxs,
                                           cat_emb_dim,
                                           group_attention_matrix)
        self.post_embed_dim = self.embedder.post_embed_dim
        #self.final_convolution = torch.nn.Conv3d(self.model_seg.base_width,1, kernel_size=1, stride=1, bias=False),

        self.final_mapping = torch.nn.Linear(32768, 3, bias=False)
        initialize_non_glu(self.final_mapping, 32768, 3)
        # self.final_mapping_2 = torch.nn.Linear(6, 3, bias=False)
        # initialize_non_glu(self.final_mapping_2, 6, 3)

        self.final_mapping1 = torch.nn.Linear(6, 3, bias=False)
        initialize_non_glu(self.final_mapping1, 2, 1)
        self.final_mapping2 = torch.nn.Linear(72, 36, bias=False)
        initialize_non_glu(self.final_mapping2, 72, 36)
        self.final_mapping3 = torch.nn.Linear(38, 3, bias=False)
        initialize_non_glu(self.final_mapping3, 38, 3)
        #self.dp = torch.nn.Dropout(0.8)
        self.final_bn1 = torch.nn.BatchNorm1d(32768)
        self.final_bn2 = torch.nn.BatchNorm1d(3)
        # self.conv2 = torch.nn.Sequential(  # for channel
        #     torch.nn.Conv1d(512,1,1, bias=False),
        #     torch.nn.BatchNorm1d(1),
        #     torch.nn.LeakyReLU()
        # )
        self.conv_cat = torch.nn.Sequential(
            torch.nn.Linear(6, 3),
            torch.nn.Dropout(0),
            torch.nn.BatchNorm1d(3),
            torch.nn.LeakyReLU()
        )

        self.conv_cat_attn = torch.nn.Sequential(
            torch.nn.Linear(6, 2),
            torch.nn.Dropout(0),
            torch.nn.BatchNorm1d(2),
            torch.nn.LeakyReLU()
        )



        self.softmax = torch.nn.Softmax(dim=1)

    def forward(self, img,x):
        x = self.embedder(x)
        x = self.final_mapping2(x)
        x = self.final_mapping3(x)

        feature = self.model_seg.encoder(img)
        img = self.model_seg.decoder(feature)
        img = self.model_seg.final_convolution(img)
        if self.model_seg.activation is not None:
            img = self.model_seg.activation(img)

        #no fuse
        #cls_x = x[0]

        #add
        # fl_seg_x = torch.flatten(img,1,-1)
        # fl_seg_x = self.final_mapping(fl_seg_x)
        # cls_x = fl_seg_x + x[0]
        #return cls_x, cls_x, x[1]

        #concatx[0]
        # m_loss = 0
        # fl_seg_x = torch.flatten(feature[0], 1, -1)
        # fl_seg_x = self.final_bn1(fl_seg_x)
        # fl_seg_x = self.final_mapping(fl_seg_x)
        # #fl_seg_x = self.dp(fl_seg_x)
        # fl_seg_x = self.final_bn2(fl_seg_x)
        # cls_x = torch.concat((fl_seg_x,x),dim=1)
        # cls_x = self.conv2(cls_x)
        # return img, cls_x, m_loss

        # concat x[0] attn
        m_loss = 0
        fl_seg_x = torch.flatten(feature[0], 1, -1)
        fl_seg_x = self.final_bn1(fl_seg_x)
        fl_seg_x = self.final_mapping(fl_seg_x)
        fl_seg_x = self.final_bn2(fl_seg_x)
        feat_fusion = torch.concat((fl_seg_x,x),dim=1)
        feat_fusion = self.conv_cat_attn(feat_fusion)
        feat_attn = self.softmax(feat_fusion)
        att_1 = feat_attn[:, 0].unsqueeze(1)
        att_2 = feat_attn[:, 1].unsqueeze(1)
        #cls_x = att_1*fl_seg_x+att_2*x
        cls_x = torch.concat((att_1*fl_seg_x,att_2*x),dim=1)
        cls_x = self.final_mapping1(cls_x)
        return img, cls_x, m_loss

        # # eins sum no channel best
        # cls_x = torch.einsum("bchwd,bk->bk",feature[0],x)
        # cls_x = self.conv_ein_no_channel(cls_x)
        # return img,cls_x,0

    def forward_masks(self, x):
        x = self.embedder(x)
        # return self.tabnet.forward_masks(x)
        return self.final_mapping2(x)


class EmbeddingGenerator(torch.nn.Module):
    """
    Classical embeddings generator
    """

    def __init__(self, input_dim, cat_dims, cat_idxs, cat_emb_dims, group_matrix):
        """This is an embedding module for an entire set of features

        Parameters
        ----------
        input_dim : int
            Number of features coming as input (number of columns)
        cat_dims : list of int
            Number of modalities for each categorial features
            If the list is empty, no embeddings will be done
        cat_idxs : list of int
            Positional index for each categorical features in inputs
        cat_emb_dim : list of int
            Embedding dimension for each categorical features
            If int, the same embedding dimension will be used for all categorical features
        group_matrix : torch matrix
            Original group matrix before embeddings
        """
        super(EmbeddingGenerator, self).__init__()

        if cat_dims == [] and cat_idxs == []:
            self.skip_embedding = True
            self.post_embed_dim = input_dim
            self.embedding_group_matrix = group_matrix.to(group_matrix.device)
            return
        else:
            self.skip_embedding = False

        self.post_embed_dim = int(input_dim + np.sum(cat_emb_dims) - len(cat_emb_dims))

        self.embeddings = torch.nn.ModuleList()

        for cat_dim, emb_dim in zip(cat_dims, cat_emb_dims):
            self.embeddings.append(torch.nn.Embedding(cat_dim, emb_dim))

        # record continuous indices
        self.continuous_idx = torch.ones(input_dim, dtype=torch.bool)
        self.continuous_idx[cat_idxs] = 0

        # update group matrix
        n_groups = group_matrix.shape[0]
        self.embedding_group_matrix = torch.empty((n_groups, self.post_embed_dim),
                                                  device=group_matrix.device)
        for group_idx in range(n_groups):
            post_emb_idx = 0
            cat_feat_counter = 0
            for init_feat_idx in range(input_dim):
                if self.continuous_idx[init_feat_idx] == 1:
                    # this means that no embedding is applied to this column
                    self.embedding_group_matrix[group_idx, post_emb_idx] = group_matrix[group_idx, init_feat_idx]  # noqa
                    post_emb_idx += 1
                else:
                    # this is a categorical feature which creates multiple embeddings
                    n_embeddings = cat_emb_dims[cat_feat_counter]
                    self.embedding_group_matrix[group_idx, post_emb_idx:post_emb_idx+n_embeddings] = group_matrix[group_idx, init_feat_idx] / n_embeddings  # noqa
                    post_emb_idx += n_embeddings
                    cat_feat_counter += 1

    def forward(self, x):
        """
        Apply embeddings to inputs
        Inputs should be (batch_size, input_dim)
        Outputs will be of size (batch_size, self.post_embed_dim)
        """
        if self.skip_embedding:
            # no embeddings required
            return x

        cols = []
        cat_feat_counter = 0
        for feat_init_idx, is_continuous in enumerate(self.continuous_idx):
            # Enumerate through continuous idx boolean mask to apply embeddings
            if is_continuous:
                cols.append(x[:, feat_init_idx].float().view(-1, 1))
            else:
                cols.append(
                    self.embeddings[cat_feat_counter](x[:, feat_init_idx].long())
                )
                cat_feat_counter += 1
        # concat
        post_embeddings = torch.cat(cols, dim=1)
        return post_embeddings
