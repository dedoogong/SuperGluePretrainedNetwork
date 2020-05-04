from copy import deepcopy
from pathlib import Path
import torch
from torch import nn

def normalize_keypoints(kpts, image_shape):
    """ Normalize keypoints locations based on image image_shape"""
    _, _, height, width = image_shape
    one = kpts.new_tensor(1)
    size = torch.stack([one*width, one*height])[None]
    center = size / 2
    scaling = size.max(1, keepdim=True).values * 0.7
    return (kpts - center[:, None, :]) / scaling[:, None, :]

def log_sinkhorn_iterations(Z, log_mu, log_nu, iters: int):
    """ Perform Sinkhorn Normalization in Log-space for stability"""
    u, v = torch.zeros_like(log_mu), torch.zeros_like(log_nu)
    for _ in range(iters):
        u = log_mu - torch.logsumexp(Z + v.unsqueeze(1), dim=2)
        v = log_nu - torch.logsumexp(Z + u.unsqueeze(2), dim=1)
    return Z + u.unsqueeze(2) + v.unsqueeze(1)

def log_optimal_transport(scores, alpha, iters: int):
    """ Perform Differentiable Optimal Transport in Log-space for stability"""
    b, m, n = scores.shape
    one = scores.new_tensor(1)
    ms, ns = (m*one).to(scores), (n*one).to(scores)

    bins0 = alpha.expand(b, m, 1)
    bins1 = alpha.expand(b, 1, n)
    alpha = alpha.expand(b, 1, 1)

    couplings = torch.cat([torch.cat([scores, bins0], -1),
                           torch.cat([bins1, alpha], -1)], 1)

    norm = - (ms + ns).log()
    log_mu = torch.cat([norm.expand(m), ns.log()[None] + norm])
    log_nu = torch.cat([norm.expand(n), ms.log()[None] + norm])
    log_mu, log_nu = log_mu[None].expand(b, -1), log_nu[None].expand(b, -1)

    Z = log_sinkhorn_iterations(couplings, log_mu, log_nu, iters)
    Z = Z - norm  # multiply probabilities by M+N
    return Z

def arange_like(x, dim: int):
    return x.new_ones(x.shape[dim]).cumsum(0) - 1  # traceable in 1.1

def MLP(channels: list, do_bn=True):
    """ Multi-layer perceptron """
    n = len(channels)
    layers = []
    for i in range(1, n):
        layers.append(
            nn.Conv1d(channels[i - 1], channels[i], kernel_size=1, bias=True))
        if i < (n-1):
            if do_bn:
                layers.append(nn.BatchNorm1d(channels[i]))
            layers.append(nn.ReLU())
    return nn.Sequential(*layers)

class KeypointEncoder(nn.Module):
    """ Joint encoding of visual appearance and location using MLPs"""
    def __init__(self, feature_dim, layers):
        super().__init__()
        self.encoder = MLP([3] + layers + [feature_dim])
        nn.init.constant_(self.encoder[-1].bias, 0.0)

    def forward2(self, inputs):# 아!! 일일이 아웃풋으로 해서 안바뀌니 퓨전 안된걸수도!! 아니면 여기까진 아웃폿 안해도 되나
        return self.encoder(inputs)
    # forward_kpt
    def forward(self, inputs):# 아!! 일일이 아웃풋으로 해서 안바뀌니 퓨전 안된걸수도!! 아니면 여기까진 아웃폿 안해도 되나
        #return self.encoder(inputs)

        conv1 = self.encoder[0](inputs)
        bn1   = self.encoder[1](conv1)
        relu1 = self.encoder[2](bn1)

        conv2 = self.encoder[3](relu1)
        bn2   = self.encoder[4](conv2)
        relu2 = self.encoder[5](bn2)

        conv3 = self.encoder[6](relu2)
        bn3   = self.encoder[7](conv3)
        relu3 = self.encoder[8](bn3)

        conv4 = self.encoder[9](relu3)
        bn4   = self.encoder[10](conv4)
        relu4 = self.encoder[11](bn4)

        conv5 = self.encoder[12](relu4)

        return conv1, bn1, relu1, conv2, bn2, relu2,  conv3, bn3, relu3,  conv4, bn4, relu4,  conv5
        ''''''

        #ret = self.encoder(inputs)
        #return ret #self.encoder(inputs)

def attention(query, key, value):
    dim = query.shape[1]
    scores = torch.einsum('bdhn,bdhm->bhnm', query, key) / dim**.5
    prob = torch.nn.functional.softmax(scores, dim=3)
    return torch.einsum('bhnm,bdhm->bdhn', prob, value), prob

class MultiHeadedAttention(nn.Module):
    """ Multi-head attention to increase model expressivitiy """
    def __init__(self, num_heads: int, d_model: int):
        super().__init__()
        assert d_model % num_heads == 0
        self.dim = d_model // num_heads
        self.num_heads = num_heads
        self.merge = nn.Conv1d(d_model, d_model, kernel_size=1)
        self.proj = nn.ModuleList([deepcopy(self.merge) for _ in range(3)])

    def forward(self, query, key, value):
        batch_dim = query.size(0)
        query, key, value = [l(x).view(batch_dim, self.dim, self.num_heads, -1)
                             for l, x in zip(self.proj, (query, key, value))]
        x, prob = attention(query, key, value)
        self.prob.append(prob)
        return self.merge(x.contiguous().view(batch_dim, self.dim*self.num_heads, -1))

class AttentionalPropagation(nn.Module):
    def __init__(self, feature_dim: int, num_heads: int):
        super().__init__()
        self.attn = MultiHeadedAttention(num_heads, feature_dim)
        self.mlp = MLP([feature_dim*2, feature_dim*2, feature_dim])
        nn.init.constant_(self.mlp[-1].bias, 0.0)

    def forward(self, x, source):
        message = self.attn(x, source, source)
        return self.mlp(torch.cat([x, message], dim=1))

class AttentionalGNN(nn.Module):
    def __init__(self, feature_dim: int, layer_names: list):
        super().__init__()
        self.layers = nn.ModuleList([
            AttentionalPropagation(feature_dim, 4)
            for _ in range(len(layer_names))])
        self.names = layer_names

    def forward_orig(self, desc0, desc1):
        #output_dummy=None
        output_dummy=desc0
        for layer, name in zip(self.layers, self.names):
            layer.attn.prob = []
            if name == 'cross':
                src0, src1 = desc1, desc0
            else:  # if name == 'self':
                src0, src1 = desc0, desc1
            delta0, delta1 = layer(desc0, src0), layer(desc1, src1)
            desc0, desc1 = (desc0 + delta0), (desc1 + delta1)
            output_dummy+=desc0+desc1
        #output_tensors=torch.cat(outputs)
        return desc0, desc1, output_dummy

    def forward(self, desc0, desc1):
        # output_dummy=None
        layer = self.layers[0]
        layer.attn.prob = []
        src0, src1 = desc0, desc1
        delta0, delta1 = layer(desc0, src0), layer(desc1, src1)
        desc_0, desc_1 = (desc0 + delta0), (desc1 + delta1)

        layer = self.layers[1]
        layer.attn.prob = []
        src0, src1 = desc_1, desc_0
        delta0, delta1 = layer(desc_0, src0), layer(desc_1, src1)
        desc2, desc3 = (desc0 + delta0), (desc1 + delta1)

        layer = self.layers[2]
        layer.attn.prob = []
        src0, src1 = desc2, desc3
        delta0, delta1 = layer(desc2, src0), layer(desc3, src1)
        desc4, desc5 = (desc2 + delta0), (desc3 + delta1)

        layer = self.layers[3]
        layer.attn.prob = []
        src0, src1 = desc5, desc4
        delta0, delta1 = layer(desc4, src0), layer(desc5, src1)
        desc6, desc7 = (desc4 + delta0), (desc5 + delta1)

        layer = self.layers[4]
        layer.attn.prob = []
        src0, src1 = desc6, desc7
        delta0, delta1 = layer(desc6, src0), layer(desc7, src1)
        desc8, desc9 = (desc6 + delta0), (desc7 + delta1)

        layer = self.layers[5]
        layer.attn.prob = []
        src0, src1 = desc9, desc8
        delta0, delta1 = layer(desc8, src0), layer(desc9, src1)
        desc10, desc11 = (desc8 + delta0), (desc9 + delta1)

        layer = self.layers[6]
        layer.attn.prob = []
        src0, src1 = desc10,desc11
        delta0, delta1 = layer(desc10, src0), layer(desc11, src1)
        desc12, desc13 = (desc10 + delta0), (desc11 + delta1)

        layer = self.layers[7]
        layer.attn.prob = []
        src0, src1 = desc13,desc12
        delta0, delta1 = layer(desc12, src0), layer(desc13, src1)
        desc14, desc15 = (desc12 + delta0), (desc13 + delta1)

        layer = self.layers[8]
        layer.attn.prob = []
        src0, src1 = desc14,desc15
        delta0, delta1 = layer(desc14, src0), layer(desc15, src1)
        desc16, desc17 = (desc14 + delta0), (desc15 + delta1)

        layer = self.layers[9]
        layer.attn.prob = []
        src0, src1 = desc17,desc16
        delta0, delta1 = layer(desc16, src0), layer(desc17, src1)
        desc18, desc19 = (desc16 + delta0), (desc17 + delta1)

        layer = self.layers[10]
        layer.attn.prob = []
        src0, src1 = desc18,desc19
        delta0, delta1 = layer(desc18, src0), layer(desc19, src1)
        desc20, desc21 = (desc18 + delta0), (desc19 + delta1)

        layer = self.layers[11]
        layer.attn.prob = []
        src0, src1 = desc21,desc20
        delta0, delta1 = layer(desc20, src0), layer(desc21, src1)
        desc22, desc23 = (desc20 + delta0), (desc21 + delta1)

        layer = self.layers[12]
        layer.attn.prob = []
        src0, src1 = desc22,desc23
        delta0, delta1 = layer(desc22, src0), layer(desc23, src1)
        desc24, desc25 = (desc22 + delta0), (desc23 + delta1)

        layer = self.layers[13]
        layer.attn.prob = []
        src0, src1 = desc25,desc24
        delta0, delta1 = layer(desc24, src0), layer(desc25, src1)
        desc26, desc27 = (desc24 + delta0), (desc25 + delta1)

        layer = self.layers[14]
        layer.attn.prob = []
        src0, src1 = desc26,desc27
        delta0, delta1 = layer(desc26, src0), layer(desc27, src1)
        desc28, desc29 = (desc26 + delta0), (desc27 + delta1)

        layer = self.layers[15]
        layer.attn.prob = []
        src0, src1 = desc29,desc28
        delta0, delta1 = layer(desc28, src0), layer(desc29, src1)
        desc30, desc31 = (desc28 + delta0), (desc29 + delta1)

        layer = self.layers[16]
        layer.attn.prob = []
        src0, src1 = desc30,desc31
        delta0, delta1 = layer(desc30, src0), layer(desc31, src1)
        desc32, desc33 = (desc30 + delta0), (desc31 + delta1)

        layer = self.layers[17]
        layer.attn.prob = []
        src0, src1 = desc33, desc32
        delta0, delta1 = layer(desc32, src0), layer(desc33, src1)
        desc34, desc35 = (desc32 + delta0), (desc33 + delta1)

        # output_tensors=torch.cat(outputs)
        return  desc_0, desc_1, \
                desc0, desc1,\
                desc2, desc3, \
                desc4, desc5, \
                desc6, desc7, \
                desc8, desc9, \
                desc10, desc11, \
                desc12, desc13, \
                desc14, desc15, \
                desc16, desc17, \
                desc18, desc19, \
                desc20, desc21, \
                desc22, desc23, \
                desc24, desc25, \
                desc26, desc27, \
                desc28, desc29, \
                desc30, desc31, \
                desc32, desc33, \
                desc34, desc35

        '''
         outputs[0], outputs[1],\
         outputs[2], outputs[3],\
         outputs[4], outputs[5],\
         outputs[6], outputs[7],\
         outputs[8], outputs[9],\
         outputs[10], outputs[11],\
         outputs[12], outputs[13],\
         outputs[14], outputs[15],\
         outputs[16], outputs[17],\
         outputs[18], outputs[19],\
         outputs[20], outputs[21],\
         outputs[22], outputs[23],\
         outputs[24], outputs[25],\
         outputs[26], outputs[27],\
         outputs[28], outputs[29],\
         outputs[30], outputs[31],\
         outputs[32], outputs[33],\
         outputs[34], outputs[35]
         '''
        # what if only conv or conv+bn or conv+bn+relu or cont....many comb outputs?
class SuperGlue(nn.Module):
    """SuperGlue feature matching middle-end

    Given two sets of keypoints and locations, we determine the
    correspondences by:
      1. Keypoint Encoding (normalization + visual feature and location fusion)
      2. Graph Neural Network with multiple self and cross-attention layers
      3. Final projection layer
      4. Optimal Transport Layer (a differentiable Hungarian matching algorithm)
      5. Thresholding matrix based on mutual exclusivity and a match_threshold

    The correspondence ids use -1 to indicate non-matching points.

    Paul-Edouard Sarlin, Daniel DeTone, Tomasz Malisiewicz, and Andrew
    Rabinovich. SuperGlue: Learning Feature Matching with Graph Neural
    Networks. In CVPR, 2020. https://arxiv.org/abs/1911.11763

    """
    default_config = {
        'descriptor_dim': 256,
        'weights': 'indoor',
        'keypoint_encoder': [32, 64, 128, 256],
        'GNN_layers': ['self', 'cross'] * 9,
        'sinkhorn_iterations': 100,
        'match_threshold': 0.2,
    }

    def __init__(self, config):
        super().__init__()
        self.config = {**self.default_config, **config}

        self.kenc = KeypointEncoder(
            self.config['descriptor_dim'], self.config['keypoint_encoder'])

        self.gnn = AttentionalGNN(
            self.config['descriptor_dim'], self.config['GNN_layers'])

        self.final_proj = nn.Conv1d(
            self.config['descriptor_dim'], self.config['descriptor_dim'],
            kernel_size=1, bias=True)

        bin_score = torch.nn.Parameter(torch.tensor(1.))
        self.register_parameter('bin_score', bin_score)

        assert self.config['weights'] in ['indoor', 'outdoor']
        path = Path(__file__).parent
        path = path / 'weights/superglue_{}.pth'.format(self.config['weights'])
        self.load_state_dict(torch.load(path))
        print('Loaded SuperGlue model (\"{}\" weights)'.format(
            self.config['weights']))
    #kpt_only
    def forward_1(self, kpts_scores0, desc0, kpts_scores1, desc1):

        conv1_1, bn1_1, relu1_1, conv2_1, bn2_1, relu2_1, conv3_1, bn3_1, relu3_1, conv4_1, bn4_1, relu4_1, conv5_1 = self.kenc(kpts_scores0)
        desc0 = desc0 + conv5_1
        conv1_2, bn1_2, relu1_2, conv2_2, bn2_2, relu2_2, conv3_2, bn3_2, relu3_2, conv4_2, bn4_2, relu4_2, conv5_2 = self.kenc(kpts_scores1)
        desc1 = desc1 + conv5_2

        return conv1_1, bn1_1, relu1_1, conv2_1, bn2_1, relu2_1, conv3_1, bn3_1, relu3_1, conv4_1, bn4_1, relu4_1, conv5_1, \
               conv1_2, bn1_2, relu1_2, conv2_2, bn2_2, relu2_2, conv3_2, bn3_2, relu3_2, conv4_2, bn4_2, relu4_2, conv5_2, \
               desc0, desc1

    # all kpt+gnn
    def forward_3(self, kpts_scores0, desc0, kpts_scores1, desc1):
        # Keypoint MLP encoder.
        conv1_1, bn1_1, relu1_1, conv2_1, bn2_1, relu2_1, conv3_1, bn3_1, relu3_1, conv4_1, bn4_1, relu4_1, conv5_1 = self.kenc(
            kpts_scores0)
        desc0 = desc0 + conv5_1
        conv1_2, bn1_2, relu1_2, conv2_2, bn2_2, relu2_2, conv3_2, bn3_2, relu3_2, conv4_2, bn4_2, relu4_2, conv5_2 = self.kenc(
            kpts_scores1)
        desc1 = desc1 + conv5_2
        # Multi-layer Transformer network.
        desc0_gnn, desc1_gnn, \
        outputs_0, outputs_1, \
        outputs_2, outputs_3, \
        outputs_4, outputs_5, \
        outputs_6, outputs_7, \
        outputs_8, outputs_9, \
        outputs_10, outputs_11, \
        outputs_12, outputs_13, \
        outputs_14, outputs_15, \
        outputs_16, outputs_17, \
        outputs_18, outputs_19, \
        outputs_20, outputs_21, \
        outputs_22, outputs_23, \
        outputs_24, outputs_25, \
        outputs_26, outputs_27, \
        outputs_28, outputs_29, \
        outputs_30, outputs_31, \
        outputs_32, outputs_33, \
        outputs_34, outputs_35    = self.gnn(desc0, desc1)

        return conv1_1, bn1_1, relu1_1, conv2_1, bn2_1, relu2_1, conv3_1, bn3_1, relu3_1, conv4_1, bn4_1, relu4_1, conv5_1,\
               conv1_2, bn1_2, relu1_2, conv2_2, bn2_2, relu2_2, conv3_2, bn3_2, relu3_2, conv4_2, bn4_2, relu4_2, conv5_2, \
               desc0, desc1, desc0_gnn, desc1_gnn

    #2 kpt + 2 gnn
    def forward_2(self, kpts_scores0, desc0, kpts_scores1, desc1):

        # Keypoint MLP encoder.
        desc0 = desc0 + self.kenc(kpts_scores0)
        desc1 = desc1 + self.kenc(kpts_scores1)
        # Multi-layer Transformer network.
        desc0_, desc1_ = self.gnn(desc0, desc1)
        return desc0, desc1, desc0_, desc1_


    # all kpt+ all gnn syntax list output error
    def forward_3(self, kpts_scores0, desc0, kpts_scores1, desc1):
        # Keypoint MLP encoder.
        conv1_1, bn1_1, relu1_1, conv2_1, bn2_1, relu2_1, conv3_1, bn3_1, relu3_1, conv4_1, bn4_1, relu4_1, conv5_1 = self.kenc(
            kpts_scores0)
        desc0 = desc0 + conv5_1
        conv1_2, bn1_2, relu1_2, conv2_2, bn2_2, relu2_2, conv3_2, bn3_2, relu3_2, conv4_2, bn4_2, relu4_2, conv5_2 = self.kenc(
            kpts_scores1)
        desc1 = desc1 + conv5_2
        # Multi-layer Transformer network.
        desc0_gnn, desc1_gnn, output_dummy = self.gnn(desc0, desc1)
        '''
        outputs_0, outputs_1, \
        outputs_2, outputs_3, \
        outputs_4, outputs_5, \
        outputs_6, outputs_7, \
        outputs_8, outputs_9, \
        outputs_10, outputs_11, \
        outputs_12, outputs_13, \
        outputs_14, outputs_15, \
        outputs_16, outputs_17, \
        outputs_18, outputs_19, \
        outputs_20, outputs_21, \
        outputs_22, outputs_23, \
        outputs_24, outputs_25, \
        outputs_26, outputs_27, \
        outputs_28, outputs_29, \
        outputs_30, outputs_31, \
        outputs_32, outputs_33, \
        outputs_34, outputs_35 '''

        return conv1_1, bn1_1, relu1_1, conv2_1, bn2_1, relu2_1, conv3_1, bn3_1, relu3_1, conv4_1, bn4_1, relu4_1, conv5_1, \
               conv1_2, bn1_2, relu1_2, conv2_2, bn2_2, relu2_2, conv3_2, bn3_2, relu3_2, conv4_2, bn4_2, relu4_2, conv5_2, \
               desc0, desc1, desc0_gnn, desc1_gnn, output_dummy
        '''
        outputs_0, outputs_1, \
        outputs_2, outputs_3, \
        outputs_4, outputs_5, \
        outputs_6, outputs_7, \
        outputs_8, outputs_9, \
        outputs_10, outputs_11, \
        outputs_12, outputs_13, \
        outputs_14, outputs_15, \
        outputs_16, outputs_17, \
        outputs_18, outputs_19, \
        outputs_20, outputs_21, \
        outputs_22, outputs_23, \
        outputs_24, outputs_25, \
        outputs_26, outputs_27, \
        outputs_28, outputs_29, \
        outputs_30, outputs_31, \
        outputs_32, outputs_33, \
        outputs_34, outputs_35
        '''

    def forward_4(self, kpts_scores0, desc0, kpts_scores1, desc1):
        # Keypoint MLP encoder.
        conv1_1, bn1_1, relu1_1, conv2_1, bn2_1, relu2_1, conv3_1, bn3_1, relu3_1, conv4_1, bn4_1, relu4_1, conv5_1 = self.kenc(
            kpts_scores0)
        desc0 = desc0 + conv5_1
        conv1_2, bn1_2, relu1_2, conv2_2, bn2_2, relu2_2, conv3_2, bn3_2, relu3_2, conv4_2, bn4_2, relu4_2, conv5_2 = self.kenc(
            kpts_scores1)
        desc1 = desc1 + conv5_2
        # Multi-layer Transformer network.
        desc0, desc1, \
        desc2, desc3, \
        desc4, desc5, \
        desc6, desc7, \
        desc8, desc9, \
        desc10, desc11, \
        desc12, desc13, \
        desc14, desc15, \
        desc16, desc17, \
        desc18, desc19, \
        desc20, desc21, \
        desc22, desc23, \
        desc24, desc25, \
        desc26, desc27, \
        desc28, desc29, \
        desc30, desc31, \
        desc32, desc33, \
        desc34, desc35 = self.gnn(desc0, desc1)
        '''
        outputs_0, outputs_1, \
        outputs_2, outputs_3, \
        outputs_4, outputs_5, \
        outputs_6, outputs_7, \
        outputs_8, outputs_9, \
        outputs_10, outputs_11, \
        outputs_12, outputs_13, \
        outputs_14, outputs_15, \
        outputs_16, outputs_17, \
        outputs_18, outputs_19, \
        outputs_20, outputs_21, \
        outputs_22, outputs_23, \
        outputs_24, outputs_25, \
        outputs_26, outputs_27, \
        outputs_28, outputs_29, \
        outputs_30, outputs_31, \
        outputs_32, outputs_33, \
        outputs_34, outputs_35 '''
        return conv1_1, bn1_1, relu1_1, conv2_1, bn2_1, relu2_1, conv3_1, bn3_1, relu3_1, conv4_1, bn4_1, relu4_1, conv5_1, \
               conv1_2, bn1_2, relu1_2, conv2_2, bn2_2, relu2_2, conv3_2, bn3_2, relu3_2, conv4_2, bn4_2, relu4_2, conv5_2, \
               desc0, desc1, \
               desc2, desc3, \
               desc4, desc5, \
               desc6, desc7, \
               desc8, desc9, \
               desc10, desc11, \
               desc12, desc13, \
               desc14, desc15, \
               desc16, desc17, \
               desc18, desc19, \
               desc20, desc21, \
               desc22, desc23, \
               desc24, desc25, \
               desc26, desc27, \
               desc28, desc29, \
               desc30, desc31, \
               desc32, desc33, \
               desc34, desc35

    def forward(self, kpts_scores0, desc0, kpts_scores1, desc1):
        # Keypoint MLP encoder.
        conv1_1, bn1_1, relu1_1, conv2_1, bn2_1, relu2_1, conv3_1, bn3_1, relu3_1, conv4_1, bn4_1, relu4_1, conv5_1 = self.kenc(
            kpts_scores0)
        desc0 = desc0 + conv5_1
        conv1_2, bn1_2, relu1_2, conv2_2, bn2_2, relu2_2, conv3_2, bn3_2, relu3_2, conv4_2, bn4_2, relu4_2, conv5_2 = self.kenc(
            kpts_scores1)
        desc1 = desc1 + conv5_2
        # Multi-layer Transformer network.
        desc_0, desc_1, \
        desc0, desc1, \
        desc2, desc3, \
        desc4, desc5, \
        desc6, desc7, \
        desc8, desc9, \
        desc10, desc11, \
        desc12, desc13, \
        desc14, desc15, \
        desc16, desc17, \
        desc18, desc19, \
        desc20, desc21, \
        desc22, desc23, \
        desc24, desc25, \
        desc26, desc27, \
        desc28, desc29, \
        desc30, desc31, \
        desc32, desc33, \
        desc34, desc35 = self.gnn(desc0, desc1)
        # Final MLP projection.
        mdesc0, mdesc1 = self.final_proj(desc34), self.final_proj(desc35)

        return conv1_1, bn1_1, relu1_1, conv2_1, bn2_1, relu2_1, conv3_1, bn3_1, relu3_1, conv4_1, bn4_1, relu4_1, conv5_1, \
               conv1_2, bn1_2, relu1_2, conv2_2, bn2_2, relu2_2, conv3_2, bn3_2, relu3_2, conv4_2, bn4_2, relu4_2, conv5_2, \
               desc_0, desc_1, \
               desc0, desc1, \
               desc2, desc3, \
               desc4, desc5, \
               desc6, desc7, \
               desc8, desc9, \
               desc10, desc11, \
               desc12, desc13, \
               desc14, desc15, \
               desc16, desc17, \
               desc18, desc19, \
               desc20, desc21, \
               desc22, desc23, \
               desc24, desc25, \
               desc26, desc27, \
               desc28, desc29, \
               desc30, desc31, \
               desc32, desc33, \
               desc34, desc35, \
               mdesc0, mdesc1