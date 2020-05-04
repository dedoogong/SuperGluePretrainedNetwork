import time
import torch

from .superpoint import SuperPoint
from .superglue import SuperGlue
from torch2trt import torch2trt
from torch2trt import TRTModule


def normalize_keypoints(kpts, image_shape):
    """ Normalize keypoints locations based on image image_shape"""
    _, _, height, width = image_shape
    one = kpts.new_tensor(1)
    size = torch.stack([one * width, one * height])[None]
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
    ms, ns = (m * one).to(scores), (n * one).to(scores)

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


class Matching(torch.nn.Module):
    """ Image Matching Frontend (SuperPoint + SuperGlue) """

    def __init__(self, config={}):
        super().__init__()
        self.superpoint = SuperPoint(config.get('superpoint', {}))
        self.superglue = SuperGlue(config.get('superglue', {}))
        # self.model_sg_trt = None
        self.convert_save_trt_model = 1
        if not self.convert_save_trt_model:
            self.model_sg_trt = TRTModule()
            self.model_sg_trt.load_state_dict(
                torch.load('superglue_trt_fixed_input_part_every_kpt_encoder_every_gnn_outputs.pth'))
            # 'superglue_trt_fixed_input_part_all_kpt_encoder_two_gnn_outputs.pth'))
            # 'superglue_trt_fixed_input_part_kpt_encoder_gnn_4outputs.pth'))
            # 'superglue_trt_fixed_input_part_kpt_encoder.pth')) #superglue_trt_fixed_input.pth'))

        self.bin_score = torch.nn.Parameter(torch.tensor(2.3457).cuda())

    def forward(self, data):
        """ Run SuperPoint (optionally) and SuperGlue
        SuperPoint is skipped if ['keypoints0', 'keypoints1'] exist in input
        Args:
          data: dictionary with minimal keys: ['image0', 'image1']
        """
        pred = {}

        # Extract SuperPoint (keypoints, scores, descriptors) if not provided
        if 'keypoints0' not in data:
            pred0 = self.superpoint({'image': data['image0']})
            pred = {**pred, **{k + '0': v for k, v in pred0.items()}}
        if 'keypoints1' not in data:
            pred1 = self.superpoint({'image': data['image1']})
            pred = {**pred, **{k + '1': v for k, v in pred1.items()}}
        t1 = time.time()
        # Batch all features
        # We should either have i) one image per batch, or
        # ii) the same number of local features for all images in the batch.
        data = {**data, **pred}

        for k in data:
            if isinstance(data[k], (list, tuple)):
                data[k] = torch.stack(data[k])

        # Perform the matching
        """Run SuperGlue on a pair of keypoints and descriptors"""
        desc0, desc1 = data['descriptors0'], data['descriptors1']
        kpts0, kpts1 = data['keypoints0'], data['keypoints1']

        # Keypoint normalization.
        kpts0 = normalize_keypoints(kpts0, data['image0'].shape)
        kpts1 = normalize_keypoints(kpts1, data['image1'].shape)

        inputs0 = [kpts0.transpose(1, 2), data['scores0'].unsqueeze(1)]
        kpts_scores0 = torch.cat(inputs0, dim=1)

        inputs1 = [kpts1.transpose(1, 2), data['scores1'].unsqueeze(1)]
        kpts_scores1 = torch.cat(inputs1, dim=1)
        max_count = 350
        len0 = max_count - desc0.shape[2]
        len1 = max_count - desc1.shape[2]
        desc0 = torch.cat((desc0, torch.zeros(1, 256, len0).cuda()), 2)
        desc1 = torch.cat((desc1, torch.zeros(1, 256, len1).cuda()), 2)
        kpts_scores0 = torch.cat((kpts_scores0, torch.zeros(1, 3, len0).cuda()), 2)
        kpts_scores1 = torch.cat((kpts_scores1, torch.zeros(1, 3, len1).cuda()), 2)

        if self.convert_save_trt_model:
            self.model_sg_trt = torch2trt(self.superglue, [kpts_scores0, desc0, kpts_scores1,
                                                           desc1])  # , input_names=['in1', 'in2', 'in3', 'in4'], output_names=['out1', 'out2'])
            # torch.save(self.model_sg_trt.state_dict(), 'superglue_trt_fixed_input_part_kpt_encoder_gnn_4outputs.pth') >> fail. different result!!
            # torch.save(self.model_sg_trt.state_dict(), 'superglue_trt_fixed_input_part_all_kpt_encoder_two_gnn_outputs.pth')
            torch.save(self.model_sg_trt.state_dict(),
                       'superglue_trt_fixed_input_part_every_kpt_encoder_every_gnn_outputs.pth')

        if 0:
            if 0:
                mdesc0_trt, mdesc1_trt = self.model_sg_trt(kpts_scores0, desc0, kpts_scores1, desc1)
            else:
                outputs = self.superglue(kpts_scores0, desc0, kpts_scores1, desc1)
                mdesc0, mdesc1 = outputs[-2], outputs[-1]
        else:
            '''
            1. superglue.py : forward, kpt init, forward , 
            2. matching.py forward,   
            3. model convert save -> comment TRT load in init. uncomment torch2trt
            4. model laod -> uncomment  TRT load in init. comment torch2trt

            # debug 1 : kpt encoder module => 100% same even after loading the saved converted model!
            conv1_1_trt, bn1_1_trt, relu1_1_trt, conv2_1_trt, bn2_1_trt, relu2_1_trt, conv3_1_trt, bn3_1_trt, relu3_1_trt, \
            conv4_1_trt, bn4_1_trt, relu4_1_trt, conv5_1_trt, \
            conv1_2_trt, bn1_2_trt, relu1_2_trt, conv2_2_trt, bn2_2_trt, relu2_2_trt, conv3_2_trt, bn3_2_trt, relu3_2_trt, \
            conv4_2_trt, bn4_2_trt, relu4_2_trt, conv5_2_trt, \
            desc0_trt, desc1_trt = self.model_sg_trt(kpts_scores0, desc0, kpts_scores1, desc1)

            conv1_1, bn1_1, relu1_1, conv2_1, bn2_1, relu2_1, conv3_1, bn3_1, relu3_1, conv4_1, bn4_1, relu4_1, conv5_1, \
            conv1_2, bn1_2, relu1_2, conv2_2, bn2_2, relu2_2, conv3_2, bn3_2, relu3_2, conv4_2, bn4_2, relu4_2, conv5_2, \
            desc0, desc1 = self.superglue(kpts_scores0, desc0, kpts_scores1, desc1)
            '''
            '''
            # debug 2 : kpt encoder module + gnn module => 100% same
            output_trts = self.model_sg_trt(kpts_scores0, desc0, kpts_scores1, desc1)
            output = self.superglue(kpts_scores0, desc0, kpts_scores1, desc1)
            '''
            '''
            # debug 3 : all kpt encoder outputs + 2 gnn output

            conv1_1, bn1_1, relu1_1, conv2_1, bn2_1, relu2_1, conv3_1, bn3_1, relu3_1, conv4_1, bn4_1, relu4_1, conv5_1,\
            conv1_2, bn1_2, relu1_2, conv2_2, bn2_2, relu2_2, conv3_2, bn3_2, relu3_2, conv4_2, bn4_2, relu4_2, conv5_2, \
            desc0, desc1,    
            desc0_gnn, desc1_gnn  
            '''
            output_trt = self.model_sg_trt(kpts_scores0, desc0, kpts_scores1, desc1)
            # output = self.superglue(kpts_scores0, desc0, kpts_scores1, desc1)

            # debug 4 : all kpt encoder outputs + all self/cross attention gnn outputs!!
            '''
            conv1_trt, bn1_trt, relu1_trt, \
            conv2_trt, bn2_trt, relu2_trt, \
            conv3_trt, bn3_trt, relu3_trt, \
            conv4_trt, bn4_trt, relu4_trt, \
            conv5_trt, _, _, _ = model_sg_trt(kpts_scores0, desc0, kpts_scores1, desc1)

            conv1, bn1, relu1, \
            conv2, bn2, relu2, \
            conv3, bn3, relu3, \
            conv4, bn4, relu4, \
            conv5, _, _, _ = self.superglue(kpts_scores0, desc0, kpts_scores1, desc1)
            '''
            # output_trt = self.model_sg_trt(kpts_scores0, desc0, kpts_scores1, desc1)
            # output = self.superglue(kpts_scores0, desc0, kpts_scores1, desc1)
            mdesc0, mdesc1 = output_trt[-2], output_trt[-1]

        if len0 > 0:
            mdesc0 = mdesc0[:, :, :-len0]
        if len1 > 0:
            mdesc1 = mdesc1[:, :, :-len1]
        # Compute matching descriptor distance.
        scores = torch.einsum('bdn,bdm->bnm', mdesc0, mdesc1)
        scores = scores / 256 ** .5

        # Run the optimal transport.
        scores = log_optimal_transport(
            scores, self.bin_score,
            iters=20)  # self.config['sinkhorn_iterations'])

        # Get the matches with score above "match_threshold".

        max0, max1 = scores[:, :-1, :-1].max(2), scores[:, :-1, :-1].max(1)
        indices0, indices1 = max0.indices, max1.indices
        mutual0 = arange_like(indices0, 1)[None] == indices1.gather(1, indices0)
        mutual1 = arange_like(indices1, 1)[None] == indices0.gather(1, indices1)
        zero = scores.new_tensor(0)
        mscores0 = torch.where(mutual0, max0.values.exp(), zero)
        mscores1 = torch.where(mutual1, mscores0.gather(1, indices1), zero)
        valid0 = mutual0 & (mscores0 > 0.0)  # self.config['match_threshold'])
        valid1 = mutual1 & valid0.gather(1, indices1)
        indices0 = torch.where(valid0, indices0, indices0.new_tensor(-1))
        indices1 = torch.where(valid1, indices1, indices1.new_tensor(-1))

        ret = {
            'matches0': indices0,  # use -1 for invalid match
            'matches1': indices1,  # use -1 for invalid match
            'matching_scores0': mscores0,
            'matching_scores1': mscores1,
        }
        pred = {**pred, **ret}
        print(1 / (time.time() - t1))
        return pred
