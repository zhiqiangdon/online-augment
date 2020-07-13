
import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class Linear(nn.Module):
    def __init__(self, linear_size, p_dropout=0.5):
        super(Linear, self).__init__()
        self.l_size = linear_size

        self.relu = nn.ReLU(inplace=True)
        self.dropout = nn.Dropout(p_dropout)

        self.w1 = nn.Linear(self.l_size, self.l_size)
        self.batch_norm1 = nn.BatchNorm1d(self.l_size)

        self.w2 = nn.Linear(self.l_size, self.l_size)
        self.batch_norm2 = nn.BatchNorm1d(self.l_size)

    def forward(self, x):
        y = self.w1(x)
        y = self.batch_norm1(y)
        y = self.relu(y)
        y = self.dropout(y)

        y = self.w2(y)
        y = self.batch_norm2(y)
        y = self.relu(y)
        y = self.dropout(y)

        out = x + y

        return out

class STN(nn.Module):
    # def __init__(self):
        # super(STN, self).__init__()
        # Spatial transformer localization-network
        # start_dim = 1
        # self.localization = nn.Sequential(
        #     nn.Linear(start_dim, start_dim*2),
        #     nn.LeakyReLU(0.2, True),
        #     # nn.Dropout(),
        #     nn.Linear(start_dim*2, start_dim*4),
        #     nn.LeakyReLU(0.2, True),
        #     # nn.Dropout(),
        #     nn.Linear(start_dim*4, start_dim*6),
        #     nn.LeakyReLU(0.2, True),
        #     # nn.Dropout(),
        #     nn.Linear(start_dim * 6, start_dim * 6),
        #     nn.LeakyReLU(0.2, True),
        #     # nn.Dropout(),
        #     nn.Linear(start_dim*6, start_dim*6),
        # )
        # Initialize the weights/bias with identity transformation
        # self.localization[-1].weight.data.zero_()
        # self.localization[-1].bias.data.copy_(torch.tensor([1, 0, 0, 0, 1, 0], dtype=torch.float))
    def __init__(self, input_size, output_size=6, linear_size=32,
                 num_stage=2, p_dropout=0.5):
        super(STN, self).__init__()
        # print('point 0')
        self.linear_size = linear_size
        print('linear_size: {}'.format(linear_size))
        self.p_dropout = p_dropout
        print('p_dropout: {}'.format(p_dropout))
        self.num_stage = num_stage
        print('num_stage: {}'.format(num_stage))

        # noise dim
        self.input_size = input_size
        print('theta generator input dim: {}'.format(self.input_size))
        # theta dim
        self.output_size = output_size

        # process input to linear size
        self.w1 = nn.Linear(self.input_size, self.linear_size)
        self.batch_norm1 = nn.BatchNorm1d(self.linear_size)

        self.linear_stages = []
        for l in range(num_stage):
            self.linear_stages.append(Linear(self.linear_size, self.p_dropout))
        self.linear_stages = nn.ModuleList(self.linear_stages)

        # post processing
        self.w2 = nn.Linear(self.linear_size, self.output_size)

        self.relu = nn.ReLU(inplace=True)
        self.dropout = nn.Dropout(self.p_dropout)

        # Initialize the weights/bias with identity transformation
        self.w2.weight.data.zero_()
        self.w2.bias.data.copy_(torch.tensor([1, 0, 0, 0, 1, 0], dtype=torch.float))
        # self.id_map = torch.tensor([[1, 0, 0, 0, 1, 0]], dtype=torch.float).cuda()
        self.id_map = torch.tensor([[1, 0, 0],
                                    [0, 1, 0],
                                    [0, 0, 1]], dtype=torch.float32).cuda()
        self.id_map_2 = torch.tensor([[[1, 0, 0],
                                    [0, 1, 0]]], dtype=torch.float32).cuda()
        self.pad_row = torch.tensor([[0., 0., 1.]], dtype=torch.float32).cuda()
        self.mse_loss = nn.MSELoss()

    # def div_loss(self, theta):
    #     # mean = torch.mean(theta, dim=0, keepdim=True)
    #     thrid_row = torch.tensor([[[0, 0, 1]]], dtype=torch.float32).cuda()
    #     tf_mats = torch.cat([theta, thrid_row.repeat(theta.size(0), 1, 1)], dim=1)
    #     tf_mat = torch.chain_matmul(*tf_mats)
    #     loss = self.loss(tf_mat, self.id_map)
    #     return loss

    def div_loss(self, theta):
        id_maps = self.id_map_2.repeat(theta.size(0), 1, 1)
        # print('id_maps shape: {}'.format(id_maps.size()))
        # print('theta shape: {}'.format(theta.size()))
        # exit()
        return self.mse_loss(theta, id_maps)

    # def cycle_loss(self, x, x_tf, theta):
    #     pad_rows = self.pad_row.repeat(x.size(0), 1, 1)
    #     theta_padded = torch.cat([theta, pad_rows], dim=1)
    #     theta_padded_inv = torch.inverse(theta_padded)
    #     theta_inv = theta_padded_inv[:, 0:-1, :]
    #     grid = F.affine_grid(theta_inv, x.size())
    #     x_recon = F.grid_sample(x_tf, grid)
    #     return self.mse_loss(x, x_recon)
    #
    # def cycle_loss_2(self, x, x_tf, theta_inv):
    #     grid = F.affine_grid(theta_inv, x_tf.size())
    #     x_recon = F.grid_sample(x_tf, grid)
    #     return self.mse_loss(x, x_recon)

    def inv_theta(self, theta):
        pad_rows = self.pad_row.repeat(theta.size(0), 1, 1)
        theta_padded = torch.cat([theta, pad_rows], dim=1)
        theta_padded_inv = torch.inverse(theta_padded)
        theta_inv = theta_padded_inv[:, 0:-1, :]
        return theta_inv

    def tf_func(self, x, theta):
        grid = F.affine_grid(theta, x.size())
        x_tf = F.grid_sample(x, grid)
        return x_tf

    def diversity_loss(self, input1, output1, input2, output2, eps):
        output_diff = F.mse_loss(output1, output2, reduction='none')
        assert output_diff.size() == output1.size()
        output_diff_vec = output_diff.view(output_diff.size(0), -1).mean(dim=1)
        assert len(output_diff_vec.size()) == 1
        # noise_diff_vec = F.l1_loss(noise, noise_2, reduction='none').sum(dim=1)
        input_diff_vec = F.mse_loss(input1, input2, reduction='none').mean(dim=1)
        assert len(input_diff_vec.size()) == 1
        loss = output_diff_vec / (input_diff_vec + eps)
        # loss = torch.clamp(loss, max=1)
        return loss.mean()

    # def cosine_similarity(self, x1, x2):
    #     similarity = F.cosine_similarity(x1.view(x1.size(0), -1),
    #                                      x2.view(x2.size(0), -1))
    #     # scale cosine similarity from [-1, 1] to [0, 1]
    #     return (similarity + 1.) / 2.
    #
    # def theta_cosine_diversity_loss(self, noise, theta, eps=1):
    #     assert noise.size(1) > 1
    #     noise_2 = torch.randn_like(noise)
    #     theta_2 = self.localization(noise_2)
    #     noise_similarity = self.cosine_similarity(noise, noise_2)
    #     # print('noise_similarity: {}'.format(noise_similarity))
    #     theta_similarity = self.cosine_similarity(theta, theta_2)
    #     # print('theta_similarity: {}'.format(theta_similarity))
    #     assert noise_similarity.size() == theta_similarity.size()
    #     assert noise_similarity.size(0) == noise.size(0)
    #
    #     # note the order of numerator and denominator
    #     # loss = theta_diff / (noise_diff + eps)
    #     # loss = noise_similarity / (theta_similarity + eps)
    #     loss = noise_similarity / theta_similarity
    #     print('noise_similarity: {}'.format(noise_similarity.mean()))
    #     print('theta_similarity: {}'.format(theta_similarity.mean()))
    #     # exit()
    #     return loss.mean()

    # def noise_mse_theta_cosine_diversity_loss(self, noise, theta, eps=10):
    #     noise_2 = torch.randn_like(noise)
    #     theta_2 = self.localization(noise_2)
    #     noise_diff = F.mse_loss(noise, noise_2, reduction='none').mean(dim=1)
    #     theta_similarity = self.cosine_similarity(theta, theta_2)
    #     assert noise_diff.size() == theta_similarity.size()
    #     assert noise_diff.size(0) == noise.size(0)
    #     loss = -theta_similarity / noise_diff
    #     print('noise_diff: {}'.format(noise_diff.mean()))
    #     print('theta_similarity: {}'.format(theta_similarity.mean()))
    #     print('theta_similarity[:10]: {}'.format(theta_similarity[:10]))
    #     print('noise_diff[:10]: {}'.format(noise_diff[:10]))
    #     print('loss: {}'.format(loss[:10]))
    #     return loss.mean()

    def theta_diversity_loss(self, noise, theta, eps=1e-3):
        # version 1
        # input = input.view(input.size(0), -1)
        # output = output.view(output.size(0), -1)
        # input_dist_mat = torch.cdist(input, input, p=1)
        # output_dist_mat = torch.cdist(output, output, p=1)
        # loss = torch.sum(output_dist_mat / (input_dist_mat+eps)) / (input.size(0)*(input.size(0)-1))

        # version 2
        noise_2 = torch.randn_like(noise)
        theta_2 = self.localization(noise_2)
        loss = self.diversity_loss(noise, theta, noise_2, theta_2, eps)
        return loss

    def img_diversity_loss(self, x, x_tf, noise, eps=1e-1):
        noise_2 = torch.randn_like(noise)
        theta_2 = self.localization(noise_2)
        x_tf_2 = self.tf_func(x, theta_2)

        # version 1
        # # img_diff = F.l1_loss(x_tf, x_tf_2, reduction='none')
        # img_diff = F.mse_loss(x_tf, x_tf_2, reduction='none')
        # assert img_diff.size() == x.size()
        # img_diff_vec = img_diff.view(img_diff.size(0), -1).mean(dim=1)
        # assert len(img_diff_vec.size()) == 1
        # # noise_diff_vec = F.l1_loss(noise, noise_2, reduction='none').sum(dim=1)
        # noise_diff_vec = F.mse_loss(noise, noise_2, reduction='none').mean(dim=1)
        #
        # assert len(noise_diff_vec.size()) == 1
        # loss = img_diff_vec / (noise_diff_vec + eps)
        # # loss = torch.clamp(loss, max=1)
        # return loss.mean()

        # version 2
        loss = self.diversity_loss(noise, x_tf, noise_2, x_tf_2, eps)
        return loss

    def localization(self, noise):
        # pre-processing
        y = self.w1(noise)
        y = self.batch_norm1(y)
        y = self.relu(y)
        y = self.dropout(y)

        # linear layers
        for i in range(self.num_stage):
            y = self.linear_stages[i](y)

        theta = self.w2(y)
        theta = theta.view(-1, 2, 3)

        return theta

    def forward(self, noise, x, label, require_loss=False, require_grid=False):
        theta = self.localization(noise)
        assert theta.size(0) == x.size(0)
        theta_inv = self.inv_theta(theta)
        assert theta_inv.size() == theta.size()
        # grid = F.affine_grid(theta, x.size())
        # x_tf = F.grid_sample(x, grid)
        x_tf = self.tf_func(x, theta)
        x_tf_inv = self.tf_func(x, theta_inv)

        # idxs = torch.randperm(x.size(0)).cuda()
        # half_size = int(x.size(0) / 2)
        # x_tf_half = torch.index_select(x_tf, 0, idxs[:half_size])
        # x_tf_inv_half = torch.index_select(x_tf_inv, 0, idxs[half_size:])
        # x_comb = torch.cat([x_tf_half, x_tf_inv_half], dim=0)

        # get the transformed x and its corresponding label
        x_comb = torch.cat([x_tf, x_tf_inv], dim=0)
        label_comb = torch.cat([label, label], dim=0)

        if not require_loss:
            if require_grid:
                return x_comb, label_comb, grid
            else:
                return x_comb, label_comb
        else:
            # return x_tf, self.div_loss(theta)
            # return x_tf, self.cycle_loss(x, x_tf, theta)

            # reconstruct x from theta tf
            x_tf_recon = self.tf_func(x_tf, theta_inv)

            # reconstruct x from inverse theta tf
            x_tf_inv_recon = self.tf_func(x_tf_inv, theta)



            return x_comb, label_comb, \
                   self.mse_loss(x, x_tf_recon)+self.mse_loss(x, x_tf_inv_recon), \
                   self.img_diversity_loss(x, x_tf, noise)
# self.theta_cosine_diversity_loss(noise, theta)
# self.theta_diversity_loss(noise, theta)