import torch.nn as nn
from layers import *


class PixelCNNLayer_up(nn.Module):
    def __init__(self, nr_resnet, nr_filters, resnet_nonlinearity):
        super(PixelCNNLayer_up, self).__init__()
        self.nr_resnet = nr_resnet
        # stream from pixels above
        self.u_stream = nn.ModuleList([gated_resnet(nr_filters, down_shifted_conv2d,
                                        resnet_nonlinearity, skip_connection=0)
                                            for _ in range(nr_resnet)])

        # stream from pixels above and to thes left
        self.ul_stream = nn.ModuleList([gated_resnet(nr_filters, down_right_shifted_conv2d,
                                        resnet_nonlinearity, skip_connection=1)
                                            for _ in range(nr_resnet)])

    def forward(self, u, ul):
        u_list, ul_list = [], []

        for i in range(self.nr_resnet):
            u  = self.u_stream[i](u)
            ul = self.ul_stream[i](ul, a=u)
            u_list  += [u]
            ul_list += [ul]

        return u_list, ul_list


class PixelCNNLayer_down(nn.Module):
    def __init__(self, nr_resnet, nr_filters, resnet_nonlinearity):
        super(PixelCNNLayer_down, self).__init__()
        self.nr_resnet = nr_resnet
        # stream from pixels above
        self.u_stream  = nn.ModuleList([gated_resnet(nr_filters, down_shifted_conv2d,
                                        resnet_nonlinearity, skip_connection=1)
                                            for _ in range(nr_resnet)])

        # stream from pixels above and to thes left
        self.ul_stream = nn.ModuleList([gated_resnet(nr_filters, down_right_shifted_conv2d,
                                        resnet_nonlinearity, skip_connection=2)
                                            for _ in range(nr_resnet)])

    def forward(self, u, ul, u_list, ul_list):
        for i in range(self.nr_resnet):
            u  = self.u_stream[i](u, a=u_list.pop())
            ul = self.ul_stream[i](ul, a=torch.cat((u, ul_list.pop()), 1))

        return u, ul

class AbsolutePositionalEncoding(nn.Module):
    # Taken from PA2, absolute positional encoding class to fuse class labels here
    def __init__(self, num_classes, d_model):
        super().__init__()
        self.W = nn.Parameter(torch.empty((num_classes, d_model)))
        nn.init.normal_(self.W)

    def forward(self, x):
        """
        args:
            x: shape B x D
        returns:
            out: shape B x D
        START BLOCK
        """
        B, D = x.shape
        out = torch.add(x, self.W[:B, :D].to(x.device))
        """
        END BLOCK
        """
        return out


class PixelCNN(nn.Module):
    MAX_LEN = 256
    APE_DIM = 32
    def __init__(self, nr_resnet=5, nr_filters=80, nr_logistic_mix=10,
                    resnet_nonlinearity='concat_elu', input_channels=3):
        super(PixelCNN, self).__init__()
        if resnet_nonlinearity == 'concat_elu' :
            self.resnet_nonlinearity = lambda x : concat_elu(x)
        else :
            raise Exception('right now only concat elu is supported as resnet nonlinearity.')

        self.nr_filters = nr_filters
        self.input_channels = input_channels
        self.nr_logistic_mix = nr_logistic_mix
        self.right_shift_pad = nn.ZeroPad2d((1, 0, 0, 0))
        self.down_shift_pad  = nn.ZeroPad2d((0, 0, 1, 0))

        down_nr_resnet = [nr_resnet] + [nr_resnet + 1] * 2
        self.down_layers = nn.ModuleList([PixelCNNLayer_down(down_nr_resnet[i], nr_filters,
                                                self.resnet_nonlinearity) for i in range(3)])

        self.up_layers   = nn.ModuleList([PixelCNNLayer_up(nr_resnet, nr_filters,
                                                self.resnet_nonlinearity) for _ in range(3)])

        self.downsize_u_stream  = nn.ModuleList([down_shifted_conv2d(nr_filters, nr_filters,
                                                    stride=(2,2)) for _ in range(2)])

        self.downsize_ul_stream = nn.ModuleList([down_right_shifted_conv2d(nr_filters,
                                                    nr_filters, stride=(2,2)) for _ in range(2)])

        self.upsize_u_stream  = nn.ModuleList([down_shifted_deconv2d(nr_filters, nr_filters,
                                                    stride=(2,2)) for _ in range(2)])

        self.upsize_ul_stream = nn.ModuleList([down_right_shifted_deconv2d(nr_filters,
                                                    nr_filters, stride=(2,2)) for _ in range(2)])

        self.u_init = down_shifted_conv2d(input_channels + 1, nr_filters, filter_size=(2,3),
                        shift_output_down=True)

        self.ul_init = nn.ModuleList([down_shifted_conv2d(input_channels + 1, nr_filters,
                                            filter_size=(1,3), shift_output_down=True),
                                       down_right_shifted_conv2d(input_channels + 1, nr_filters,
                                            filter_size=(2,1), shift_output_right=True)])

        num_mix = 3 if self.input_channels == 1 else 10
        self.nin_out = nin(nr_filters, num_mix * nr_logistic_mix)
        self.init_padding = None

        self.pos_encoding = AbsolutePositionalEncoding(self.MAX_LEN, self.nr_filters)


    def forward(self, x, labels, sample=False):
        # I think I need labels added here, so I can fuse them according to this: https://piazza.com/class/lqypkqwt2v84ky/post/374
        # TODO: Fuse labels somehow - Piazza and TAs mention APE from Transformer assignment 2

        # print(f"SHAPE OF X AT INPPUT: {x.shape}")
        # Early fusing can be done before anything else in forward:
        # _, D, _, _ = x.shape
        # one_hot_labels = self.one_hot_labels(labels, D)
        # one_hot_labels = one_hot_labels.to(x.device)

        # embeddings = self.pos_encoding(one_hot_labels)
        # print(f"SHAPE OF X AT INPPUT: {x.shape}")


        # similar as done in the tf repo :
        if self.init_padding is not sample:
            xs = [int(y) for y in x.size()]
            padding = Variable(torch.ones(xs[0], 1, xs[2], xs[3]), requires_grad=False)
            self.init_padding = padding.cuda() if x.is_cuda else padding.to(x.device)

        if sample :
            xs = [int(y) for y in x.size()]
            padding = Variable(torch.ones(xs[0], 1, xs[2], xs[3]), requires_grad=False)
            padding = padding.cuda() if x.is_cuda else padding.to(x.device)
            x = torch.cat((x, padding), 1)

        ###      UP PASS    ###
        x = x if sample else torch.cat((x, self.init_padding), 1)
        u_list  = [self.u_init(x)]
        ul_list = [self.ul_init[0](x) + self.ul_init[1](x)]
        for i in range(3):
            # resnet block
            u_out, ul_out = self.up_layers[i](u_list[-1], ul_list[-1])
            u_list  += u_out
            ul_list += ul_out

            if i != 2:
                # downscale (only twice)
                u_list  += [self.downsize_u_stream[i](u_list[-1])]
                ul_list += [self.downsize_ul_stream[i](ul_list[-1])]

        # print(f"PRE EMBD SHAPE U_LIST: {u_list[0].shape}")

        # Seems like u_list and ul_list have shapes [? * B * D * H * W] according to https://piazza.com/class/lqypkqwt2v84ky/post/301
        _, D, _, _ = u_list[0].shape
        one_hot_labels = self.one_hot_labels(labels, D)
        one_hot_labels = one_hot_labels.to(x.device)
        # print(f"SHAPE OF ONE HOT LABEL: {one_hot_labels.shape}")
        
        embeddings = self.pos_encoding(one_hot_labels).unsqueeze(-1).unsqueeze(-1)
        # print(f"SHAPE OF EMB AT MIDDLE: {embeddings.shape}")
        
        # Middle fusing here, TA says pytorch does broadcasting according to Piazza post @301 but that seems to change
        # the length of the u_list and ul_list :( try using iteration

        self.add_embedding_to_u_ul(u_list, embeddings)
        self.add_embedding_to_u_ul(ul_list, embeddings)

        # print(f"POST EMBD SHAPE U_LIST: {u_list[0].shape}")

        ###    DOWN PASS    ###
        u  = u_list.pop()
        ul = ul_list.pop()

        for i in range(3):
            # resnet block
            u, ul = self.down_layers[i](u, ul, u_list, ul_list)

            # upscale (only twice)
            if i != 2 :
                u  = self.upsize_u_stream[i](u)
                ul = self.upsize_ul_stream[i](ul)

        x_out = self.nin_out(F.elu(ul))

        assert len(u_list) == len(ul_list) == 0, pdb.set_trace()

        return x_out
    
    def one_hot_labels(self, labels, dim):
        # Encodes labels in one-hot encoding format so that they have shape BxD
        # labels has shape (B,) and dim represents the dimension D and output has shape (B, D)

        one_hot = torch.zeros(labels.size(0), dim)
        for batch_idx in range(labels.size(0)):
            one_hot[batch_idx, labels[batch_idx]] = 1

        return one_hot
    
    def add_embedding_to_u_ul(self, tensor_list, embedding_tensor):
        for i in range(len(tensor_list)):
            # Add the embedding tensor to the current tensor in-place
            tensor_list[i] += embedding_tensor


class random_classifier(nn.Module):
    def __init__(self, NUM_CLASSES):
        super(random_classifier, self).__init__()
        self.NUM_CLASSES = NUM_CLASSES
        self.fc = nn.Linear(3, NUM_CLASSES)
        print("Random classifier initialized")
        # create a folder
        if 'models' not in os.listdir():
            os.mkdir('models')
        torch.save(self.state_dict(), 'models/conditional_pixelcnn.pth')
    def forward(self, x, device):
        return torch.randint(0, self.NUM_CLASSES, (x.shape[0],)).to(device)
    