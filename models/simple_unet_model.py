import torch
from .base_model import BaseModel, accuracy
from . import networks


class SimpleUnetModel(BaseModel):
    def name(self):
        return 'SimpleUnetModel'

    @staticmethod
    def modify_commandline_options(parser, is_train=True):
        return parser

    def initialize(self, opt):
        BaseModel.initialize(self, opt)
        self.isTrain = opt.isTrain
        # specify the training losses you want to print out. The program will call base_model.get_current_losses
        self.loss_names = ['G'] # self.loss_G
        # specify the images you want to save/display. The program will call base_model.get_current_visuals
        self.visual_names = ['data','mask_gt','pred_mask']  
        # specify the models you want to save to the disk. The program will call base_model.save_networks and base_model.load_networks
        self.model_names = ['G'] # self.netG

        # load/define networks
        self.netG = networks.define_G(3, 1, opt.ngf, 
                              "unet_256", opt.norm, not opt.no_dropout, opt.init_type, opt.init_gain, self.gpu_ids)
        if self.isTrain:
            # define loss functions
            self.criterion = torch.nn.BCELoss().to(self.device)

            # initialize optimizers
            self.optimizers = []
            self.optimizer_G = torch.optim.Adam(self.netG.parameters(),
                                                lr=opt.lr, betas=(opt.beta1, 0.999))
            self.optimizers.append(self.optimizer_G)

    def set_input(self, input):
        self.data = input['data'].to(self.device)
        self.mask_gt = input['gt'].to(self.device)

    def forward(self):
        self.pred_mask = self.netG(self.data)
        # self.pred_mask = self.netG(self.data)

    def backward_G(self):
        output = self.pred_mask
        output = torch.sigmoid(output)
        gt = self.mask_gt
        loss = self.criterion(output, gt)

        # Combined loss
        self.loss_G = loss

        self.loss_G.backward()

    def optimize_parameters(self):
        self.forward()
        # update D
        self.set_requires_grad(self.netG, True)
        self.optimizer_G.zero_grad()
        self.backward_G()
        self.optimizer_G.step()


    def print_metrics(self):
        print("Loss: %.3f"%(self._get_loss()))

    def _get_loss(self):
        return float(self.loss_G.cpu().detach().numpy())
