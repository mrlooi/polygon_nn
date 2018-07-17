import torch
import torchvision.models as models
from .base_model import BaseModel, accuracy
from . import networks


class ResnetClassifierModel(BaseModel):
    def name(self):
        return 'ResnetClassifierModel'

    @staticmethod
    def modify_commandline_options(parser, is_train=True):
        return parser

    def initialize(self, opt):
        BaseModel.initialize(self, opt)
        self.isTrain = opt.isTrain
        # specify the training losses you want to print out. The program will call base_model.get_current_losses
        self.loss_names = ['D'] # self.loss_D
        # specify the images you want to save/display. The program will call base_model.get_current_visuals
        self.visual_names = ['data']  # self.data
        # specify the models you want to save to the disk. The program will call base_model.save_networks and base_model.load_networks
        self.model_names = ['D'] # self.netD

        # load/define networks
        self.netD = networks.define_D("resnet18", opt.num_classes, opt.init_type, opt.init_gain, self.gpu_ids)
        if self.isTrain:
            # define loss functions
            self.criterion = torch.nn.CrossEntropyLoss().to(self.device)

            # initialize optimizers
            self.optimizers = []
            self.optimizer_D = torch.optim.Adam(self.netD.parameters(),
                                                lr=opt.lr, betas=(opt.beta1, 0.999))
            self.optimizers.append(self.optimizer_D)


    def set_input(self, input):
        self.data = input['data'].to(self.device)
        self.labels = input['labels'].to(self.device)
        self.image_paths = input['paths']

    def forward(self):
        self.pred_labels = self.netD(self.data)

    def backward_D(self):
        output = self.pred_labels
        target = self.labels
        loss = self.criterion(output, target)

        # Combined loss
        self.loss_D = loss

        self.loss_D.backward()

    def optimize_parameters(self):
        self.forward()
        # update D
        self.set_requires_grad(self.netD, True)
        self.optimizer_D.zero_grad()
        self.backward_D()
        self.optimizer_D.step()

    def get_accuracy(self):
        return accuracy(self.pred_labels, self.labels, topk=(1,))