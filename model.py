import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F

from sei_body import SeiBody
from selene_sdk.utils import NonStrandSpecific
from torch.nn.utils import weight_norm, spectral_norm


class BelugaBody(nn.Module):
    def __init__(self, sequence_length, model_name=''):

        super(BelugaBody, self).__init__()
        self.backbone = nn.Sequential(
            nn.Conv2d(4, 320, (1, 8)),
            nn.ReLU(),
            nn.Conv2d(320, 320, (1, 8)),
            nn.ReLU(),
            nn.Dropout(0.2),

            nn.MaxPool2d((1, 4), (1, 4)),
            nn.Conv2d(320, 480, (1, 8)),
            nn.ReLU(),
            nn.Conv2d(480, 480, (1, 8)),
            nn.ReLU(),
            nn.Dropout(0.2),

            nn.MaxPool2d((1, 4), (1, 4)),
            nn.Conv2d(480, 640, (1, 8)),
            nn.ReLU(),
            nn.Conv2d(640, 640, (1, 8)),
            nn.ReLU(),

            # added a new layer
            nn.MaxPool2d((1, 4), (1, 4)),
            nn.Conv2d(640, 640, (1, 8)),
            nn.ReLU(),
            nn.Conv2d(640, 640, (1, 8)),
            nn.ReLU(),

            # # added +2nd layer
            nn.MaxPool2d((1, 4), (1, 4)),
            nn.Conv2d(640, 640, (1, 4)),
            nn.ReLU(),
        )

        # pass in a test vector to see what we expect to get into linear
        test_tensor = torch.Tensor(np.zeros((1, 4, 1, sequence_length)))
        shape_test = self.backbone(test_tensor).shape
        self.seq_input = shape_test[1]*shape_test[-1]

    def forward(self, x):
        x = torch.unsqueeze(x, 2)
        return self.backbone(x)


class XpressoBody(nn.Module):
    def __init__(self, sequence_length=0, model_name=''):
        super(XpressoBody, self).__init__()
        self.backbone = nn.Sequential(
            nn.Conv1d(4, 128, kernel_size=6, padding='same', 
                         dilation=1),
            nn.ReLU(),
            nn.MaxPool1d(30),
            nn.Conv1d(128, 32, kernel_size=9, padding='same', 
                         dilation=1),
            nn.ReLU(),
            nn.MaxPool1d(10),
        )


    def forward(self, x):
        #x = torch.unsqueeze(x, 2)
        return self.backbone(x)



class ModelContrastive(nn.Module):
    """Wrapper class around backbone models for contrastive"""

    def __init__(self, sequence_length, final_mlp_size=128,
                 model_type='Albert',
                 model_name="", hidden_size=2048):
        super().__init__()
        if model_type == 'Beluga':
            self.backbone = BelugaBody(sequence_length,
                                       model_name=model_name)

        if model_type == 'Xpresso':
            self.backbone = XpressoBody(sequence_length,
                                  model_name=model_name)

        if model_type == 'Sei':
            self.backbone = NonStrandSpecific(SeiBody(sequence_length,
                                                      model_name=model_name))

        if model_type == 'Beluga' or model_type == 'Experimental':
            seq_input = self.backbone.seq_input
        elif model_type == 'Sei':
            seq_input = self.backbone.model.seq_input
        elif model_type == 'Xpresso':
            seq_input = 1120
        else:
            seq_input = 4*self.backbone.get_model_hidden_size()

        self.contr_layer = nn.Sequential(nn.Linear(seq_input, final_mlp_size),
                                         nn.ReLU(inplace=True),
                                         nn.Linear(final_mlp_size,
                                                   final_mlp_size),
                                         )

    def forward(self, x):
        x1 = self.backbone(x)
        x2 = x1.flatten(1)  # flatten x4 dimension
        x2 = self.contr_layer(x2)
        return x2, x1


def load_checkpoint(checkpoint, cpu=False):
    print('load_checkpoint() function')
    if cpu:
        checkpoint_weights = torch.load(
            checkpoint, map_location=torch.device('cpu'))
    else:
        checkpoint_weights = torch.load(checkpoint)
    state_dict = checkpoint_weights
    if 'state_dict' in checkpoint_weights:
        state_dict = checkpoint_weights['state_dict']

    # the weights are saves w/ backbone.backbone.backbone.<name>, need to remove that
    for k in list(state_dict.keys()):
        if "finetune_layer" in k:
            new_key = k.replace("model.", "")
        else:
            new_key = k.replace(
                "model.backbone.backbone.", "backbone.")
            new_key = k.replace(
                "backbone.backbone.backbone.", "backbone.")
            new_key = k.replace("backbone.backbone.", "")
            new_key = k.replace('module.', '')
        state_dict[new_key] = state_dict.pop(k)
    print(state_dict[new_key])
    return state_dict


class ModelChromProfiling(nn.Module):
    """Wrapper class around backbone models to finetune.
    Warning: checkpoint loading is for the pretrain loading, not the finetuning
    """

    def __init__(self, num_classes=2002,
                 sequence_length=256, model_type='Albert',
                 model_name="", hidden_size=2048, checkpoint=None, cpu=False):
        super().__init__()
        self.sequence_length = sequence_length
        if model_type == 'Beluga':
            backbone = BelugaBody(sequence_length,
                                  model_name=model_name)
            
        if model_type == 'Xpresso':
            backbone = XpressoBody(sequence_length,
                                  model_name=model_name)
            
        if model_type == 'Sei':
            print(num_classes)
            backbone = NonStrandSpecific(
                SeiBody(sequence_length, model_name=model_name))

        if checkpoint and len(checkpoint) > 2:
            checkpoint_weights = load_checkpoint(checkpoint)
            state_dict = checkpoint_weights
            if 'state_dict' in checkpoint_weights:
                state_dict = checkpoint_weights['state_dict']
            try:
                backbone.load_state_dict(state_dict, strict=True)
            except:
                backbone.load_state_dict(state_dict, strict=False)
            
            print('Checkpoint was specified')
            print(checkpoint)
            print(list(backbone.parameters())[0])

        self.backbone = backbone

        if model_type == 'Beluga':
            seq_input = self.backbone.seq_input
        elif model_type == 'Sei':
            seq_input = self.backbone.model.seq_input
        else:
            seq_input = self.sequence_length*self.backbone.get_model_hidden_size()

        self.finetune_layer = nn.Sequential(
            nn.Linear(seq_input, num_classes),
            nn.ReLU(),
            nn.Dropout(p=0.2),
            nn.Linear(num_classes, num_classes),
            nn.Sigmoid()
        )

    def forward(self, x):
        x = self.backbone(x)
        x = x.flatten(1)  # flatten x4 dimension
        x = self.finetune_layer(x)
        return x







class ModelExpression(nn.Module):
    """Wrapper class around backbone models to finetune.
    Warning: checkpoint loading is for the pretrain loading, not the finetuning

    TODO: additional_len is not passed as a parameter
    """

    def __init__(self, num_classes=54,
                 sequence_length=256, model_type='Albert',
                 model_name="", hidden_size=2048, checkpoint=None, cpu=False,
                 additional_len = 0, 
                 shift_noncoding=False, effective_length=10500):
        super().__init__()

        # set seed for the init
        torch.manual_seed(3)

        self.shift_noncoding = shift_noncoding
        self.effective_length = effective_length

        self.sequence_length = sequence_length

        if model_type == 'Beluga':
            backbone = BelugaBody(sequence_length,
                                  model_name=model_name)
            
            
        if model_type == 'Xpresso':
            backbone = XpressoBody(sequence_length,
                                       model_name=model_name)

        if model_type == 'Sei':
            backbone = NonStrandSpecific(
                SeiBody(sequence_length, model_name=model_name))

        if checkpoint and len(checkpoint) > 2:
            print(checkpoint)
            checkpoint_weights = load_checkpoint(checkpoint, cpu=cpu)
            state_dict = checkpoint_weights
            if 'state_dict' in checkpoint_weights:
                state_dict = checkpoint_weights['state_dict']
            try:
                print('****Backbone load : True', backbone.load_state_dict(state_dict, strict=True))
                backbone.load_state_dict(state_dict, strict=True)
            except:
                print('****Backbone load : False', backbone.load_state_dict(state_dict, strict=False))
                backbone.load_state_dict(state_dict, strict=False)

            print('Checkpoint was specified')
            print(list(backbone.parameters())[0])

        self.backbone = backbone

        if model_type == 'Beluga' or model_type=='Experimental':
            seq_input = self.backbone.seq_input + additional_len
        elif model_type == 'Sei':
            seq_input = self.backbone.model.seq_input + additional_len
        elif model_type == 'Xpresso':
            seq_input = -1
        else:
            seq_input = self.sequence_length*self.backbone.get_model_hidden_size()


        if model_type == 'Xpresso':
            self.finetune_layer = nn.Sequential(
                nn.Linear(1120+additional_len, 64),
                nn.ReLU(),
                nn.Dropout(p=0.00099), 
                
                nn.Linear(64, 2),
                nn.ReLU(),
                nn.Dropout(p=0.01546),
                
                nn.Linear(2, 1),
            )

        else:

            # add finetune layers
            fc1 = nn.Linear(seq_input, hidden_size)

            fc2 = nn.Linear(hidden_size, num_classes)

            self.finetune_layer = nn.Sequential(
                fc1,
                nn.BatchNorm1d(hidden_size),
                nn.ReLU(),
                nn.Dropout(p=0.2),
                fc2,
            )



    def forward(self, x, additional=None):
        if self.shift_noncoding:
            step_forward = self.sequence_length-self.effective_length
            x = x[:,:,step_forward:]
        x = self.backbone(x)
        x = x.flatten(1)  # flatten
        if additional is not None:
            x = torch.cat((x, additional.float()), 1)
        x = self.finetune_layer(x)
        return x

