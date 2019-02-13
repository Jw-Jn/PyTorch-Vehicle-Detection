import torch
import torch.nn as nn
import shutil
from torch.autograd import Variable
from collections import OrderedDict


def forward_from(module_seq: nn.Sequential,
                 start_idx: int,
                 end_index: int,
                 input_x: torch.Tensor):
    """
    Forward the network from layer
    :param module_seq: a sequential of network layers, must be nn.Sequential
    :param start_idx: start index of
    :param end_index: end index of forwarding layer
    :param input_x: input tensor to be forwarded
    :return: result of forwarding multiple layers
    """
    x = input_x
    for layer in module_seq[start_idx: end_index]:
        x = layer(x)
    return x


def save_checkpoint(state, is_best, filename='checkpoint.pth.tar'):
    torch.save(state, filename)
    if is_best:
        shutil.copyfile(filename, 'model_best.pth.tar')


def load_checkpoints(file_path):
    return torch.load(file_path)


def summary_layers(model, input_size):
    def register_hook(module):
        def hook(module, input, output):
            class_name = str(module.__class__).split('.')[-1].split("'")[0]
            module_idx = len(summary)
            if hasattr(module, 'tag'):
                module_tag = getattr(module, 'tag')
            else:
                module_tag = ''

            m_key = '%s-%i' % (class_name, module_idx + 1)
            summary[m_key] = OrderedDict()
            summary[m_key]['type'] = class_name
            summary[m_key]['idx'] = module_idx
            summary[m_key]['tag'] = module_tag
            summary[m_key]['input_shape'] = list(input[0].size())
            summary[m_key]['input_shape'][0] = -1
            if isinstance(output, (list, tuple)):
                summary[m_key]['output_shape'] = [[-1] + list(o.size())[1:] for o in output]
            else:
                summary[m_key]['output_shape'] = list(output.size())
                summary[m_key]['output_shape'][0] = -1

            params = 0
            if hasattr(module, 'weight'):
                params += torch.prod(torch.LongTensor(list(module.weight.size())))
                summary[m_key]['trainable'] = module.weight.requires_grad
            if hasattr(module, 'bias') and hasattr(module.bias, 'size'):
                params += torch.prod(torch.LongTensor(list(module.bias.size())))
            summary[m_key]['nb_params'] = params

        if (not isinstance(module, nn.Sequential) and
                not isinstance(module, nn.ModuleList) and
                not (module == model)):
            hooks.append(module.register_forward_hook(hook))

    if torch.cuda.is_available():
        dtype = torch.cuda.FloatTensor
    else:
        dtype = torch.FloatTensor

    # check if there are multiple inputs to add_graphthe network
    if isinstance(input_size[0], (list, tuple)):
        x = [Variable(torch.rand(1, *in_size)).type(dtype) for in_size in input_size]
    else:
        x = Variable(torch.rand(1, *input_size)).type(dtype)

    # print(type(x[0]))
    # create properties
    summary = OrderedDict()
    hooks = []
    # register hook
    model.apply(register_hook)
    # make a forward pass
    # print(x.shape)
    model(x)
    # remove these hooks
    for h in hooks:
        h.remove()

    print('-------------------------------------------------------------------------------------------------')
    line_new = '{:>20} {:>20} {:>10} {:>25} {:>15}'.format('Type', 'Tag', 'Index', 'Output Shape', 'Param #')
    print(line_new)
    print('=================================================================================================')
    total_params = 0
    trainable_params = 0
    for layer in summary:
        # input_shape, output_shape, trainable, nb_params
        line_new = '{:>20} {:>20} {:>10} {:>25} {:>15}'.format(summary[layer]['type'],
                                                               summary[layer]['tag'],
                                                               str(summary[layer]['idx']),
                                                               str(summary[layer]['output_shape']),
                                                               summary[layer]['nb_params'])
        total_params += summary[layer]['nb_params']
        if 'trainable' in summary[layer]:
            if summary[layer]['trainable'] == True:
                trainable_params += summary[layer]['nb_params']
        print(line_new)
    print('=================================================================================================')
    print('Total params: ' + str(total_params))
    print('Trainable params: ' + str(trainable_params))
    print('Non-trainable params: ' + str(total_params - trainable_params))
    print('-------------------------------------------------------------------------------------------------')
    # return summary
