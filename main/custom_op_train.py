
r"""
This file is an example of defining a custom operation and providing its
relaxations for bound computation. Here we consider a modified ReLU
function which is a mixture between a linear function and ReLU function:
             / a_i x_i + b_i   if m_i = 1,
    f(x_i) = |
             \ ReLU(x_i)       if m_i = 0.
where a, b are element-wise slopes and biases when the function is linear,
and m is the mask controlling the behavior of this function. We consider
perturbations on x.

An example command to run verification on this customized model:

python robustness_verifier.py --config exp_configs/custom_op_example.yaml --mode crown-only-verified-acc --batch_size 256

Note that if you also want to conduct branch and bound on your customized
op, you may also need to customize BaB code.
"""
import torch
import torch.nn as nn
from auto_LiRPA import register_custom_op
from auto_LiRPA.bound_ops import Bound, BoundRelu, Interval

__all__ = ['mnist_8_200_graft', 'mnist_conv_big_graft', 'svhn_conv_big_graft',
           'cifar_cnn_b_graft', 'cifar_cnn_17m_graft', 'cifar_conv_big_graft', 'cifar_resnet4b_graft', 'cifar_cnn_17m_graft_bn']

class LinearMaskedReluOp(torch.autograd.Function):
    """A relu function with some neurons replaced with linear operations."""
    @staticmethod
    def forward(ctx, input: torch.Tensor, mask: torch.Tensor, slope: torch.Tensor, bias: torch.Tensor) -> torch.Tensor:
        # mask = 1 => using linear operation input * slope + bias, mask = 0 => using ReLU
        ctx.save_for_backward(input, mask, slope, bias)
        return input.clamp(min=0) * (1.0 - mask) + (input * slope + bias) * mask

    def backward(ctx, grad_output):
        """
        In the backward pass we receive a Tensor containing the gradient of the loss
        with respect to the output, and we need to compute the gradient of the loss
        with respect to the input.
        """
        input, mask, slope, bias = ctx.saved_tensors
        relu_grad = grad_output.clone()
        relu_grad[input < 0] = 0
        grad_input = relu_grad * (1.0 - mask) + grad_output * mask * slope
        grad_slope = grad_output * input * mask
        grad_bias = grad_output * mask
        grad_mask = -input.clamp(min=0) + (input * slope + bias)
        return grad_input, grad_mask, grad_slope, grad_bias

    @staticmethod
    def symbolic(g, input, mask, weight, bias):
        # This will be parsed as a custom operation when doing the ONNX conversion.
        return g.op("customOp::LinearMaskedRelu", input, mask, weight, bias)


class LinearMaskedRelu(nn.Module):
    """Create a module to wrap the parameters for LinearMaskedReluOp."""
    def __init__(self, size, value = 1.0, bias=0.0):
        super().__init__()
        if isinstance(size, int):
            size = (size, )
        # All mask, slope and bias are element-wise.
        self.register_buffer('mask', (torch.zeros(size=size)).to(dtype=torch.get_default_dtype()))
        self.register_parameter('slope', nn.Parameter(value * torch.ones(size=size)))
        self.register_parameter('bias', nn.Parameter(bias * torch.ones(size=size)))

    def forward(self, input):
        # mask = 1 => using linear operation input * slope + bias, mask = 0 => using ReLU
        return LinearMaskedReluOp.apply(input, self.mask, self.slope, self.bias)


class BoundLinearMaskedRelu(BoundRelu):
    """This class defines how we compute the bounds for our customized Relu function."""

    @Bound.save_io_shape
    def forward(self, x, mask, slope, bias):
        """Regular forward propagation (e.g., for evaluating clean accuracy)."""
        # Save the shape, which will be used in other parts of the verifier.
        self.shape = x.shape[1:]
        if self.flattened_nodes is None:
            self.flattened_nodes = x[0].reshape(-1).shape[0]
        return LinearMaskedReluOp.apply(x, mask, slope, bias)

    def interval_propagate(self, x, mask, slope, bias):
        """Interval bound propagation (IBP)."""
        # Each x, mask, slope, bias is a tuple, or a Interval object representing lower and upper bounds.
        # We assume Linf norm perturbation on input.
        assert Interval.get_perturbation(x)[0] == float("inf")
        x_L, x_U = x[0], x[1]  # The inputs (x)
        # We assume no perturbations on mask, slope and bias.
        mask, slope, bias = mask[0], slope[0], bias[0]
        # Lower and upper bounds when ReLU is selected.
        relu_lb = x_L.clamp(min=0)
        relu_ub = x_U.clamp(min=0)
        # Lower and upper bounds when linear coefficients are selected.
        pos_slope = (slope >= 0).to(dtype=torch.get_default_dtype())
        neg_slope = 1.0 - pos_slope
        linear_lb = pos_slope * slope * x_L + neg_slope * slope * x_U + bias
        linear_ub = pos_slope * slope * x_U + neg_slope * slope * x_L + bias
        # Select the final bounds according to the mask.
        final_lb = mask * linear_lb + (1.0 - mask) * relu_lb
        final_ub = mask * linear_ub + (1.0 - mask) * relu_ub
        return final_lb, final_ub

    def _backward_relaxation(self, last_lA, last_uA, x, start_node, unstable_idx):
        """Element-wise CROWN relaxation for our special ReLU activation function."""
        # Call parent class to relax ReLU neurons.
        upper_d, upper_b, lower_d, lower_b, lb_lower_d, ub_lower_d = super()._backward_relaxation(
                last_lA, last_uA, x, start_node, unstable_idx)
        # Modify the relaxation coefficients for these linear neurons.
        neg_mask = 1.0 - self._mask
        masked_slope = self._mask * self._slope
        masked_bias = self._mask * self._bias
        upper_d = masked_slope + neg_mask * upper_d
        upper_b = masked_bias + neg_mask * upper_b
        if lower_d is not None:
            # Shared slope between lower and upper bounds.
            lower_d = masked_slope + neg_mask * lower_d
        else:
            # Not shared slopes: we have two set of slopes one for lA, one for uA.
            # One of them might be not necessary (None), if only lower or upper bound is computed.
            lb_lower_d = masked_slope + neg_mask * lb_lower_d if lb_lower_d is not None else None
            ub_lower_d = masked_slope + neg_mask * ub_lower_d if ub_lower_d is not None else None
        assert lower_b is None  # For ReLU, there is no lower bias (=0)
        # The required dimension is (batch, spec, C, H, W). The size of masked_bias is (C,H,W),
        # and we need to expand other dimensions.
        lower_b = masked_bias.unsqueeze(0).unsqueeze(0).expand(upper_b.size())
        return upper_d, upper_b, lower_d, lower_b, lb_lower_d, ub_lower_d

    def bound_backward(self, last_lA, last_uA, x, mask, slope, bias, **kwargs):
        """Backward LiRPA (CROWN) bound propagation."""
        # These are additional variabels that will be used in _backward_relaxation(), so we save them here.
        self._mask = mask.buffer  # These are registered as buffers; see class BoundBuffer.
        self._slope = slope.buffer
        self._bias = bias.buffer
        # The parent class will call _backward_relaxation() and obtain the relaxations,
        # and that's all we need; after obtaining linear relaxations for each neuron, other
        # parts of class BoundRelu can be reused.
        As, lbias, ubias = super().bound_backward(last_lA, last_uA, x, **kwargs)
        # Returned As = [(lA, uA)]; these A matrices are for input x.
        # Our customized ReLU has three additional buffers as inputs; we need to set their
        # corresponding A matrices to None. The length of As must match the number of inputs
        # of this customize function.
        As += [(None, None), (None, None), (None, None)]
        return As, lbias, ubias


class Flatten(nn.Module):
    def forward(self, x):
        return x.reshape(x.size(0), -1)


class mnist_8_200_graft(nn.Module):

    def __init__(self, v=1.0, b=0.0):
        super(mnist_8_200_graft, self).__init__()

        self.layer0 = Flatten()
        self.layer1 = nn.Linear(784, 200)
        self.linear_masked_relu2 = LinearMaskedRelu(size=(200), value=v, bias=b)
        self.layer3 = nn.Linear(200, 200)
        self.linear_masked_relu4 = LinearMaskedRelu(size=(200), value=v, bias=b)
        self.layer5 = nn.Linear(200, 200)
        self.linear_masked_relu6 = LinearMaskedRelu(size=(200), value=v, bias=b)
        self.layer7 = nn.Linear(200, 200)
        self.linear_masked_relu8 = LinearMaskedRelu(size=(200), value=v, bias=b)
        self.layer9 = nn.Linear(200, 200)
        self.linear_masked_relu10 = LinearMaskedRelu(size=(200), value=v, bias=b)
        self.layer11 = nn.Linear(200, 200)
        self.linear_masked_relu12 = LinearMaskedRelu(size=(200), value=v, bias=b)
        self.layer13 = nn.Linear(200, 200)
        self.linear_masked_relu14 = LinearMaskedRelu(size=(200), value=v, bias=b)
        self.layer15 = nn.Linear(200, 10)

        register_custom_op("customOp::LinearMaskedRelu", BoundLinearMaskedRelu)

    def forward(self, x):

        out = self.layer0(x)
        out = self.layer1(out)
        out = self.linear_masked_relu2(out)
        out = self.layer3(out)
        out = self.linear_masked_relu4(out)
        out = self.layer5(out)
        out = self.linear_masked_relu6(out)
        out = self.layer7(out)
        out = self.linear_masked_relu8(out)
        out = self.layer9(out)
        out = self.linear_masked_relu10(out)
        out = self.layer11(out)
        out = self.linear_masked_relu12(out)
        out = self.layer13(out)
        out = self.linear_masked_relu14(out)
        out = self.layer15(out)

        return out

class mnist_conv_big_graft(nn.Module):

    def __init__(self, v=1.0, b=0.0):
        super(mnist_conv_big_graft, self).__init__()

        self.layer0 = nn.Conv2d(1, 32, 3, stride=1, padding=1)
        self.linear_masked_relu1 = LinearMaskedRelu(size=(32,28,28), value=v, bias=b)
        self.layer2 = nn.Conv2d(32, 32, 4, stride=2, padding=1)
        self.linear_masked_relu3 = LinearMaskedRelu(size=(32,14,14), value=v, bias=b)
        self.layer4 = nn.Conv2d(32, 64, 3, stride=1, padding=1)
        self.linear_masked_relu5 = LinearMaskedRelu(size=(64,14,14), value=v, bias=b)
        self.layer6 = nn.Conv2d(64, 64, 4, stride=2, padding=1)
        self.linear_masked_relu7 = LinearMaskedRelu(size=(64,7,7), value=v, bias=b)
        self.layer8 = Flatten()
        self.layer9 = nn.Linear(3136, 512)
        self.linear_masked_relu10 = LinearMaskedRelu(size=(512), value=v, bias=b)
        self.layer11 = nn.Linear(512, 512)
        self.linear_masked_relu12 = LinearMaskedRelu(size=(512), value=v, bias=b)
        self.layer13 = nn.Linear(512, 10)

        register_custom_op("customOp::LinearMaskedRelu", BoundLinearMaskedRelu)

    def forward(self, x):

        out = self.layer0(x)
        out = self.linear_masked_relu1(out)
        out = self.layer2(out)
        out = self.linear_masked_relu3(out)
        out = self.layer4(out)
        out = self.linear_masked_relu5(out)
        out = self.layer6(out)
        out = self.linear_masked_relu7(out)
        out = self.layer8(out)
        out = self.layer9(out)
        out = self.linear_masked_relu10(out)
        out = self.layer11(out)
        out = self.linear_masked_relu12(out)
        out = self.layer13(out)
        return out

class svhn_conv_big_graft(nn.Module):

    def __init__(self, v=1.0, b=0.0):
        super(svhn_conv_big_graft, self).__init__()

        self.layer0 = nn.Conv2d(3, 32, 3, stride=1, padding=1)
        self.linear_masked_relu1 = LinearMaskedRelu(size=(32,32,32), value=v, bias=b)
        self.layer2 = nn.Conv2d(32, 32, 4, stride=2, padding=1)
        self.linear_masked_relu3 = LinearMaskedRelu(size=(32,16,16), value=v, bias=b)
        self.layer4 = nn.Conv2d(32, 64, 3, stride=1, padding=1)
        self.linear_masked_relu5 = LinearMaskedRelu(size=(64,16,16), value=v, bias=b)
        self.layer6 = nn.Conv2d(64, 64, 4, stride=2, padding=1)
        self.linear_masked_relu7 = LinearMaskedRelu(size=(64,8,8), value=v, bias=b)
        self.layer8 = Flatten()
        self.layer9 = nn.Linear(4096, 512)
        self.linear_masked_relu10 = LinearMaskedRelu(size=(512), value=v, bias=b)
        self.layer11 = nn.Linear(512, 512)
        self.linear_masked_relu12 = LinearMaskedRelu(size=(512), value=v, bias=b)
        self.layer13 = nn.Linear(512, 10)

        register_custom_op("customOp::LinearMaskedRelu", BoundLinearMaskedRelu)

    def forward(self, x):

        out = self.layer0(x)
        out = self.linear_masked_relu1(out)
        out = self.layer2(out)
        out = self.linear_masked_relu3(out)
        out = self.layer4(out)
        out = self.linear_masked_relu5(out)
        out = self.layer6(out)
        out = self.linear_masked_relu7(out)
        out = self.layer8(out)
        out = self.layer9(out)
        out = self.linear_masked_relu10(out)
        out = self.layer11(out)
        out = self.linear_masked_relu12(out)
        out = self.layer13(out)
    
        return out

class cifar_cnn_b_graft(nn.Module):

    def __init__(self, v=1.0, b=0.0):
        super(cifar_cnn_b_graft, self).__init__()

        self.layer0 = nn.ZeroPad2d((1,2,1,2))
        self.layer1 = nn.Conv2d(3, 32, (5,5), stride=2, padding=0)
        self.linear_masked_relu2 = LinearMaskedRelu(size=(32,16,16), value=v, bias=b)
        self.layer3 = nn.Conv2d(32, 128, (4,4), stride=2, padding=1)
        self.linear_masked_relu4 = LinearMaskedRelu(size=(128,8,8), value=v, bias=b)
        self.layer5 = Flatten()
        self.layer6 = nn.Linear(8192, 250)
        self.linear_masked_relu7 = LinearMaskedRelu(size=(250), value=v, bias=b)
        self.layer8 = nn.Linear(250, 10)

        register_custom_op("customOp::LinearMaskedRelu", BoundLinearMaskedRelu)

    def forward(self, x):

        out = self.layer0(x)
        out = self.layer1(out)
        out = self.linear_masked_relu2(out)
        out = self.layer3(out)
        out = self.linear_masked_relu4(out)
        out = self.layer5(out)
        out = self.layer6(out)
        out = self.linear_masked_relu7(out)
        out = self.layer8(out)

        return out

class cifar_cnn_17m_graft(nn.Module):

    def __init__(self, v=1.0, b=0.0):
        super(cifar_cnn_17m_graft, self).__init__()

        self.layer0 = nn.Conv2d(3, 64, 3, stride=1, padding=1)
        self.linear_masked_relu1 = LinearMaskedRelu(size=(64,32,32), value=v, bias=b)
        self.layer2 = nn.Conv2d(64, 64, 3, stride=1, padding=1)
        self.linear_masked_relu3 = LinearMaskedRelu(size=(64,32,32), value=v, bias=b)
        self.layer4 = nn.Conv2d(64, 128, 3, stride=2, padding=1)
        self.linear_masked_relu5 = LinearMaskedRelu(size=(128,16,16), value=v, bias=b)
        self.layer6 = nn.Conv2d(128, 128, 3, stride=1, padding=1)
        self.linear_masked_relu7 = LinearMaskedRelu(size=(128,16,16), value=v, bias=b)
        self.layer8 = nn.Conv2d(128, 128, 3, stride=1, padding=1)
        self.linear_masked_relu9 = LinearMaskedRelu(size=(128,16,16), value=v, bias=b)
        self.layer10 = Flatten()
        self.layer11 = nn.Linear(32768, 512)
        self.linear_masked_relu12 = LinearMaskedRelu(size=(512), value=v, bias=b)
        self.layer13 = nn.Linear(512, 10)

        register_custom_op("customOp::LinearMaskedRelu", BoundLinearMaskedRelu)

    def forward(self, x):

        out = self.layer0(x)
        out = self.linear_masked_relu1(out)
        out = self.layer2(out)
        out = self.linear_masked_relu3(out)
        out = self.layer4(out)
        out = self.linear_masked_relu5(out)
        out = self.layer6(out)
        out = self.linear_masked_relu7(out)
        out = self.layer8(out)
        out = self.linear_masked_relu9(out)
        out = self.layer10(out)
        out = self.layer11(out)
        out = self.linear_masked_relu12(out)
        out = self.layer13(out)

        return out


class cifar_cnn_17m_graft_bn(nn.Module):

    def __init__(self, v=1.0, b=0.0):
        super(cifar_cnn_17m_graft_bn, self).__init__()

        self.layer0 = nn.Conv2d(3, 64, 3, stride=1, padding=1)
        self.layer1 = nn.BatchNorm2d(64)
        self.linear_masked_relu2 = LinearMaskedRelu(size=(64,32,32), value=v, bias=b)
        self.layer3 = nn.Conv2d(64, 64, 3, stride=1, padding=1)
        self.layer4 = nn.BatchNorm2d(64)
        self.linear_masked_relu5 = LinearMaskedRelu(size=(64,32,32), value=v, bias=b)
        self.layer6 = nn.Conv2d(64, 128, 3, stride=2, padding=1)
        self.layer7 = nn.BatchNorm2d(128)
        self.linear_masked_relu8 = LinearMaskedRelu(size=(128,16,16), value=v, bias=b)
        self.layer9 = nn.Conv2d(128, 128, 3, stride=1, padding=1)
        self.layer10 = nn.BatchNorm2d(128)
        self.linear_masked_relu11 = LinearMaskedRelu(size=(128,16,16), value=v, bias=b)
        self.layer12 = nn.Conv2d(128, 128, 3, stride=1, padding=1)
        self.layer13 = nn.BatchNorm2d(128)
        self.linear_masked_relu14 = LinearMaskedRelu(size=(128,16,16), value=v, bias=b)
        self.layer15 = Flatten()
        self.layer16 = nn.Linear(32768, 512)
        self.layer17 = nn.BatchNorm1d(512)
        self.linear_masked_relu18 = LinearMaskedRelu(size=(512), value=v, bias=b)
        self.layer19 = nn.Linear(512, 10)

        register_custom_op("customOp::LinearMaskedRelu", BoundLinearMaskedRelu)

    def forward(self, x):

        out = self.layer0(x)
        out = self.layer1(out)
        out = self.linear_masked_relu2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = self.linear_masked_relu5(out)
        out = self.layer6(out)
        out = self.layer7(out)
        out = self.linear_masked_relu8(out)
        out = self.layer9(out)
        out = self.layer10(out)
        out = self.linear_masked_relu11(out)
        out = self.layer12(out)
        out = self.layer13(out)
        out = self.linear_masked_relu14(out)
        out = self.layer15(out)
        out = self.layer16(out)
        out = self.layer17(out)
        out = self.linear_masked_relu18(out)
        out = self.layer19(out)

        return out


class cifar_conv_big_graft(nn.Module):

    def __init__(self, v=1.0, b=0.0):
        super(cifar_conv_big_graft, self).__init__()

        self.layer0 = nn.Conv2d(3, 32, 3, stride=1, padding=1)
        self.linear_masked_relu1 = LinearMaskedRelu(size=(32,32,32), value=v, bias=b)
        self.layer2 = nn.Conv2d(32, 32, 4, stride=2, padding=1)
        self.linear_masked_relu3 = LinearMaskedRelu(size=(32,16,16), value=v, bias=b)
        self.layer4 = nn.Conv2d(32, 64, 3, stride=1, padding=1)
        self.linear_masked_relu5 = LinearMaskedRelu(size=(64,16,16), value=v, bias=b)
        self.layer6 = nn.Conv2d(64, 64, 4, stride=2, padding=1)
        self.linear_masked_relu7 = LinearMaskedRelu(size=(64,8,8), value=v, bias=b)
        self.layer8 = Flatten()
        self.layer9 = nn.Linear(4096, 512)
        self.linear_masked_relu10 = LinearMaskedRelu(size=(512), value=v, bias=b)
        self.layer11 = nn.Linear(512, 512)
        self.linear_masked_relu12 = LinearMaskedRelu(size=(512), value=v, bias=b)
        self.layer13 = nn.Linear(512, 10)

        register_custom_op("customOp::LinearMaskedRelu", BoundLinearMaskedRelu)

    def forward(self, x):

        out = self.layer0(x)
        out = self.linear_masked_relu1(out)
        out = self.layer2(out)
        out = self.linear_masked_relu3(out)
        out = self.layer4(out)
        out = self.linear_masked_relu5(out)
        out = self.layer6(out)
        out = self.linear_masked_relu7(out)
        out = self.layer8(out)
        out = self.layer9(out)
        out = self.linear_masked_relu10(out)
        out = self.layer11(out)
        out = self.linear_masked_relu12(out)
        out = self.layer13(out)
    
        return out

class cifar_resnet4b_hook_layer1_0(nn.Module):
    def __init__(self):
        super(cifar_resnet4b_hook_layer1_0, self).__init__()
        self.conv1 = nn.Conv2d(16, 32, 3, stride=2, padding=1)
        self.conv2 = nn.Conv2d(32, 32, 3, stride=1, padding=1)
        self.shortcut = nn.Sequential(
            nn.Conv2d(16, 32, 1, stride=2)
        )
    def forward(self, x):
        return x

class cifar_resnet4b_hook_layer1_1(nn.Module):
    def __init__(self):
        super(cifar_resnet4b_hook_layer1_1, self).__init__()
        self.conv1 = nn.Conv2d(32, 32, 3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(32, 32, 3, stride=1, padding=1)
        self.shortcut = nn.Sequential()

class cifar_resnet4b_hook_layer2_0(nn.Module):
    def __init__(self):
        super(cifar_resnet4b_hook_layer2_0, self).__init__()
        self.conv1 = nn.Conv2d(32, 32, 3, stride=2, padding=1)
        self.conv2 = nn.Conv2d(32, 32, 3, stride=1, padding=1)
        self.shortcut = nn.Sequential(
            nn.Conv2d(32, 32, 1, stride=2)
        )

class cifar_resnet4b_hook_layer2_1(nn.Module):
    def __init__(self):
        super(cifar_resnet4b_hook_layer2_1, self).__init__()
        self.conv1 = nn.Conv2d(32, 32, 3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(32, 32, 3, stride=1, padding=1)
        self.shortcut = nn.Sequential()

class cifar_resnet4b_graft(nn.Module):

    def __init__(self, v=1.0, b=0.0):
        super(cifar_resnet4b_graft, self).__init__()

        self.conv1 = nn.Conv2d(3, 16, 3, stride=2, padding=1)
        self.layer1 = nn.Sequential(
            cifar_resnet4b_hook_layer1_0(),
            cifar_resnet4b_hook_layer1_1())
        self.layer2 = nn.Sequential(
            cifar_resnet4b_hook_layer2_0(),
            cifar_resnet4b_hook_layer2_1())
        self.linear1 = nn.Linear(512, 100)
        self.linear2 = nn.Linear(100, 10)

        self.linear_masked_relu1 = LinearMaskedRelu(size=(16,16,16), value=v, bias=b)
        self.linear_masked_relu2 = LinearMaskedRelu(size=(32,8,8), value=v, bias=b)
        self.linear_masked_relu3 = LinearMaskedRelu(size=(32,8,8), value=v, bias=b)
        self.linear_masked_relu4 = LinearMaskedRelu(size=(32,8,8), value=v, bias=b)
        self.linear_masked_relu5 = LinearMaskedRelu(size=(32,8,8), value=v, bias=b)
        self.linear_masked_relu6 = LinearMaskedRelu(size=(32,4,4), value=v, bias=b)
        self.linear_masked_relu7 = LinearMaskedRelu(size=(32,4,4), value=v, bias=b)
        self.linear_masked_relu8 = LinearMaskedRelu(size=(32,4,4), value=v, bias=b)
        self.linear_masked_relu9 = LinearMaskedRelu(size=(32,4,4), value=v, bias=b)
        self.linear_masked_relu10 = LinearMaskedRelu(size=(100), value=v, bias=b)

        register_custom_op("customOp::LinearMaskedRelu", BoundLinearMaskedRelu)


    def forward(self, x):

        out = self.conv1(x)
        out = self.linear_masked_relu1(out)
        # layer1-0
        out_sub = self.layer1[0].conv1(out)
        out_sub = self.linear_masked_relu2(out_sub)
        out_sub = self.layer1[0].conv2(out_sub)
        out = out_sub + self.layer1[0].shortcut(out)
        out = self.linear_masked_relu3(out)

        # layer1-1
        out_sub = self.layer1[1].conv1(out)
        out_sub = self.linear_masked_relu4(out_sub)
        out_sub = self.layer1[1].conv2(out_sub)
        out = out_sub + self.layer1[1].shortcut(out)
        out = self.linear_masked_relu5(out)

        # layer2-0
        out_sub = self.layer2[0].conv1(out)
        out_sub = self.linear_masked_relu6(out_sub)
        out_sub = self.layer2[0].conv2(out_sub)
        out = out_sub + self.layer2[0].shortcut(out)
        out = self.linear_masked_relu7(out)

        # layer2-1
        out_sub = self.layer2[1].conv1(out)
        out_sub = self.linear_masked_relu8(out_sub)
        out_sub = self.layer2[1].conv2(out_sub)
        out = out_sub + self.layer2[1].shortcut(out)
        out = self.linear_masked_relu9(out)

        out = torch.flatten(out, 1)
        out = self.linear_masked_relu10(self.linear1(out))
        out = self.linear2(out)
        return out 




















