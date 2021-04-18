import math
from torch import empty


class Module:
    def forward(self, *input):
        raise NotImplementedError
    
    def backward(self, *grad_wrt_output):
        raise NotImplementedError
    
    def parameters(self):
        return []
    
    def grads(self):
        return []
    
    def zero_grad(self):
        for grad in self.grads():
            grad.zero_()
    
    def __call__(self, *input):
       return self.forward(*input)
    
    
class Linear(Module):
    def __init__(self, in_features, out_features, bias=True):
        bound = 1 / math.sqrt(in_features)
        
        self.weight = empty(out_features, in_features).uniform_(-bound, bound)
        self.weight_grad = empty(out_features, in_features).zero_()
        
        if bias:
            self.bias = empty(out_features).uniform_(-bound, bound)
            self.bias_grad = empty(out_features).zero_()
        else:
            self.bias = None
            self.bias_grad = None
    
    def forward(self, input):
        self.input = input
        output = input @ self.weight.t()
        
        if self.bias is not None:
            output += self.bias
            
        return output
    
    def backward(self, grad_wrt_output):
        self.weight_grad += grad_wrt_output.t() @ self.input
        
        if self.bias_grad is not None:
            self.bias_grad += grad_wrt_output.sum(0)
            # self.bias_grad += empty(grad_wrt_output.size(0)).fill_(1) @ grad_wrt_output  # math notation
            
        return grad_wrt_output @ self.weight
    
    def parameters(self):
        if self.bias is not None:
            return self.weight, self.bias
        else:
            return self.weight,  # return as tuple even though one tensor
    
    def grads(self):
        if self.bias_grad is not None:
            return self.weight_grad, self.bias_grad
        else:
            return self.weight_grad,  # return as tuple even though one tensor


class ReLU(Module):
    def forward(self, input):
        self.input = input
        return input.clamp(min=0)
    
    def backward(self, grad_wrt_output):
        # Pass through the output gradient where input is positive, zero it otherwise.
        grad_wrt_input = grad_wrt_output.clone()
        grad_wrt_input[self.input < 0] = 0
        return grad_wrt_input
        # mask = empty(grad_wrt_output.size()).fill_(1)
        # mask[self.input < 0] = 0
        # return grad_wrt_output * mask


class Tanh(Module):
    def forward(self, input):
        self.output = input.tanh()
        return self.output
    
    def backward(self, grad_wrt_output):
        return grad_wrt_output * (1 - self.output.pow(2))

  
class Sequential(Module):
    def __init__(self, *modules):
        self.modules = modules

    def forward(self, input):
        cur = input
        for module in self.modules:
            cur = module(cur)
        return cur
    
    def backward(self, grad_wrt_output):
        for module in reversed(self.modules):
            grad_wrt_output = module.backward(grad_wrt_output)
        return grad_wrt_output
    
    def parameters(self):
        """Return a flattened list of the parameters of all modules."""
        return [param for module in self.modules for param in module.parameters()]
    
    def grads(self):
        """Return a flattened list of the gradients for all the parameters."""
        return [grad for module in self.modules for grad in module.grads()]
    
    def zero_grad(self):
        for module in self.modules:
            module.zero_grad()    


class MSE(Module):
    def forward(self, pred, target):
        self.pred = pred
        self.target = target
        return (pred - target).pow(2).mean()
    
    def backward(self):
        return 2 * (self.pred - self.target) / self.pred.numel()
