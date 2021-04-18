class SGD:
    def __init__(self, parameters, grads, lr):
        self.parameters = parameters
        self.grads = grads
        self.lr = lr
        
    def step(self):
        for param, grad in zip(self.parameters, self.grads):
            param.add_(grad, alpha=-self.lr)
    
    def zero_grad(self):
        for grad in self.grads:
            grad.zero_()