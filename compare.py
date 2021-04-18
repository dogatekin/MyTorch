import torch  
from torch import nn
from torch import optim
from math import pi
from modules import Linear, Tanh, ReLU, Sequential, MSE
from optim import SGD
from matplotlib import pyplot as plt
from time import time

torch.manual_seed(0)

def generate_disc_set(nb):
    """Generate the dataset with one-hot encoded targets."""
    input = torch.rand(nb, 2)
    target = torch.zeros((nb, 2))
    target[(input - 0.5).pow(2).sum(1) < 0.5/pi, 1] = 1
    target[(input - 0.5).pow(2).sum(1) >= 0.5/pi, 0] = 1
    return input, target

train_input, train_target = generate_disc_set(1000)
test_input, test_target = generate_disc_set(1000)

batch_size = 100
num_batches = len(train_input) // batch_size

torch.manual_seed(0)
our_relu = Sequential(
    Linear(2, 25),
    ReLU(),
    Linear(25, 25),
    ReLU(),
    Linear(25, 25),
    ReLU(),
    Linear(25, 2)
)

torch.manual_seed(0)
torch_relu = nn.Sequential(
    nn.Linear(2, 25),
    nn.ReLU(),
    nn.Linear(25, 25),
    nn.ReLU(),
    nn.Linear(25, 25),
    nn.ReLU(),
    nn.Linear(25, 2)
)

torch.manual_seed(0)
our_tanh = Sequential(
    Linear(2, 25),
    Tanh(),
    Linear(25, 25),
    Tanh(),
    Linear(25, 25),
    Tanh(),
    Linear(25, 2)
)

torch.manual_seed(0)
torch_tanh = nn.Sequential(
    nn.Linear(2, 25),
    nn.Tanh(),
    nn.Linear(25, 25),
    nn.Tanh(),
    nn.Linear(25, 25),
    nn.Tanh(),
    nn.Linear(25, 2)
)

# Check same initialization
print(f'All parameters initialized the same: {all(torch.all(ours.eq(theirs)) for ours, theirs in zip(our_relu.parameters(), torch_relu.parameters()))}')

def our_fit(model, epochs=500, verbose=False):
    criterion = MSE()
    optimizer = SGD(model.parameters(), model.grads(), lr=0.1)
    
    start = time()
    losses = []
    if verbose: print('Epoch | Loss')
    for epoch in range(epochs):
        epoch_loss = 0
        for b in range(num_batches):
            batch_input = train_input[b*batch_size:(b+1)*batch_size]
            batch_target = train_target[b*batch_size:(b+1)*batch_size]
            
            batch_output = model(batch_input)
            batch_loss = criterion(batch_output, batch_target)
            epoch_loss += batch_loss
            
            output_grad = criterion.backward()
            model.backward(output_grad)
        
            optimizer.step()
            optimizer.zero_grad()

        losses.append(epoch_loss.item() / num_batches)
        if verbose: print(f'{epoch+1:>5} | {epoch_loss.item() / num_batches:.5f}')
    end = time()
     
    train_output = model(train_input)
    print(f'\nTrain Error: {sum(train_output.argmax(1) != train_target.argmax(1)).item() / len(train_output)}')
    test_output = model(test_input)
    print(f'Test Error: {sum(test_output.argmax(1) != test_target.argmax(1)).item() / len(test_output)}')
    
    return losses, end-start

def torch_fit(model, epochs=500, verbose=False):
    criterion = nn.MSELoss()
    optimizer = optim.SGD(model.parameters(), lr=0.1)

    start = time()
    losses = []
    if verbose: print('Epoch | Loss')
    for epoch in range(epochs):
        epoch_loss = 0
        for b in range(num_batches):
            batch_input = train_input[b*batch_size:(b+1)*batch_size]
            batch_target = train_target[b*batch_size:(b+1)*batch_size]
            
            batch_output = model(batch_input)
            batch_loss = criterion(batch_output, batch_target)
            epoch_loss += batch_loss
            
            batch_loss.backward()
            
            optimizer.step()
            optimizer.zero_grad()

        losses.append(epoch_loss.item() / num_batches)
        if verbose: print(f'{epoch+1:>5} | {epoch_loss.item() / num_batches:.5f}')
    end = time()
     
    train_output = model(train_input)
    print(f'\nTrain Error: {sum(train_output.argmax(1) != train_target.argmax(1)).item() / len(train_output)}')
    test_output = model(test_input)
    print(f'Test Error: {sum(test_output.argmax(1) != test_target.argmax(1)).item() / len(test_output)}')
    
    return losses, end-start

print('\nOur ReLU:', end='')
our_relu_losses, our_relu_time = our_fit(our_relu)
print('\nPyTorch ReLU:', end='')
torch_relu_losses, torch_relu_time = torch_fit(torch_relu)

print('\nOur Tanh:', end='')
our_tanh_losses, our_tanh_time = our_fit(our_tanh)
print('\nPyTorch Tanh:', end='')
torch_tanh_losses, torch_tanh_time = torch_fit(torch_tanh)

epochs = range(1, len(our_relu_losses))
fig, axs = plt.subplots(2, 1, sharex=True, figsize=(6.4, 5.2))

axs[0].plot(epochs, our_relu_losses[1:], label=f'Ours ({our_relu_time:.2f}s)', lw=4)
axs[0].plot(epochs, torch_relu_losses[1:], label=f'PyTorch ({torch_relu_time:.2f}s)', ls='--', lw=3)
axs[0].set_title('ReLU')
axs[0].set_ylabel('Training Loss')
axs[0].legend()

axs[1].plot(epochs, our_tanh_losses[1:], label=f'Ours ({our_tanh_time:.2f}s)', lw=4)
axs[1].plot(epochs, torch_tanh_losses[1:], label=f'PyTorch ({torch_tanh_time:.2f}s)', ls='--', lw=3)
axs[1].set_title('Tanh')
axs[1].set_xlabel('Epochs')
axs[1].set_ylabel('Training Loss')
axs[1].legend()

# plt.show()
plt.savefig('Comparisons.png', bbox_inches='tight')

# Training time comparison
# our_relu_times = torch.Tensor([our_fit(our_relu, verbose=False)[1] for _ in range(10)])
# print(f'{our_relu_times.mean().item():.3f} ± {our_relu_times.std(unbiased=False).item():.3f}')

# torch_relu_times = torch.Tensor([torch_fit(torch_relu, verbose=False)[1] for _ in range(10)])
# print(f'{torch_relu_times.mean().item():.3f} ± {torch_relu_times.std(unbiased=False).item():.3f}')

# our_tanh_times = torch.Tensor([our_fit(our_tanh, verbose=False)[1] for _ in range(10)])
# print(f'{our_tanh_times.mean().item():.3f} ± {our_tanh_times.std(unbiased=False).item():.3f}')

# torch_tanh_times = torch.Tensor([torch_fit(torch_tanh, verbose=False)[1] for _ in range(10)])
# print(f'{torch_tanh_times.mean().item():.3f} ± {torch_tanh_times.std(unbiased=False).item():.3f}')