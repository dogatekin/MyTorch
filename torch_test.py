import torch
from math import pi
from torch import nn
from torch import optim
from matplotlib import pyplot as plt


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

# Reset the seeds before each model creation so that 
# parameters are initialized the same for a fair comparison.
torch.manual_seed(0)
relu = nn.Sequential(
    nn.Linear(2, 25),
    nn.ReLU(),
    nn.Linear(25, 25),
    nn.ReLU(),
    nn.Linear(25, 25),
    nn.ReLU(),
    nn.Linear(25, 2)
)

torch.manual_seed(0)
tanh = nn.Sequential(
    nn.Linear(2, 25),
    nn.Tanh(),
    nn.Linear(25, 25),
    nn.Tanh(),
    nn.Linear(25, 25),
    nn.Tanh(),
    nn.Linear(25, 2)
)

criterion = nn.MSELoss()

def fit(model):
    optimizer = optim.SGD(model.parameters(), lr=0.1)

    losses = []
    print('Epoch | Loss')
    for epoch in range(500):
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
        print(f'{epoch+1:>5} | {epoch_loss.item() / num_batches:.5f}')
        
    train_output = model(train_input)
    print(f'\nTrain Error: {sum(train_output.argmax(1) != train_target.argmax(1)).item() / 1000}')
    test_output = model(test_input)
    print(f'Test Error: {sum(test_output.argmax(1) != test_target.argmax(1)).item() / 1000}')
    
    return losses

print('ReLU Network\n------------')    
relu_losses = fit(relu)

print('\nTanh Network\n------------')
tanh_losses = fit(tanh)

plt.plot(relu_losses[1:], label='ReLU')
plt.plot(tanh_losses[1:], label='Tanh')
plt.xlabel('Epochs')
plt.ylabel('Training Loss')
plt.legend()
plt.savefig('TorchReLUvsTanh.png', bbox_inches='tight')