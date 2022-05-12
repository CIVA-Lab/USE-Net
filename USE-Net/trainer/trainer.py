import torch
import copy
import time

from utils import calcLoss
from utils import printStats
from collections import defaultdict

def trainModel(dataloaders, model, optimizer, scheduler, numEpochs=40, modelN='USENet'):

    # set initial best model weights and loss
    bestModelWeights = copy.deepcopy(model.state_dict())
    bestValLoss = 100000
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # iterate thorugh epoches
    for epoch in range(numEpochs):
        print('Epoch {}/{}'.format(epoch + 1, numEpochs))
        print('*' * 20)

        startTime = time.time()

        # look for training and validation phase in each epoch
        for phase in ['train', 'val']:
            # if phase is training
            if phase == 'train':
                scheduler.step()
                for groupParam in optimizer.param_groups:
                    print("Learning rate:", groupParam['lr'])
                # train model
                model.train()
            else:
                # clean model for validation phase
                model.eval()

            stats = defaultdict(float)
            epochSamples = 0

            for inputs, labels in dataloaders[phase]:
                inputs = inputs.to(device)
                labels = labels.to(device)

                optimizer.zero_grad()

                # forward
                with torch.set_grad_enabled(phase == 'train'):

                    inputs = inputs.float()
                    outputs = model(inputs)

                    labels = labels.unsqueeze(1)
                    labels = labels.type_as(outputs)

                    # calculate loss
                    loss = calcLoss(outputs, labels, stats)

                    # backward and optimize in traning phase
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                # statistics
                epochSamples += inputs.size(0)

                del inputs, labels


            # print statistics
            printStats(stats, epochSamples, phase)

            # calculate epoch loss
            epochLoss = stats['loss'] / epochSamples

            # save the best model
            if phase == 'val' and epochLoss < bestValLoss:
                print("Saving best model")
                bestValLoss = epochLoss
                bestModelWeights = copy.deepcopy(model.state_dict())
                torch.save(bestModelWeights, './models/' + modelN + '.pt')

        finalTime = time.time() - startTime
        print('{:.0f}m {:.0f}s'.format(finalTime // 60, finalTime % 60))

    print('Best validation loss: {:3f}'.format(bestValLoss))
