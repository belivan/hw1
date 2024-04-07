from __future__ import print_function

import torch
import numpy as np
from torch.utils.tensorboard import SummaryWriter
import utils
from voc_dataset import VOCDataset


def save_this_epoch(args, epoch):  # function to check if we should save the model at the end of the epoch
    if args.save_freq > 0 and (epoch + 1) % args.save_freq == 0:  # save every save_freq epochs
        return True
    if args.save_at_end and (epoch + 1) == args.epochs:  # save at the end
        return True
    return False


def save_model(epoch, model_name, model):  # function to save the model
    filename = 'checkpoint-{}-epoch{}.pth'.format(  # the filename is of the form checkpoint-model_name-epoch.pth
        model_name, epoch + 1)
    print("saving model at ", filename)
    torch.save(model, filename)


def train(args, model, optimizer, scheduler=None, model_name='model'):  # function to train the model
    writer = SummaryWriter()  # create a SummaryWriter object to write to tensorboard
    train_loader = utils.get_data_loader(  # get the training data loader
        'voc', train=True, batch_size=args.batch_size, split='trainval', inp_size=args.inp_size)
    test_loader = utils.get_data_loader(  # get the test data loader
        'voc', train=False, batch_size=args.test_batch_size, split='test', inp_size=args.inp_size)

    # Ensure model is in correct mode and on right device
    model.train()
    model = model.to(args.device)  # move the model to the device

    cnt = 0  # counter to keep track of the number of iterations

    for epoch in range(args.epochs):  # iterate over the number of epochs, each epoch is a complete pass over the dataset
        for batch_idx, (data, target, wgt) in enumerate(train_loader):  # iterate over the training data
            data, target, wgt = data.to(args.device), target.to(args.device), wgt.to(args.device)  # move the data, target and weight to the device

            optimizer.zero_grad()  # zero the gradients
            output = model(data)  # forward pass

            ##################################################################
            # TODO: Implement a suitable loss function for multi-label
            # classification. You are NOT allowed to use any pytorch built-in
            # functions. Remember to take care of underflows / overflows.
            # Function Inputs:
            #   - `output`: Outputs from the network
            #   - `target`: Ground truth labels, refer to voc_dataset.py
            #   - `wgt`: Weights (difficult or not), refer to voc_dataset.py
            # Function Outputs:
            #   - `output`: Computed loss, a single floating point number
            ##################################################################
            probabilities = torch.sigmoid(output)

            # Compute BCE loss
            bce_loss = - (target * torch.log(probabilities + 1e-10) + (1 - target) * torch.log(1 - probabilities + 1e-10))
            weighted_loss = bce_loss * wgt
            loss = weighted_loss.mean()

            ##################################################################
            #                          END OF YOUR CODE                      #
            ##################################################################

            loss.backward()  # backward pass

            if cnt % args.log_every == 0:  # log the loss every log_every iterations
                writer.add_scalar("Loss/train", loss.item(), cnt)
                print('Train Epoch: {} [{} ({:.0f}%)]\tLoss: {:.6f}'.format(epoch, cnt, 100. * batch_idx / len(train_loader), loss.item()))

                # Log gradients
                for tag, value in model.named_parameters():  # log the gradients of the model parameters
                    if value.grad is not None:
                        writer.add_histogram(tag + "/grad", value.grad.cpu().numpy(), cnt)

            optimizer.step()  # update the weights

            # Validation iteration
            if cnt % args.val_every == 0:  # validate every val_every iterations
                model.eval()  # set the model to evaluation mode
                ap, map = utils.eval_dataset_map(model, args.device, test_loader)  # evaluate the model
                print("map: ", map)
                writer.add_scalar("map", map, cnt)  # log the map
                model.train()

            cnt += 1  # increment the counter

        if scheduler is not None:
            scheduler.step()
            writer.add_scalar("learning_rate", scheduler.get_last_lr()[0], cnt)  # log the learning rate

        # save model
        if save_this_epoch(args, epoch):
            save_model(epoch, model_name, model)

    # Validation iteration
    test_loader = utils.get_data_loader('voc', train=False, batch_size=args.test_batch_size, split='test', inp_size=args.inp_size)  # get the test data loader
    ap, map = utils.eval_dataset_map(model, args.device, test_loader)  # evaluate the model
    return ap, map
