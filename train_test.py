import numpy as np
import torch
import torch.optim as optim
from tqdm import trange
import copy
import movie2

def build_optimizer(args, params):

    weight_decay = args.weight_decay
    filter_fn = filter(lambda p : p.requires_grad, params)
    if args.opt == 'adam':
        optimizer = optim.Adam(filter_fn, lr=args.lr, weight_decay=weight_decay)
    elif args.opt == 'sgd':
        optimizer = optim.SGD(filter_fn, lr=args.lr, momentum=0.95, weight_decay=weight_decay)
    elif args.opt == 'rmsprop':
        optimizer = optim.RMSprop(filter_fn, lr=args.lr, weight_decay=weight_decay)
    elif args.opt == 'adagrad':
        optimizer = optim.Adagrad(filter_fn, lr=args.lr, weight_decay=weight_decay)
    if args.opt_scheduler == 'none':
        return None, optimizer
    elif args.opt_scheduler == 'step':
        scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=args.opt_decay_step, gamma=args.opt_decay_rate)
    elif args.opt_scheduler == 'cos':
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.opt_restart)
    return scheduler, optimizer


def train(loader, test_loader, model, loss_function, args, idxMovie=None, n=None, logistic=False):

    scheduler, opt = build_optimizer(args, model.parameters())

    # train
    losses = []
    test_losses = []
    best_loss = 10000
    best_model = None
    for epoch in trange(args.epochs, desc="Training", unit="Epochs"):
        total_loss = 0
        model.train()
        for batch in loader:
            opt.zero_grad()
            #print(batch)
            pred = model(batch)
            label = batch.y
            if idxMovie is not None:
                loss = loss_function(pred, label, idxMovie, n)
            else:
                loss = loss_function(pred, label)
            loss.backward()
            opt.step()
            if len(loader.dataset) == 1:
                total_loss += loss.item() * batch.num_graphs
            else:
                total_loss += loss.item()
        #total_loss /= len(loader.dataset)
        losses.append(total_loss)

        if epoch % 10 == 0:
          test_loss = test(test_loader, model, idxMovie, n, logistic=logistic)
          test_losses.append(test_loss)
          print('Training loss', loss.item())
          print('Test loss', test_loss.item())
          if test_loss < best_loss:
            best_loss = test_loss
            best_model = copy.deepcopy(model)
        else:
          test_losses.append(test_losses[-1])
    
    return test_losses, losses, model, best_loss

def test(loader, test_model, idxMovie=None, n=None, logistic=False):
    test_model.eval()
    loss = []
    for data in loader:
        with torch.no_grad():
            pred = test_model(data)
            label = data.y
        if not logistic:
            if idxMovie is not None:
                loss.append(movie2.movieMSELoss(pred,label,idxMovie,n))
            else:
                loss.append(torch.nn.functional.mse_loss(torch.reshape(pred,[-1,1]),torch.reshape(label,[-1,1]),reduction='sum').cpu())
        else:
            loss.append(torch.nn.functional.cross_entropy(pred,label).cpu())
    return loss[0]
