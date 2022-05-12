import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F


# tversky loss
def tverskyLoss(pred, target, alpha = 0.5, beta = 0.5, smooth = 1.):
    pred = pred.contiguous()
    target = target.contiguous()
    
    
    TP = (pred * target).sum(dim=2).sum(dim=2)    
    FP = ((1-target) * pred).sum(dim=2).sum(dim=2)
    FN = (target * (1-pred)).sum(dim=2).sum(dim=2)
    
    loss = (TP + smooth) / (TP + alpha * FP + beta * FN + smooth)
    loss = 1 - loss
    
    return loss.mean() 


# calculate overall loss
def calcLoss(pred, target, stats, bceWeight=0.5):

    pred = pred.squeeze()
    target = target.squeeze()

    bce = F.binary_cross_entropy_with_logits(pred, target)

    pred = F.sigmoid(pred)
    tversky = tverskyLoss(pred, target)

    loss = bce * bceWeight + tversky * (1 - bceWeight)

    stats['bce'] += bce.data.cpu().numpy() * target.size(0)
    stats['tversky'] += tversky.data.cpu().numpy() * target.size(0)
    stats['loss'] += loss.data.cpu().numpy() * target.size(0)

    return loss


# print loss values
def printStats(stats, epochSamples, phase):
   
    outStats = []

    for i in stats.keys():
        outStats.append("{}: {:3f}".format(i, stats[i] / epochSamples))

    print("{}: {}".format(phase, ", ".join(outStats)))
