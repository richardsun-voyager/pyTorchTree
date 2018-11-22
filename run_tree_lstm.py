import sys
import random
import progressbar
import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F
from torch.nn.utils import clip_grad_norm
from SenTree import *
from tree_lstm import *

def Var(v):
    if CUDA: return Variable(v.cuda())
    else: return Variable(v)

CUDA=False
if len(sys.argv)>1:
  if sys.argv[1].lower()=="cuda": CUDA=True

print("Reading and parsing trees")
trn = SenTree.getTrees("./trees/train.txt","train.vocab")
dev = SenTree.getTrees("./trees/dev.txt",vocabIndicesMapFile="train.vocab")

if CUDA: model = TreeLSTM(SenTree.vocabSize).cuda()
else: model = TreeLSTM(SenTree.vocabSize)
max_epochs = 100
widgets = [progressbar.Percentage(), ' ', progressbar.Bar(), ' ', progressbar.ETA()]
optimizer = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.9, dampening=0.0)
bestAll=bestRoot=0.0
for epoch in range(max_epochs):
  print("Epoch %d" % epoch)
  pbar = progressbar.ProgressBar(widgets=widgets, maxval=len(trn)).start()
  for step, tree in enumerate(trn):
     predictions, loss = model.getLoss(tree)
     optimizer.zero_grad()
     loss.backward()
     clip_grad_norm(model.parameters(), 5, norm_type=2.)
     optimizer.step()
     pbar.update(step)
  pbar.finish()
  correctRoot, correctAll = model.evaluate(dev)
  if bestAll<correctAll: bestAll=correctAll
  if bestRoot<correctRoot: bestRoot=correctRoot
  print("\nValidation All-nodes accuracy:"+str(correctAll)+"(best:"+str(bestAll)+")")
  print("Validation Root accuracy:" + str(correctRoot)+"(best:"+str(bestRoot)+")")
  random.shuffle(trn)
