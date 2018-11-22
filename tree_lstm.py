import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F
import pickle
from torch.nn.utils import clip_grad_norm
from SenTree import *
def Var(v):
    return Variable(v)


class TreeLSTM(nn.Module):
    def __init__(self, vocabSize, emb_dim=300, hdim=128, numClasses=3):
        super(TreeLSTM, self).__init__()
        self.embedding = nn.Embedding(int(vocabSize), emb_dim)
        self.Wi = nn.Linear(emb_dim, hdim, bias=True)
        self.Wo = nn.Linear(emb_dim, hdim, bias=True)
        self.Wu = nn.Linear(emb_dim, hdim, bias=True)
        self.Ui = nn.Linear(3 * hdim, hdim, bias=True)
        self.Uo = nn.Linear(3 * hdim, hdim, bias=True)
        self.Uu = nn.Linear(3 * hdim, hdim, bias=True)
        self.Uf1 = nn.Linear(hdim, hdim, bias=True)
        self.Uf2 = nn.Linear(hdim, hdim, bias=True)
        self.lstm = nn.LSTM(emb_dim, hdim, batch_first=True)
        self.projection = nn.Linear(hdim, numClasses, bias=True)
        self.activation = F.relu
        self.nodeProbList = []
        self.labelList = []

    def traverse(self, node):
        #print(node.label())
        if isinstance(node, int):
            e = self.embedding(Var(torch.LongTensor([node])))
            i = F.sigmoid(self.Wi(e))
            o = F.sigmoid(self.Wo(e))
            u = self.activation(self.Wu(e))
            c = i * u
        else:
            leftH,leftC = self.traverse(node.left())
            rightH,rightC = self.traverse(node.right())
            
            #Take target info into consideration
            target_info = self.target_hidden.squeeze(0)
            e = torch.cat((leftH, rightH, target_info), 1)
            i = F.sigmoid(self.Ui(e))
            o = F.sigmoid(self.Uo(e))
            u = self.activation(self.Uu(e))
            c = i * u + F.sigmoid(self.Uf1(leftH)) * leftC + F.sigmoid(self.Uf2(rightH)) * rightC
        h = o * self.activation(c)
        self.nodeProbList.append(self.projection(h))
        if isinstance(node, int):
            self.labelList.append(node)
        else:
            self.labelList.append(node.label())
        return h,c

    def forward(self, binary_tree, targets):
        '''
        Args:
        binary_tree: a nltk binary tree
        targets: target words, a list 
        '''
        self.nodeProbList = []
        self.labelList = []
        #Get hidden states for the target words
        target_emb = self.embedding(Var(torch.LongTensor(targets)))
        _, (target_hidden, _) = self.lstm(target_emb.unsqueeze(0))
        self.target_hidden = target_hidden#1, 1, hidden_dim
        self.traverse(binary_tree)
        #self.labelList = Var(torch.cat(self.labelList))
        return torch.cat(self.nodeProbList)

    def getLoss(self, tree):
        nodes = self.forward(tree)
        predictions = nodes.max(dim=1)[1]
        loss = F.cross_entropy(input=nodes, target=self.labelList)
        return predictions,loss

    def evaluate(self, trees):
        pbar = progressbar.ProgressBar(widgets=widgets, maxval=len(trees)).start()
        n = nAll = correctRoot = correctAll = 0.0
        for j, tree in enumerate(trees):
            predictions,loss = self.getLoss(tree)
            correct = (predictions.data==self.labelList.data)
            correctAll += correct.sum()
            nAll += correct.squeeze().size()[0]
            correctRoot += correct.squeeze()[-1]
            n += 1
            pbar.update(j)
        pbar.finish()
        return correctRoot / n, correctAll/nAll
    
    def load_vector(self, embed_path):
        with open(embed_path, 'rb') as f:
            vectors = pickle.load(f)
            print("Loaded from {} with shape {}".format(embed_path, vectors.shape))
            self.embedding.weight.data.copy_(torch.from_numpy(vectors))
            self.embedding.weight.requires_grad = True