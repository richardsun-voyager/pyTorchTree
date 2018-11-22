from nltk.tree import ParentedTree
import _pickle as cPickle

class SenTreeOld(ParentedTree):
    def __init__(self, node, children=None):
        super(SenTreeOld,self).__init__(node, children)

    def left(self):
        return self[0]

    def right(self):
        return self[1]

    def isLeaf(self):
        return self.height()==2

    def getLeafWord(self):
        return self[0]


    @staticmethod
    def getTrees(file='./trees/small.txt',vocabOutFile=None, vocabIndicesMapFile=None):
         if vocabIndicesMapFile is None:
            return SenTreeOld.constructVocabAndGetTrees(file, vocabOutFile=vocabOutFile)
         else:
            return SenTreeOld.getTreesGivenVocab(file, vocabIndicesMapFile)

    @staticmethod
    def getTreesGivenVocab(file, vocabIndicesMapFile):
        trees = []
        vocabIndicesMap=cPickle.load(open(vocabIndicesMapFile,'rb'))
        with open(file, "r") as f:
            for line in f:
                tree = SenTree.fromstring(line)
                SenTreeOld.mapTreeNodes(tree,vocabIndicesMap)
                SenTreeOld.castLabelsToInt(tree)
                trees.append(tree)
        SenTree.vocabSize=len(vocabIndicesMap)
        return trees

    @staticmethod
    def constructVocabAndGetTrees(file, vocabOutFile=None):
        trees = []
        vocab = set()
        with open(file, "r") as f:
            for line in f:
                tree = SenTreeOld.fromstring(line)
                trees.append(tree)
                vocab.update(tree.leaves())
        vocabIndicesMap = dict(zip(vocab,range(len(vocab))))
        vocabIndicesMap['UNK'] = len(vocab)
        if vocabOutFile is not None:
            with open(vocabOutFile,'wb') as fp: cPickle.dump(vocabIndicesMap,fp)
        for tree in trees:
            SenTreeOld.mapTreeNodes(tree,vocabIndicesMap)
            SenTreeOld.castLabelsToInt(tree)
        SenTreeOld.vocabSize=len(vocabIndicesMap)
        return trees

    @staticmethod
    def mapTreeNodes(tree, vocabIndicesMap):
        for leafPos in tree.treepositions('leaves'):
            if tree[leafPos] in vocabIndicesMap: tree[leafPos] = vocabIndicesMap[tree[leafPos]]
            else: tree[leafPos]= vocabIndicesMap['UNK']

    @staticmethod
    def castLabelsToInt(tree):
        for subtree in tree.subtrees():
            subtree.set_label(int(subtree.label()))
