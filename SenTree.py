from nltk.tree import ParentedTree
import _pickle as cPickle

class SenTree(ParentedTree):
    def __init__(self, node, children=None):
        super(SenTree,self).__init__(node, children)

    def left(self):
        return self[0]

    def right(self):
        return self[1]

    def isLeaf(self):
        return self.height()==2

    def getLeafWord(self):
        return self[0]


    @staticmethod
    def getTrees(file='trees/tree.txt',vocabOutFile=None, vocabIndicesMapFile='trees/vocab/dict.pkl'):
         if vocabIndicesMapFile is None:
            return SenTree.constructVocabAndGetTrees(file, vocabOutFile=vocabOutFile)
         else:
            return SenTree.getTreesGivenVocab(file, vocabIndicesMapFile)

    @staticmethod
    def getTreesGivenVocab(file, vocabIndicesMapFile='trees/vocab/dict.pkl'):
        trees = []
        labels = []
        targets = []
        word2id, _, _ = cPickle.load(open(vocabIndicesMapFile,'rb'))
        vocabIndicesMap = word2id#map a word to an id
        with open(file, "r") as f:
            for line in f:
                item = line.split('||')
                tree = SenTree.fromstring(item[0])
                SenTree.mapTreeNodes(tree,vocabIndicesMap)
                SenTree.castLabelsToInt(tree)
                trees.append(tree)
                labels.append(int(item[1].strip()))
                target_words = item[2].strip().split()
                targets.append([SenTree.get_word_id(word2id, w) for w in target_words])
        SenTree.vocabSize=len(vocabIndicesMap)
        return trees, targets, labels
    
    @staticmethod
    def get_word_id(word2id, word):
        '''
        Find the id for each word, if not, replace with 'unk'
        '''
        try:
            index = word2id[word]
        except:
            index = word2id['unk']
        return index

    @staticmethod
    def constructVocabAndGetTrees(file, vocabOutFile=None):
        trees = []
        vocab = set()
        with open(file, "r") as f:
            for line in f:
                tree = SenTree.fromstring(line)
                trees.append(tree)
                vocab.update(tree.leaves())
        vocabIndicesMap = dict(zip(vocab,range(len(vocab))))
        vocabIndicesMap['UNK'] = len(vocab)
        if vocabOutFile is not None:
            with open(vocabOutFile,'wb') as fp: cPickle.dump(vocabIndicesMap,fp)
        for tree in trees:
            SenTree.mapTreeNodes(tree,vocabIndicesMap)
            SenTree.castLabelsToInt(tree)
        SenTree.vocabSize=len(vocabIndicesMap)
        return trees

    @staticmethod
    def mapTreeNodes(tree, vocabIndicesMap):
        for leafPos in tree.treepositions('leaves'):
            if tree[leafPos] in vocabIndicesMap: tree[leafPos] = vocabIndicesMap[tree[leafPos]]
            else: tree[leafPos]= vocabIndicesMap['unk']

    @staticmethod
    def castLabelsToInt(tree):
        for subtree in tree.subtrees():
            subtree.set_label(subtree.label())

#trees = SenTree.getTrees()





