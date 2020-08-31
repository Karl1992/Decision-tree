import numpy as np
import utils as Util


class DecisionTree():
    def __init__(self):
        self.clf_name = "DecisionTree"
        self.root_node = None

    def train(self, features, labels):
        # features: List[List[float]], labels: List[int]
        # init
        assert (len(features) > 0)
        num_cls = np.unique(labels).size

        # build the tree
        self.root_node = TreeNode(features, labels, num_cls)
        if self.root_node.splittable:
            self.root_node.split()

        return

    def predict(self, features):
        # features: List[List[any]]
        # return List[int]
        y_pred = []
        for idx, feature in enumerate(features):
            pred = self.root_node.predict(feature)
            y_pred.append(pred)
        return y_pred


class TreeNode(object):
    def __init__(self, features, labels, num_cls):
        # features: List[List[any]], labels: List[int], num_cls: int
        self.features = features
        self.labels = labels
        self.children = []
        self.num_cls = num_cls
        # find the most common labels in current node, compute S
        self.S = 0
        count_max = 0
        for label in np.unique(labels):
            self.S -= self.labels.count(label)/len(self.labels) * np.log2(self.labels.count(label)/len(self.labels))
            if self.labels.count(label) > count_max:
                count_max = labels.count(label)
                self.cls_max = label
                # splitable is false when all features belongs to one class
        if len(np.unique(labels)) < 2:
            self.splittable = False
        else:
            self.splittable = True

        self.dim_split = None  # the index of the feature to be split
        self.feature_uniq_split = None  # the possible unique values of the feature to be split

    #TODO: try to split current node
    def split(self):
        IG_best = 0
        num_atr = 0
        feature_array = np.array(self.features)

        # construct a dictionary to find the label index
        labels_dict = {}
        for i in range(self.num_cls):
            labels_dict[np.unique(self.labels)[i]] = i

        for i in range(len(self.features[1])):
            # construct a dictionary to find the attribute index in this feature
            attribute_dict = {}
            this_attribute = feature_array[:, i]
            this_feature_unique_split = np.unique(this_attribute)
            for attribute in range(this_feature_unique_split.size):
                attribute_dict[this_feature_unique_split[attribute]] = attribute

            # initialize the children of depending on this feature
            this_children = []
            this_children_features = [[] for _ in range(this_feature_unique_split.size)]
            this_children_labels = [[] for _ in range(this_feature_unique_split.size)]
            this_children_num_cls = []

            # initialize branches of this feature to compute IG
            this_branches = [[0 for _ in range(self.num_cls)] for _ in range(this_feature_unique_split.size)]

            # Travel all elements in this feature, store this child feature and labels to related branches
            for j in range(len(self.features)):
                # add this obs to related branches
                this_children_features[attribute_dict[feature_array[j, i]]].append(self.features[j])

                # add the labels to related branches
                this_children_labels[attribute_dict[feature_array[j, i]]].append(self.labels[j])

                # add 1 to elements have related attribute and class
                this_branches[attribute_dict[feature_array[j, i]]][labels_dict[self.labels[j]]] += 1

            for k in range(len(this_children_features)):
                # compute number of class in each branch and store each node to children of this feature
                this_children_num_cls.append(len(np.unique(this_children_labels[k])))
                this_child = TreeNode(this_children_features[k],this_children_labels[k], this_children_num_cls[k])
                this_children.append(this_child)

            this_IG = Util.Information_Gain(self.S, this_branches)

            if this_IG > IG_best:
                IG_best = this_IG
                self.children = this_children
                self.feature_uniq_split = this_feature_unique_split
                self.dim_split = i
                num_atr = len(this_feature_unique_split)
            elif this_IG == IG_best:
                if len(this_feature_unique_split) > num_atr:
                    self.children = this_children
                    self.feature_uniq_split = this_feature_unique_split
                    self.dim_split = i
                    num_atr = len(this_feature_unique_split)

        for node in range(len(self.children)):
            if self.children[node].splittable:
                self.children[node].split()

    # TODO: predict the branch or the class
    def predict(self, feature):
        # feature: List[any]
        # return: int
        if self.splittable:
            nextnode = self.children[list(self.feature_uniq_split).index(feature[self.dim_split])]
            nextnode.predict(feature)
            return
        else:
            return 1


