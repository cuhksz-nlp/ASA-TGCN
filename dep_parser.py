import copy

class DepTreeParser():
    def __init__(self):
        pass

    def parsing(self, sentence):
        pass

class DepInstanceParser():
    def __init__(self, basicDependencies, tokens):
        self.basicDependencies = basicDependencies
        self.tokens = tokens
        self.words = []
        self.dep_governed_info = []
        self.dep_parsing()


    def dep_parsing(self):
        words = []
        for token in self.tokens:
            token['word'] = token['word'].replace('\xa0', '')
            words.append(self.change_word(token['word']))
        dep_governed_info = [
            {"word": word}
            for i,word in enumerate(words)
        ]
        for dep in self.basicDependencies:
            dependent_index = dep['dependent'] - 1
            governed_index = dep['governor'] - 1
            dep_governed_info[dependent_index] = {
                "governor": governed_index,
                "dep": dep['dep']
            }
        self.words = words
        self.dep_governed_info = dep_governed_info

    def change_word(self, word):
        if "-RRB-" in word:
            return word.replace("-RRB-", ")")
        if "-LRB-" in word:
            return word.replace("-LRB-", "(")
        return word

    def get_first_order(self, direct=False):
        dep_adj_matrix  = [[0] * len(self.words) for _ in range(len(self.words))]
        dep_type_matrix = [["none"] * len(self.words) for _ in range(len(self.words))]
        for i, dep_info in enumerate(self.dep_governed_info):
            governor = dep_info["governor"]
            dep_type = dep_info["dep"]
            dep_adj_matrix[i][governor] = 1
            dep_adj_matrix[governor][i] = 1
            dep_type_matrix[i][governor] = dep_type if direct is False else "{}_in".format(dep_type)
            dep_type_matrix[governor][i] = dep_type if direct is False else "{}_out".format(dep_type)
        return dep_adj_matrix, dep_type_matrix

    def get_next_order(self, dep_adj_matrix, dep_type_matrix):
        new_dep_adj_matrix = copy.deepcopy(dep_adj_matrix)
        new_dep_type_matrix = copy.deepcopy(dep_type_matrix)
        for target_index in range(len(dep_adj_matrix)):
            for first_order_index in range(len(dep_adj_matrix[target_index])):
                if dep_adj_matrix[target_index][first_order_index] == 0:
                    continue
                for second_order_index in range(len(dep_adj_matrix[first_order_index])):
                    if dep_adj_matrix[first_order_index][second_order_index] == 0:
                        continue
                    if second_order_index == target_index:
                        continue
                    if new_dep_adj_matrix[target_index][second_order_index] == 1:
                        continue
                    new_dep_adj_matrix[target_index][second_order_index] = 1
                    new_dep_type_matrix[target_index][second_order_index] = dep_type_matrix[first_order_index][second_order_index]
        return new_dep_adj_matrix, new_dep_type_matrix

    def get_second_order(self, direct=False):
        dep_adj_matrix, dep_type_matrix = self.get_first_order(direct=direct)
        return self.get_next_order(dep_adj_matrix, dep_type_matrix)

    def get_third_order(self, direct=False):
        dep_adj_matrix, dep_type_matrix = self.get_second_order(direct=direct)
        return self.get_next_order(dep_adj_matrix, dep_type_matrix)

    def search_dep_path(self, start_idx, end_idx, adj_max, dep_path_arr):
        for next_id in range(len(adj_max[start_idx])):
            if next_id in dep_path_arr or adj_max[start_idx][next_id] in ["none"]:
                continue
            if next_id == end_idx:
                return 1, dep_path_arr + [next_id]
            stat, dep_arr = self.search_dep_path(next_id, end_idx, adj_max, dep_path_arr + [next_id])
            if stat == 1:
                return stat, dep_arr
        return 0, []

    def get_dep_path(self, start_index, end_index, direct=False):
        dep_adj_matrix, dep_type_matrix = self.get_first_order(direct=direct)
        _, dep_path = self.search_dep_path(start_index, end_index, dep_type_matrix, [start_index])
        return dep_path

