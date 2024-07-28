from typing import Dict, Any, List, FrozenSet
from collections.abc import Iterable, Callable
import numpy as np

class Node:
    def __init__(self, features: tuple, values: tuple[str, ...], data: np.ndarray, parent, available_splitters: FrozenSet[Callable], threshold: int):
        self.features = features
        self.values = values
        self.data = data
        self.is_leaf = True
        self.parent=parent
        self.impurity = self.__entropy_1d(data)
        self.best_child_positive_rate = np.count_nonzero(data[:,-1]) / len(data)
        if len(data) >= threshold:
            best_splitter = None
            best_split = {}
            best_impurity = 100
            for splitter in available_splitters:
                split = self.__split(splitter, data)
                imp = self.__entropy(split.values())
                if len(split) > 1 and imp < best_impurity and all(map(lambda x: len(x) > (threshold/(len(split)*2)), split.values())):
                    best_splitter = splitter
                    best_split = split
                    best_impurity = imp
            
            if best_splitter == None:
                return
            
            self.is_leaf = False
            self.split = best_splitter
            self.post_split_impurity = best_impurity
            self.children = []

            for feature in best_split:
                if feature == True:
                    new_val = best_splitter.__name__
                elif feature == False:
                    new_val = f"NOT {best_splitter.__name__}"
                else:
                    new_val = f"{best_splitter.__name__}={feature}"
                
                self.children.append(Node(
                    features=(*features, feature), values=(*values, new_val), data=best_split[feature], parent=self, available_splitters=available_splitters.difference((best_splitter,)), threshold=threshold
                ))
            
            self.children: List[Node]
            self.children.sort(key=lambda x: -x.best_child_positive_rate)
            self.best_child_positive_rate = self.children[0].best_child_positive_rate
    
    def __split(self, splitter: Callable, data: np.ndarray) -> Dict[Any, np.ndarray]:
        assert data.ndim == 2
        result = {}
        split_vals = np.array([splitter(x[0]) for x in data])
        vals = set(split_vals)
        for val in vals:
            result[val] = data[split_vals == val]
        return result
        
    def __entropy_1d(self, labelled_data: np.ndarray) -> float:
        entropy = 0
        n = len(labelled_data)

        pos = np.count_nonzero(labelled_data[:,-1])
        if pos > 0:
            entropy -= (pos/n) * np.log2(pos/n)
        
        neg = n-pos
        if neg > 0:
            entropy -= (neg/n) * np.log2(neg/n)
        return entropy
    
    def __entropy(self, split_data: Iterable[np.ndarray]) -> float:
        total_len = 0
        entropy = 0
        for data in split_data:
            total_len += len(data)
            entropy += len(data)*self.__entropy_1d(data)
        assert total_len > 0
        return entropy/total_len

class bcolors:
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKCYAN = '\033[96m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'

bar, tee, elbow = "│   ", "├── ", "└── "
unit_length, split_padding = len(bar), 4

class DecisionTree:
    def __init__(self, data: Iterable, binary_labeller: Callable[[Any], bool], split_funcs: Iterable[Callable[[Any], Any]], threshold: int = 50):
        self.data = np.array([(x, binary_labeller(x)) for x in data])
        self.root = Node(features=tuple(), values=tuple("Root root"), data=self.data, parent=None, available_splitters=frozenset(split_funcs), threshold=threshold)
    
    def __string_bits(self, node: Node, prefix: str) -> List[str]:
        result = []

        next_pre_base, this_pre_end = prefix[0:-unit_length], prefix[-unit_length:]
        if this_pre_end == tee:
            next_pre = next_pre_base + bar + " " * split_padding
        else:
            next_pre = next_pre_base + " " * (unit_length + split_padding)
        
        if node.is_leaf:
            pos = np.count_nonzero(node.data[:, -1] == True)
            assert pos / len(node.data) == node.best_child_positive_rate
            result.append(f"{prefix}{bcolors.OKBLUE}{node.values[-1]}.{bcolors.ENDC} Entropy: {node.impurity:.3f}")
            no_data_warning = ""
            if len(node.data) == 0:
                no_data_warning = f"{bcolors.WARNING} - 0 data"
                rate = 0
            else:
                rate = pos / len(node.data)
            result.append(f"{next_pre}{bcolors.BOLD}{bcolors.OKBLUE}Positive Rate = {rate:.3f}{no_data_warning} (n={len(node.data)}){bcolors.ENDC}")
            result.append(next_pre)
        else:
            result.append(f"{prefix}{node.values[-1]}. Entropy: {node.impurity:.3f}")
            result.append(f"{next_pre}{bcolors.OKGREEN}{bcolors.BOLD}Split{bcolors.ENDC} using \"{node.split.__name__}\". Entropy Decrease = {node.impurity - node.post_split_impurity:.3f} ({node.impurity:.3f} ─> {node.post_split_impurity:.3f})")
            
            for child in node.children[0:-1]:
                result.extend(self.__string_bits(child, f"{next_pre}{tee}"))
            result.extend(self.__string_bits(node.children[-1], f"{next_pre}{elbow}"))
        
        return result
    
    def ansi_str(self) -> str:
        return "\n".join(self.__string_bits(self.root, "   "))
    
    def __str__(self) -> str:
        result = self.ansi_str()
        for substr in [bcolors.OKBLUE, bcolors.OKGREEN, bcolors.BOLD, bcolors.WARNING, bcolors.ENDC]:
            result = result.replace(substr, "")
        return result