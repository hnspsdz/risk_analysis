import sys
import os

curPath = os.path.abspath(os.path.dirname(__file__))
rootPath = os.path.split(curPath)[0]
datasetPath = './'
sys.path.append(rootPath)

"""
datasets selection
0: bird-40
"""

global_data_selection = 1
global_deep_learning_selection = 1

class Configuration(object):
    def __init__(self, data_selection, deep_learning_selection):
        self.data_selection = data_selection
        self.deep_learning_selection = deep_learning_selection

        # setting risk dataset
        # 2020.07.11 remove the datasets file to parent directory
        self.data_dict = {
            1:'chest_hosp0',
            2:'chest_RSNA100',
        }
        self.class_num_dict = {
            1: 2,
            2: 2,
        }

        self.data_path = self.data_dict[self.data_selection]
        self.image_dataset_path = os.path.join(self.data_path, 'image_dataset')
        self.risk_dataset_path = os.path.join(self.data_path, 'risk_dataset')
        self.npy_dataset_path = os.path.join(self.data_path, 'npy_dataset')
        self.data2csv_path = os.path.join(self.data_path, 'data2csv')
        self.data2mulcsv_path = os.path.join(self.data_path, 'data2mulcsv')
        self.rules_path = os.path.join(self.data_path, 'rules')
        self.base_risk_nums = 10
        self.base_risk_list = ['den169', 'res101', 'res50']


        # setting epochs
        # these parameters are not used now
        self.train_size = 20
        self.deep_learning_epochs = 1
        self.risk_epochs = 100


        self.interval_number_4_continuous_value = 50
        # self.learing_rate = 0.001    default

        self.learing_rate = 0.0005
        self.risk_training_epochs = 50
        self.learn_variance = True
        self.apply_function_to_weight_classifier_output = True
        self.minimum_observation_num = 0.0
        self.rule_acc = 0.0
        self.risk_confidence = 0.90
        self.model_save_path = os.path.join(self.risk_dataset_path, 'tf_model')


        # setting decision_tree
        self.match_gini = 0.2
        self.unmatch_gini = 0.0001
        self.tree_depth = 1
        self.generate_rules = False

        self.raw_data_path = None
        self.raw_decision_tree_rules_path = None
        self.decision_tree_rules_path = os.path.join(self.risk_dataset_path, 'decision_tree_rules_clean.txt')
        self.info_decision_tree_rules_path = os.path.join(self.risk_dataset_path, 'decision_tree_rules_info.txt')
        self.train = None

        # the frame is used
        self.risk_model_type = 'f'  # torch or tf

    def get_parent_path(self):
        return self.risk_dataset_path

    def get_npy_dataset_path(self):
        return self.npy_dataset_path

    def get_data2csv_path(self):
        return self.data2csv_path

    def get_data2mulcsv_path(self):
        return self.data2mulcsv_path

    def get_risk_dataset_path(self):
        return self.risk_dataset_path

    def get_class_num(self):
        return self.class_num_dict[self.data_selection]

    def get_rules_dataset_path(self):
        return self.rules_path

    def get_raw_decision_tree_rules_path(self):
        return self.raw_decision_tree_rules_path

    def get_info_decision_tree_rules_path(self):
        return self.info_decision_tree_rules_path

    def get_decision_tree_rules_path(self):
        return self.decision_tree_rules_path

    def get_raw_data_path(self):
        return self.raw_data_path

    def get_train_path(self):
        return self.train

    def get_all_data_path(self):
        return os.path.join(self.risk_dataset_path, 'all_data_info.csv')
