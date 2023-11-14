import ruamel.yaml

class ConfigGenerator:
    def __init__(self, class_name):
        self.class_name = class_name
        self.yaml = ruamel.yaml.YAML()
        self.yaml_config = ruamel.yaml.comments.CommentedMap()

    def add_key(self, key, value, comment=None, comment_before=None):
        self.yaml_config[key] = value
        if comment:
            self.yaml_config.yaml_set_comment_before_after_key(key, before=comment)
        if comment_before:
            self.yaml_config.yaml_set_comment_before_after_key(key, before=comment_before)

    def generate_config(self, output_file_path):
        with open(output_file_path, "w") as yaml_file:
            self.yaml.dump(self.yaml_config, yaml_file)




# # inference 과정
# if __name__ == "__main__":
#     class_name = "YourClassNameHere"
#     config_generator = ConfigGenerator(class_name)

#     # Add sections and keys with comments
#     config_generator.add_key("root_path", "data")
#     config_generator.add_key("seed", 3407, "The seed 3407 comes from the paper:\n''Torch.manual_seed(3407) is all you need:\nOn the influence of random seeds in deep learning architectures for computer vision''")
#     config_generator.add_key("search_hp", True)
#     config_generator.add_key("search_scale", [7, 7, 0.5])
#     config_generator.add_key("search_step", [200, 20, 5])
#     config_generator.add_key("init_beta", 1)
#     config_generator.add_key("init_alpha", 2)
#     config_generator.add_key("init_gamma", 0.1)
#     config_generator.add_key("best_beta", 0)
#     config_generator.add_key("best_alpha", 0)
#     config_generator.add_key("best_gamma", 0)
#     config_generator.add_key("eps", 0.001)
#     config_generator.add_key("feat_num", 500)
#     config_generator.add_key("w_training_free", [0.7, 0.3])
#     config_generator.add_key("w_training", [0.2, 0.8])
#     config_generator.add_key("dataset", class_name, comment_before="# ------ Basic Config ------")
#     config_generator.add_key("backbone", "ViT-B/32")
#     config_generator.add_key("lr", 0.0001)
#     config_generator.add_key("augment_epoch", 10)
#     config_generator.add_key("train_epoch", 20)

#     output_file_path = f"./configs/{class_name}.yaml"
#     config_generator.generate_config(output_file_path)
