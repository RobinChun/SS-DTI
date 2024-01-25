from datetime import datetime


class hyperparameter():
    def __init__(self):
        self.current_time = datetime.now().strftime('%b%d_%H-%M-%S')
        self.Learning_rate = 5e-5
        self.Epoch = 60
        self.Batch_size = 32
        self.test_split = 0.2
        self.decay_interval = 10
        self.lr_decay = 0.5
        self.weight_decay = 1e-4
        self.protein_kernel = [3, 7, 15]
        self.drug_kernel = [3, 5, 7]
        self.char_dim = 128
