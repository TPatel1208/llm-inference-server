import torch

class KVCache:
    def __init__(self):
        self.cache = {'key': None, 'value': None}

    def update(self,key, value):
        if self.cache['key'] is None:
            self.cache['key'] = key
            self.cache['value'] = value
        else:
            self.cache['key'] = torch.cat([self.cache['key'],key],dim=1)
            self.cache['value'] = torch.cat([self.cache['value'],value],dim=1)

    def get_cache(self):
        return self.cache