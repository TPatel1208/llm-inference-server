from llm_inference_server.kv_cache.block_allocator import BlockAllocator
import math
class BlockManager:
    def __init__(self, block_size: int,num_blocks: int):
        self.block_size = block_size
        self.allocator = BlockAllocator(num_blocks)
        self.block_table = {}
        self.num_tokens = {}

    def add_sequence(self,seq_id)->None:
        if seq_id in self.block_table:
            raise ValueError('Cannot have duplicate sequences')
        else:
            self.block_table[seq_id] = []
            self.num_tokens[seq_id] = 0
    def append_tokens(self,seq_id, n:int):
        if n<0:
            raise ValueError("n must be non-negative")
        t = self.num_tokens[seq_id]
        seq = self.block_table[seq_id]
        if (t+n)/self.block_size > len(seq):
            new_block_count = math.ceil(((t+n)-len(seq)*self.block_size)/self.block_size)
            allocated = self.allocator.allocate(new_block_count)
            self.block_table[seq_id].extend(allocated)
        self.num_tokens[seq_id] += n

    def free_sequence(self,seq_id)->None:
        if seq_id in self.block_table:
            self.allocator.free(self.block_table[seq_id])
            del self.block_table[seq_id]
            del self.num_tokens[seq_id]
        else:
            raise KeyError(f"seq_id {seq_id} does not exist")
    def get_block_table(self,seq_id):
        if seq_id in self.block_table:
            return self.block_table[seq_id]
        else:
            raise KeyError(f"seq_id {seq_id} does not exist")
    def get_length(self,seq_id):
        if seq_id in self.num_tokens:
            return self.num_tokens[seq_id]
        else:
            raise KeyError(f"seq_id {seq_id} does not exist")