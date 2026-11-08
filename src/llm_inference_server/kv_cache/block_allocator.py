class OutofBlocksError(Exception):
    pass
class DoubleFreeError(Exception):
    pass
class BlockAllocator:
    def __init__(self, num_blocks):
        if not isinstance(num_blocks,int):
            raise TypeError("value must be an integer")
        if num_blocks < 0:
            raise ValueError("num_blocks must be non-negative")
        
        self.num_blocks = num_blocks
        self.free_blocks = {i: None for i in range(num_blocks)}
    @property
    def num_free(self)->int:
        return len(self.free_blocks)

    def allocate(self, n:int)->list[int]:
        if n<0:
            raise ValueError("n must be non-negative")
        if self.num_free < n:
            raise OutofBlocksError("Not enough free blocks")
        else:
            removed = []
            for _ in range(n):
                key, _ = self.free_blocks.popitem()
                removed.append(key)
            return removed

    def free(self, block_ids: list[int])->None:
        seen_ids = set()
        for block in block_ids:
            if not isinstance(block,int):
                raise TypeError("value must be an integer")
            if block in seen_ids:
                raise ValueError('Duplicate blocks in block_ids')
            seen_ids.add(block)
            if block in self.free_blocks:
                raise DoubleFreeError('Cannot free block more than once')
            if block not in range(self.num_blocks):
                raise ValueError(f"Block must be in range of allocable blocks 0 to {self.num_blocks-1}")
            
        for block in block_ids:
            self.free_blocks[block] = None
    