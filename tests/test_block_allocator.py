import pytest

from llm_inference_server.kv_cache.block_allocator import BlockAllocator, DoubleFreeError, OutofBlocksError


def test_allocate_returns_distinct_ids_in_range():
    allocator = BlockAllocator(8)
    ids = allocator.allocate(5)
    assert len(ids) == 5
    assert len(set(ids)) == 5
    assert all(0 <= block < 8 for block in ids)
    assert allocator.num_free == 3


def test_allocate_more_than_free_raises_and_changes_nothing():
    allocator = BlockAllocator(4)
    allocator.allocate(3)
    with pytest.raises(OutofBlocksError):
        allocator.allocate(2)
    assert allocator.num_free == 1


def test_freed_blocks_can_be_allocated_again():
    allocator = BlockAllocator(4)
    ids = allocator.allocate(4)
    assert allocator.num_free == 0
    allocator.free(ids)
    assert allocator.num_free == 4
    assert sorted(allocator.allocate(4)) == [0, 1, 2, 3]


def test_double_free_raises():
    allocator = BlockAllocator(4)
    ids = allocator.allocate(2)
    allocator.free(ids)
    with pytest.raises(DoubleFreeError):
        allocator.free(ids)


def test_free_with_one_bad_id_changes_nothing():
    allocator = BlockAllocator(4)
    ids = allocator.allocate(2)
    before = allocator.num_free
    with pytest.raises(ValueError):
        allocator.free([ids[0], 99])
    assert allocator.num_free == before        
    allocator.free(ids)                      
    assert allocator.num_free == 4