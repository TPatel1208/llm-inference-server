import pytest

from llm_inference_server.kv_cache.block_allocator import OutofBlocksError
from llm_inference_server.kv_cache.block_manager import BlockManager


def test_append_allocates_blocks_only_when_needed():
    manager = BlockManager(block_size=16, num_blocks=8)
    manager.add_sequence("a")
    expected = [(20, 2), (12, 2), (1, 3), (5, 3)] 
    total = 0
    for n, blocks in expected:
        manager.append_tokens("a", n)
        total += n
        assert manager.get_length("a") == total
        assert len(manager.get_block_table("a")) == blocks


def test_exact_fill_does_not_allocate_an_extra_block():
    manager = BlockManager(block_size=16, num_blocks=8)
    manager.add_sequence("a")
    manager.append_tokens("a", 16)
    assert len(manager.get_block_table("a")) == 1
    manager.append_tokens("a", 16)
    assert len(manager.get_block_table("a")) == 2


def test_failed_append_changes_nothing():
    manager = BlockManager(block_size=16, num_blocks=2)
    manager.add_sequence("a")
    manager.append_tokens("a", 20)
    with pytest.raises(OutofBlocksError):
        manager.append_tokens("a", 100)
    assert manager.get_length("a") == 20
    assert len(manager.get_block_table("a")) == 2
    assert manager.allocator.num_free == 0


def test_sequences_hold_disjoint_blocks():
    manager = BlockManager(block_size=16, num_blocks=8)
    manager.add_sequence("a")
    manager.add_sequence("b")
    manager.append_tokens("a", 40)
    manager.append_tokens("b", 40)
    assert not set(manager.get_block_table("a")) & set(manager.get_block_table("b"))


def test_free_sequence_returns_blocks_and_forgets_sequence():
    manager = BlockManager(block_size=16, num_blocks=8)
    manager.add_sequence("a")
    manager.append_tokens("a", 40)
    assert manager.allocator.num_free == 5
    manager.free_sequence("a")
    assert manager.allocator.num_free == 8
    with pytest.raises(KeyError):
        manager.get_length("a")
