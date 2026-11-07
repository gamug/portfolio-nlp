"""The shape of the sentiment training set the owner tags (PLAN D35): blocks, batches, slices.

The 450 summaries come in blocks of ``BLOCK`` rows with the same number of each predicted class, so
any whole number of blocks is balanced. A batch is ``BLOCKS_PER_BATCH`` blocks; the last
``VALIDATION_BLOCKS`` blocks of every batch are the **validation** slice (it picks the settings and
is never trained on), the others are ``train``. The draw writes these, the builder of the marks
checks them, and the trainer reads them: this is the one place the rule is written.
"""

from __future__ import annotations

BLOCK = 15  # rows per block: BLOCK / 3 of each predicted class
BLOCKS_PER_BATCH = 10  # 150 rows
VALIDATION_BLOCKS = 2  # the last blocks of every batch
TRAIN, VALIDATION = "train", "validation"


def batch_of(block: int) -> int:
    """The batch (from 1) a block (from 1) belongs to."""
    return (block - 1) // BLOCKS_PER_BATCH + 1


def slice_of(block: int) -> str:
    """``validation`` for the last ``VALIDATION_BLOCKS`` blocks of a batch, else ``train``."""
    in_batch = (block - 1) % BLOCKS_PER_BATCH + 1
    return VALIDATION if in_batch > BLOCKS_PER_BATCH - VALIDATION_BLOCKS else TRAIN
