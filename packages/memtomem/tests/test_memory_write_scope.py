"""ADR-0011 PR-D — write surface gate tests.

Three load-bearing pins for the memory write surface:

1. **Gate B explicit-flag-and-confirm.** ``mem_add(scope='project_shared',
   ...)`` without ``confirm_project_shared=True`` rejects with a clear
   error. Mirrored on ``mem_batch_add``.
2. **Gate A unbypassable on project_shared.** ``force_unsafe=True``
   plus ``scope='project_shared'`` plus a hit returns
   ``blocked_project_shared`` regardless of the surface (single
   ``mem_add``, batch ``mem_batch_add``).
3. **Inferred scope on edit.** ``mem_edit`` reads the loaded chunk's
   ``metadata.scope`` and feeds it to the guard — a client cannot
   bypass Gate A by omitting an explicit scope param.

The mocks in this file pre-stage the canonical pieces ``_mem_add_core``
calls: the embedding mismatch check, the AppContext, the index_engine
file-index, and the storage chunk lookup.
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import AsyncMock
from uuid import uuid4

import pytest

from helpers import StubCtx
from memtomem import privacy
from memtomem.models import Chunk, ChunkMetadata
from memtomem.server.context import AppContext
from memtomem.server.tools import memory_crud

_SECRET = "api_key=AKIA1234567890ABCDEF"


@pytest.fixture(autouse=True)
def _reset_counters():
    privacy.reset_for_tests()
    yield
    privacy.reset_for_tests()


# ---------------------------------------------------------------------------
# Gate B: explicit-flag-and-confirm
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_mem_add_project_shared_without_confirm_rejects(bm25_only_components):
    comp, _mem_dir = bm25_only_components
    app = AppContext.from_components(comp)
    ctx = StubCtx(app)
    out = await memory_crud.mem_add(
        content="harmless team rule",
        scope="project_shared",
        ctx=ctx,
    )
    assert "confirm_project_shared=True" in out
    assert "Error" in out


@pytest.mark.asyncio
async def test_mem_batch_add_project_shared_without_confirm_rejects(bm25_only_components):
    comp, _mem_dir = bm25_only_components
    app = AppContext.from_components(comp)
    ctx = StubCtx(app)
    out = await memory_crud.mem_batch_add(
        entries=[{"key": "k", "value": "harmless team rule"}],
        scope="project_shared",
        ctx=ctx,
    )
    assert "confirm_project_shared=True" in out
    assert "Error" in out


# ---------------------------------------------------------------------------
# Gate A: project_shared force_unsafe is hard-refused on every surface
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_mem_add_project_shared_force_unsafe_blocked(bm25_only_components):
    comp, _mem_dir = bm25_only_components
    app = AppContext.from_components(comp)
    ctx = StubCtx(app)
    out = await memory_crud.mem_add(
        content=_SECRET,
        scope="project_shared",
        confirm_project_shared=True,
        force_unsafe=True,
        ctx=ctx,
    )
    assert "force_unsafe=True is not permitted" in out
    assert "git history is forever" in out


@pytest.mark.asyncio
async def test_mem_batch_add_project_shared_force_unsafe_blocked(bm25_only_components):
    comp, _mem_dir = bm25_only_components
    app = AppContext.from_components(comp)
    ctx = StubCtx(app)
    out = await memory_crud.mem_batch_add(
        entries=[
            {"key": "clean", "value": "harmless"},
            {"key": "secret", "value": _SECRET},
        ],
        scope="project_shared",
        confirm_project_shared=True,
        force_unsafe=True,
        ctx=ctx,
    )
    assert "force_unsafe=True is not permitted" in out
    assert "Whole batch rejected" in out
    # The blocked_project_shared counter records once per hit item.
    snap = privacy.snapshot()
    assert snap["by_tool"]["mem_batch_add"]["blocked_project_shared"] == 1
    # Critically: clean entries do NOT register a pass on a rejected
    # batch (transactional reject preserved).
    assert snap["by_tool"]["mem_batch_add"]["pass"] == 0


# ---------------------------------------------------------------------------
# Default scope behavior preserved (user)
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_mem_add_default_user_scope_force_unsafe_still_works(bm25_only_components):
    """Existing user-scope force_unsafe path is unchanged by ADR-0011."""
    comp, _mem_dir = bm25_only_components
    app = AppContext.from_components(comp)
    ctx = StubCtx(app)
    out = await memory_crud.mem_add(
        content=_SECRET,
        force_unsafe=True,  # user scope by default
        ctx=ctx,
    )
    # No project_shared error; the old bypass path proceeds.
    assert "force_unsafe=True is not permitted" not in out
    assert "Memory added to" in out
    snap = privacy.snapshot()
    assert snap["by_tool"]["mem_add"]["bypassed"] == 1


# ---------------------------------------------------------------------------
# mem_edit inferred scope — gate sees chunk.metadata.scope
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_mem_edit_inferred_scope_blocks_project_shared_force_unsafe(
    bm25_only_components, monkeypatch
):
    comp, _mem_dir = bm25_only_components
    app = AppContext.from_components(comp)
    ctx = StubCtx(app)

    proj = Path("/tmp/proj_x")
    chunk_id = uuid4()
    fake_chunk = Chunk(
        content="original",
        metadata=ChunkMetadata(
            source_file=proj / ".memtomem" / "memories" / "x.md",
            scope="project_shared",
            project_root=proj,
        ),
        embedding=[0.1] * 1024,
    )
    monkeypatch.setattr(comp.storage, "get_chunk", AsyncMock(return_value=fake_chunk))

    out = await memory_crud.mem_edit(
        chunk_id=str(chunk_id),
        new_content=_SECRET,
        force_unsafe=True,
        ctx=ctx,
    )
    # The edit surface inferred scope=project_shared from the loaded
    # chunk's metadata; force_unsafe=True is hard-refused.
    assert "force_unsafe=True is not permitted" in out
    assert "git history is forever" in out
    snap = privacy.snapshot()
    assert snap["by_tool"]["mem_edit"]["blocked_project_shared"] == 1


@pytest.mark.asyncio
async def test_mem_edit_inferred_user_scope_force_unsafe_proceeds(
    bm25_only_components, monkeypatch, tmp_path
):
    """A user-scope chunk's edit surface still allows force_unsafe (no regression)."""
    comp, _mem_dir = bm25_only_components
    app = AppContext.from_components(comp)
    ctx = StubCtx(app)
    src = tmp_path / "u.md"
    src.write_text("## hi\n\noriginal\n")
    chunk_id = uuid4()
    fake_chunk = Chunk(
        content="original",
        metadata=ChunkMetadata(
            source_file=src,
            scope="user",
            project_root=None,
            start_line=1,
            end_line=3,
        ),
        embedding=[0.1] * 1024,
    )
    monkeypatch.setattr(comp.storage, "get_chunk", AsyncMock(return_value=fake_chunk))

    # Stub the file mutation + reindex so the test stays at the gate
    # boundary (file IO happens in real bm25 storage, but with a
    # synthetic chunk the line-replace + reindex pipeline isn't useful).
    async def fake_index_file(*args, **kwargs):
        from memtomem.models import IndexingStats

        return IndexingStats(0, 0, 0, 0, 0, 0.0)

    monkeypatch.setattr(comp.index_engine, "index_file", fake_index_file)

    out = await memory_crud.mem_edit(
        chunk_id=str(chunk_id),
        new_content=_SECRET,
        force_unsafe=True,
        ctx=ctx,
    )
    # No project_shared error — user-scope chunks still allow bypass.
    assert "force_unsafe=True is not permitted" not in out
    snap = privacy.snapshot()
    assert snap["by_tool"]["mem_edit"]["bypassed"] == 1
