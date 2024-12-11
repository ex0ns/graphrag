# Copyright (c) 2024 Microsoft Corporation.
# Licensed under the MIT License

"""All the steps to create the base entity graph."""

from typing import Any
from uuid import uuid4

import pandas as pd
from datashaper import (
    AsyncType,
    VerbCallbacks,
)

from graphrag.cache.pipeline_cache import PipelineCache
from graphrag.index.operations.create_graph import create_graph
from graphrag.index.operations.extract_entities import extract_entities
from graphrag.index.operations.snapshot import snapshot
from graphrag.index.operations.snapshot_graphml import snapshot_graphml
from graphrag.index.operations.summarize_descriptions import (
    summarize_descriptions,
)
from graphrag.storage.pipeline_storage import PipelineStorage


async def extract_graph(
    text_units: pd.DataFrame,
    callbacks: VerbCallbacks,
    cache: PipelineCache,
    storage: PipelineStorage,
    extraction_strategy: dict[str, Any] | None = None,
    extraction_num_threads: int = 4,
    extraction_async_mode: AsyncType = AsyncType.AsyncIO,
    entity_types: list[str] | None = None,
    summarization_strategy: dict[str, Any] | None = None,
    summarization_num_threads: int = 4,
    snapshot_graphml_enabled: bool = False,
    snapshot_transient_enabled: bool = False,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """All the steps to create the base entity graph."""
    entities, relationships = await extract_entities(
        text_units,
        callbacks,
        cache,
        text_column="text",
        id_column="id",
        strategy=extraction_strategy,
        async_mode=extraction_async_mode,
        entity_types=entity_types,
        num_threads=extraction_num_threads,
    )

    entity_summaries, relationship_summaries = await summarize_descriptions(
        entities,
        relationships,
        callbacks,
        cache,
        strategy=summarization_strategy,
        num_threads=summarization_num_threads,
    )

    base_relationship_edges = _prep_edges(relationships, relationship_summaries)

    base_entity_nodes = _prep_nodes(entities, entity_summaries)

    if snapshot_graphml_enabled:
        # todo: extract graphs at each level, and add in meta like descriptions
        graph = create_graph(base_relationship_edges)
        await snapshot_graphml(
            graph,
            name="graph",
            storage=storage,
        )

    if snapshot_transient_enabled:
        await snapshot(
            base_entity_nodes,
            name="base_entity_nodes",
            storage=storage,
            formats=["parquet"],
        )
        await snapshot(
            base_relationship_edges,
            name="base_relationship_edges",
            storage=storage,
            formats=["parquet"],
        )

    return (base_entity_nodes, base_relationship_edges)


def _prep_nodes(entities, summaries) -> pd.DataFrame:
    entities.drop(columns=["description"], inplace=True)
    nodes = (
        entities.merge(summaries, on="name", how="left")
        .drop_duplicates(subset="name")
        .rename(columns={"name": "title", "source_id": "text_unit_ids"})
    )
    nodes = nodes.loc[nodes["title"].notna()].reset_index()
    nodes["human_readable_id"] = nodes.index
    nodes["id"] = nodes["human_readable_id"].apply(lambda _x: str(uuid4()))
    return nodes


def _prep_edges(relationships, summaries) -> pd.DataFrame:
    edges = (
        relationships.drop(columns=["description"])
        .drop_duplicates(subset=["source", "target"])
        .merge(summaries, on=["source", "target"], how="left")
        .rename(columns={"source_id": "text_unit_ids"})
    )
    edges["human_readable_id"] = edges.index
    edges["id"] = edges["human_readable_id"].apply(lambda _x: str(uuid4()))
    return edges
