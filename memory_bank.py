"""
GHOST System — Memory Bank (ChromaDB Vector Store)
Layer 3, Component 1: Store and retrieve spatial fingerprints using
cosine similarity search against a ChromaDB server running in Docker.

Input:  State Vector (6,) from FeatureExtractor
Output: Ranked candidate zones with similarity scores

The ChromaDB server must be running (docker compose up -d) before
using this module.

Usage:
    from memory_bank import MemoryBank
    bank = MemoryBank()
    bank.enroll(state_vector, zone="Kitchen", activity="Static")
    candidates = bank.query(state_vector, n_results=3)
"""

import logging
import time
import uuid
from dataclasses import dataclass

import chromadb
import numpy as np

from config import CHROMA_SETTINGS, ChromaSettings
from feature_extractor import NUM_FEATURES

logger = logging.getLogger("ghost.memory_bank")


# ---------------------------------------------------------------------------
# Candidate — single query result
# ---------------------------------------------------------------------------
@dataclass
class Candidate:
    """A single query result from the memory bank.

    Attributes:
        zone: The zone label (e.g., "Kitchen").
        activity: The activity label (e.g., "Static").
        label: Human-readable description.
        score: Cosine similarity score (0.0–1.0, higher = more similar).
        fingerprint_id: The ChromaDB document ID.
    """

    zone: str
    activity: str
    label: str
    score: float
    fingerprint_id: str


# ---------------------------------------------------------------------------
# MemoryBank — ChromaDB client
# ---------------------------------------------------------------------------
class MemoryBank:
    """ChromaDB-backed vector store for spatial fingerprints.

    Connects to a ChromaDB server over HTTP and manages a collection
    of 6-element state-vector embeddings with zone/activity metadata.

    Args:
        settings: ChromaDB connection settings.  Uses defaults if None.
    """

    def __init__(self, settings: ChromaSettings | None = None):
        self._settings = settings or CHROMA_SETTINGS
        self._client = chromadb.HttpClient(
            host=self._settings.host,
            port=self._settings.port,
        )
        self._collection = self._client.get_or_create_collection(
            name=self._settings.collection_name,
            metadata={"hnsw:space": "cosine"},
        )
        logger.info(
            "MemoryBank connected to %s:%d, collection '%s' (%d fingerprints)",
            self._settings.host,
            self._settings.port,
            self._settings.collection_name,
            self._collection.count(),
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def enroll(
        self,
        state_vector: np.ndarray,
        zone: str,
        activity: str,
        label: str = "",
        fingerprint_id: str | None = None,
    ) -> str:
        """Store a labeled fingerprint in the memory bank.

        Args:
            state_vector: Shape (6,) float64 from FeatureExtractor.
            zone: Zone label (e.g., "Kitchen", "Bedroom").
            activity: Activity label (e.g., "Static", "Walking").
            label: Optional human-readable description.
            fingerprint_id: Optional explicit ID.  Auto-generated if None.

        Returns:
            The fingerprint ID that was stored.
        """
        self._validate_vector(state_vector)

        fid = fingerprint_id or (
            f"{zone.lower()}_{activity.lower()}_{uuid.uuid4().hex[:8]}"
        )
        metadata = {
            "zone": zone,
            "activity": activity,
            "label": label or f"{activity} in {zone}",
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
        }

        self._collection.add(
            ids=[fid],
            embeddings=[state_vector.tolist()],
            metadatas=[metadata],
            documents=[metadata["label"]],
        )

        logger.info(
            "Enrolled fingerprint '%s': zone=%s activity=%s",
            fid, zone, activity,
        )
        return fid

    def enroll_batch(
        self,
        state_vectors: list[np.ndarray],
        zones: list[str],
        activities: list[str],
        labels: list[str] | None = None,
    ) -> list[str]:
        """Store multiple fingerprints in a single batch.

        Args:
            state_vectors: List of (6,) arrays.
            zones: Zone labels, same length as state_vectors.
            activities: Activity labels, same length as state_vectors.
            labels: Optional descriptions, same length.

        Returns:
            List of generated fingerprint IDs.
        """
        n = len(state_vectors)
        if len(zones) != n or len(activities) != n:
            raise ValueError("All lists must have the same length")
        if labels and len(labels) != n:
            raise ValueError("labels must match state_vectors length")

        ids = []
        embeddings = []
        metadatas = []
        documents = []

        for i in range(n):
            self._validate_vector(state_vectors[i])
            fid = (
                f"{zones[i].lower()}_{activities[i].lower()}_"
                f"{uuid.uuid4().hex[:8]}"
            )
            lbl = (labels[i] if labels else "") or f"{activities[i]} in {zones[i]}"

            ids.append(fid)
            embeddings.append(state_vectors[i].tolist())
            metadatas.append({
                "zone": zones[i],
                "activity": activities[i],
                "label": lbl,
                "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
            })
            documents.append(lbl)

        self._collection.add(
            ids=ids,
            embeddings=embeddings,
            metadatas=metadatas,
            documents=documents,
        )

        logger.info("Enrolled %d fingerprints in batch", n)
        return ids

    def query(
        self,
        state_vector: np.ndarray,
        n_results: int = 3,
        zone_filter: str | None = None,
    ) -> list[Candidate]:
        """Query the memory bank for the closest matching fingerprints.

        Args:
            state_vector: Current state vector, shape (6,).
            n_results: Number of candidates to return.
            zone_filter: Optional — restrict results to a specific zone.

        Returns:
            List of Candidate objects sorted by descending similarity.
        """
        self._validate_vector(state_vector)

        where_filter = {"zone": zone_filter} if zone_filter else None

        results = self._collection.query(
            query_embeddings=[state_vector.tolist()],
            n_results=min(n_results, self._collection.count() or 1),
            where=where_filter,
            include=["metadatas", "distances"],
        )

        candidates = []
        if results["ids"] and results["ids"][0]:
            for i, fid in enumerate(results["ids"][0]):
                meta = results["metadatas"][0][i]
                # ChromaDB returns cosine *distance* (0 = identical).
                # Convert to similarity: similarity = 1 - distance.
                distance = results["distances"][0][i]
                similarity = 1.0 - distance

                candidates.append(Candidate(
                    zone=meta.get("zone", "Unknown"),
                    activity=meta.get("activity", "Unknown"),
                    label=meta.get("label", ""),
                    score=round(similarity, 4),
                    fingerprint_id=fid,
                ))

        logger.info(
            "Query returned %d candidates (top: %s %.4f)",
            len(candidates),
            candidates[0].zone if candidates else "none",
            candidates[0].score if candidates else 0.0,
        )
        return candidates

    def count(self) -> int:
        """Return the number of fingerprints in the collection."""
        return self._collection.count()

    def clear(self) -> None:
        """Delete all fingerprints and recreate the collection.

        WARNING: This is destructive and cannot be undone.
        """
        name = self._settings.collection_name
        self._client.delete_collection(name)
        self._collection = self._client.get_or_create_collection(
            name=name,
            metadata={"hnsw:space": "cosine"},
        )
        logger.warning("Collection '%s' cleared", name)

    def health_check(self) -> bool:
        """Verify that the ChromaDB server is reachable.

        Returns:
            True if the server responds to a heartbeat.
        """
        try:
            self._client.heartbeat()
            return True
        except Exception as e:
            logger.error("ChromaDB health check failed: %s", e)
            return False

    def get_all_zones(self) -> list[str]:
        """Return a sorted list of all unique zone names in the collection."""
        results = self._collection.get(include=["metadatas"])
        zones: set[str] = set()
        for meta in results.get("metadatas", []):
            if meta and "zone" in meta:
                zones.add(meta["zone"])
        return sorted(zones)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------
    @staticmethod
    def _validate_vector(state_vector: np.ndarray) -> None:
        """Raise ValueError if the vector is not a valid 6-element array."""
        if not isinstance(state_vector, np.ndarray):
            raise TypeError(
                f"Expected np.ndarray, got {type(state_vector).__name__}"
            )
        if state_vector.shape != (NUM_FEATURES,):
            raise ValueError(
                f"Expected shape ({NUM_FEATURES},), got {state_vector.shape}"
            )