import shutil
import uuid
from pathlib import Path
from typing import Any

import chromadb
from sentence_transformers import SentenceTransformer


class CodeVectorStore:
    """
    RAG vector store for project source code.

    A unique Chroma collection is created for every pipeline run.
    This prevents:
    - "Collection already exists" errors
    - previous project data contamination
    - conflicts when the Flask app processes multiple requests
    """

    def __init__(
        self,
        persist_directory: str = "output/chroma_db",
        model_name: str = "all-MiniLM-L6-v2",
    ):
        self.persist_directory = Path(persist_directory)
        self.model_name = model_name

        print("Loading embedding model...")

        self.embedding_model = SentenceTransformer(
            model_name
        )

        # ---------------------------------------------------------
        # Create persistent Chroma database
        # ---------------------------------------------------------

        self.persist_directory.mkdir(
            parents=True,
            exist_ok=True
        )

        self.client = chromadb.PersistentClient(
            path=str(self.persist_directory)
        )

        # ---------------------------------------------------------
        # IMPORTANT:
        # Use a UNIQUE collection name for every run.
        #
        # This completely avoids:
        # Collection [project_source] already exists
        # ---------------------------------------------------------

        unique_id = uuid.uuid4().hex[:12]

        self.collection_name = (
            f"project_source_{unique_id}"
        )

        self.collection = (
            self.client.create_collection(
                name=self.collection_name
            )
        )

        print(
            f"Chroma collection created: "
            f"{self.collection_name}"
        )

    # ============================================================
    # ADD CHUNKS
    # ============================================================

    def add_chunks(
        self,
        chunks: list[Any]
    ):
        if not chunks:
            print("No chunks to store.")
            return

        documents = []
        metadatas = []
        ids = []

        # --------------------------------------------------------
        # Prepare documents
        # --------------------------------------------------------

        for index, chunk in enumerate(chunks):

            # ----------------------------------------------------
            # String chunk
            # ----------------------------------------------------

            if isinstance(chunk, str):

                text = chunk

                metadata = {
                    "chunk_index": index
                }

            # ----------------------------------------------------
            # Dictionary chunk
            # ----------------------------------------------------

            elif isinstance(chunk, dict):

                text = str(
                    chunk.get(
                        "text",
                        chunk.get(
                            "content",
                            ""
                        )
                    )
                )

                metadata = {
                    "chunk_index": index
                }

                # Preserve useful metadata
                for key in [
                    "file",
                    "filename",
                    "path",
                    "source",
                    "start_line",
                    "end_line",
                ]:

                    if key in chunk:

                        value = chunk[key]

                        if value is not None:

                            metadata[key] = str(
                                value
                            )

            # ----------------------------------------------------
            # Generic object
            # ----------------------------------------------------

            else:

                text = str(chunk)

                metadata = {
                    "chunk_index": index
                }

            text = text.strip()

            if not text:
                continue

            documents.append(text)

            metadatas.append(metadata)

            ids.append(
                f"chunk_{index}"
            )

        # --------------------------------------------------------
        # Nothing readable
        # --------------------------------------------------------

        if not documents:

            print(
                "No readable chunks found."
            )

            return

        # --------------------------------------------------------
        # Generate embeddings
        # --------------------------------------------------------

        embeddings = (
            self.embedding_model.encode(
                documents,
                show_progress_bar=False
            )
        )

        # Convert NumPy arrays to normal lists
        embeddings = [
            vector.tolist()
            for vector in embeddings
        ]

        # --------------------------------------------------------
        # Store in Chroma
        # --------------------------------------------------------

        self.collection.add(
            ids=ids,
            documents=documents,
            embeddings=embeddings,
            metadatas=metadatas
        )

        print(
            f"Stored {len(documents)} "
            "chunks in ChromaDB."
        )

    # ============================================================
    # SEARCH
    # ============================================================

    def search(
        self,
        query: str,
        top_k: int = 3
    ):

        query = str(query).strip()

        if not query:
            return []

        # --------------------------------------------------------
        # Query embedding
        # --------------------------------------------------------

        query_embedding = (
            self.embedding_model.encode(
                [query],
                show_progress_bar=False
            )[0]
            .tolist()
        )

        # --------------------------------------------------------
        # Chroma search
        # --------------------------------------------------------

        results = (
            self.collection.query(
                query_embeddings=[
                    query_embedding
                ],
                n_results=top_k,
                include=[
                    "documents",
                    "metadatas",
                    "distances",
                ]
            )
        )

        documents = (
            results.get(
                "documents",
                [[]]
            )
        )

        if not documents:
            return []

        return documents[0]

    # ============================================================
    # SEARCH WITH METADATA
    # ============================================================

    def search_with_metadata(
        self,
        query: str,
        top_k: int = 3
    ):

        query = str(query).strip()

        if not query:
            return []

        # --------------------------------------------------------
        # Query embedding
        # --------------------------------------------------------

        query_embedding = (
            self.embedding_model.encode(
                [query],
                show_progress_bar=False
            )[0]
            .tolist()
        )

        # --------------------------------------------------------
        # Chroma search
        # --------------------------------------------------------

        results = (
            self.collection.query(
                query_embeddings=[
                    query_embedding
                ],
                n_results=top_k,
                include=[
                    "documents",
                    "metadatas",
                    "distances",
                ]
            )
        )

        documents = results.get(
            "documents",
            [[]]
        )

        metadatas = results.get(
            "metadatas",
            [[]]
        )

        distances = results.get(
            "distances",
            [[]]
        )

        if not documents:
            return []

        output = []

        for index, document in enumerate(
            documents[0]
        ):

            metadata = {}

            if (
                metadatas
                and metadatas[0]
                and index < len(
                    metadatas[0]
                )
            ):

                metadata = (
                    metadatas[0][index]
                )

            distance = None

            if (
                distances
                and distances[0]
                and index < len(
                    distances[0]
                )
            ):

                distance = (
                    distances[0][index]
                )

            output.append(
                {
                    "text": document,
                    "metadata": metadata,
                    "distance": distance,
                }
            )

        return output

    # ============================================================
    # CLEAR STORE
    # ============================================================

    def clear(self):

        try:

            self.client.delete_collection(
                self.collection_name
            )

        except Exception as exc:

            print(
                "Warning: Could not delete "
                "Chroma collection:"
            )

            print(exc)

        # Create a new unique collection
        self.collection_name = (
            f"project_source_"
            f"{uuid.uuid4().hex[:12]}"
        )

        self.collection = (
            self.client.create_collection(
                name=self.collection_name
            )
        )

        print(
            f"New Chroma collection created: "
            f"{self.collection_name}"
        )


# ================================================================
# DIRECT TEST
# ================================================================

if __name__ == "__main__":

    print()
    print("=" * 60)
    print("CODE VECTOR STORE TEST")
    print("=" * 60)
    print()

    store = CodeVectorStore(
        persist_directory="output/test_chroma_db",
        model_name="all-MiniLM-L6-v2"
    )

    test_chunks = [

        """
        FILE: lib/main.dart

        SonicSplit is a Flutter application
        for processing audio files.
        """,

        """
        FILE: pubspec.yaml

        The project uses Flutter packages
        for file selection and audio playback.
        """,

        """
        FILE: README.md

        SonicSplit allows users to split audio
        and play individual stems.
        """,

    ]

    store.add_chunks(
        test_chunks
    )

    results = store.search(
        "audio splitting and playback",
        top_k=3
    )

    print()
    print("Search results:")

    for index, result in enumerate(
        results,
        start=1
    ):

        print()
        print(f"[{index}]")

        print(
            result[:500]
        )

    print()
    print(
        "Vector store test completed."
    )