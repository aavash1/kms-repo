# src/core/services/knowledge_graph.py
import networkx as nx
import logging
from typing import List, Dict, Any
from bs4 import BeautifulSoup

logger = logging.getLogger(__name__)

class KnowledgeGraph:
    def __init__(self):
        self.graph = nx.DiGraph()
        self._initialize_static_nodes()

    def _initialize_static_nodes(self):
        """
        Initialize static ErrorCode nodes using the static data cache.
        """
        from src.core.services.static_data_cache import static_data_cache
        for error_code_id, info in static_data_cache.error_code_data.items():
            self.graph.add_node(
                f"ErrorCode_{error_code_id}",
                type="ErrorCode",
                error_code_id=error_code_id,
                error_code_nm=info["error_code_nm"],
                explanation_en=info["explanation_en"],
                message_en=info["message_en"],
                recom_action_en=info["recom_action_en"]
            )
        logger.info(f"Initialized {self.graph.number_of_nodes()} ErrorCode nodes in knowledge graph.")

    def add_resolve(self, resolve_data: dict, file_urls: List[str]):
        """
        Add a Resolve node and its associated AttachmentFile nodes to the knowledge graph.

        Args:
            resolve_data (dict): Parsed resolve_data containing errorCodeId, clientNm, osVersionId, content, resolveId.
            file_urls (List[str]): List of S3 URLs for attachment files.
        """
        error_code_id = str(resolve_data.get("errorCodeId", ""))
        resolve_id = str(resolve_data.get("resolveId", ""))
        client_name = resolve_data.get("clientNm", "")
        os_version_id = str(resolve_data.get("osVersionId", "11"))
        content = resolve_data.get("content", "")

        # Add Resolve node
        resolve_node = f"Resolve_{resolve_id}"
        self.graph.add_node(
            resolve_node,
            type="Resolve",
            resolve_id=resolve_id,
            client_name=client_name,
            os_version_id=os_version_id,
            content=content
        )

        # Connect Resolve to ErrorCode
        error_code_node = f"ErrorCode_{error_code_id}"
        if error_code_node in self.graph:
            self.graph.add_edge(error_code_node, resolve_node, relationship="HAS_RESOLVE")

        # Add AttachmentFile nodes
        for url in file_urls:
            logical_nm = url.split("/")[-1]
            attachment_node = f"AttachmentFile_{logical_nm}"
            self.graph.add_node(
                attachment_node,
                type="AttachmentFile",
                logical_nm=logical_nm,
                url=url
            )
            self.graph.add_edge(resolve_node, attachment_node, relationship="HAS_ATTACHMENT")

        # Extract embedded URLs from content and link them
        soup = BeautifulSoup(content, "html.parser")
        for img in soup.find_all("img"):
            src = img.get("src", "")
            if src and "netbackup-kms-prod.s3" in src:
                logical_nm = src.split("/")[-1]
                attachment_node = f"AttachmentFile_{logical_nm}"
                if attachment_node not in self.graph:
                    self.graph.add_node(
                        attachment_node,
                        type="AttachmentFile",
                        logical_nm=logical_nm,
                        url=src
                    )
                self.graph.add_edge(resolve_node, attachment_node, relationship="HAS_EMBEDDED_ATTACHMENT")

        logger.debug(f"Added Resolve_{resolve_id} with {len(file_urls)} attachments to knowledge graph.")

    def query_related_resolves(self, error_code_id: str) -> Dict[str, Any]:
        """
        Query all Resolve nodes related to a given error_code_id.

        Args:
            error_code_id (str): The error code ID to query.

        Returns:
            Dict: Dictionary containing error code info and related Resolve nodes.
        """
        error_code_node = f"ErrorCode_{error_code_id}"
        if error_code_node not in self.graph:
            return {"error_code_id": error_code_id, "error_code_nm": None, "resolves": []}

        error_code_data = self.graph.nodes[error_code_node]
        resolves = []
        for resolve_node in self.graph.successors(error_code_node):
            resolve_data = self.graph.nodes[resolve_node]
            attachments = []
            for attachment_node in self.graph.successors(resolve_node):
                attachment_data = self.graph.nodes[attachment_node]
                attachments.append({
                    "logical_nm": attachment_data["logical_nm"],
                    "url": attachment_data["url"],
                    "type": self.graph.edges[resolve_node, attachment_node]["relationship"]
                })
            resolves.append({
                "resolve_id": resolve_data["resolve_id"],
                "client_name": resolve_data["client_name"],
                "os_version_id": resolve_data["os_version_id"],
                "content": resolve_data["content"],
                "attachments": attachments
            })
        return {
            "error_code_id": error_code_id,
            "error_code_nm": error_code_data["error_code_nm"],
            "explanation_en": error_code_data["explanation_en"],
            "message_en": error_code_data["message_en"],
            "recom_action_en": error_code_data["recom_action_en"],
            "resolves": resolves
        }

# Singleton instance
knowledge_graph = KnowledgeGraph()