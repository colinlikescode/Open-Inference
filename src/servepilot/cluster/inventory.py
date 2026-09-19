"""Existing machine inventory. A controller need not be one of the GPU nodes."""

from __future__ import annotations

import re
import socket
from pathlib import Path

import psutil
import yaml
from pydantic import Field, field_validator, model_validator

from servepilot.exceptions import ConfigurationError
from servepilot.optimization.schemas import Contract


class SSHNode(Contract):
    host: str
    address: str | None = None
    user: str | None = None
    port: int = Field(default=22, ge=1, le=65535)
    identity_file: str | None = None
    local: bool | None = None
    python: str = "python3"

    @field_validator("host", "address")
    @classmethod
    def safe_host(cls, value: str | None) -> str | None:
        if value is not None and not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.:%-]*", value):
            raise ValueError("host/address must be a hostname, SSH alias, or IP address")
        return value

    @field_validator("user")
    @classmethod
    def safe_user(cls, value: str | None) -> str | None:
        if value is not None and not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_.-]*", value):
            raise ValueError("invalid SSH username")
        return value

    @property
    def node_id(self) -> str:
        return self.host

    @property
    def network_address(self) -> str:
        return self.address or self.host

    @property
    def is_local(self) -> bool:
        if self.local is not None:
            return self.local
        addresses = {"localhost", "127.0.0.1", "::1", socket.gethostname(), socket.getfqdn()}
        addresses.update(
            a.address.split("%", 1)[0] for values in psutil.net_if_addrs().values() for a in values
        )
        return self.host in addresses


class NodeInventory(Contract):
    nodes: list[SSHNode] = Field(min_length=1)
    ssh_user: str | None = None
    ssh_identity_file: str | None = None
    ssh_known_hosts: str | None = None
    head: str | None = None
    connect_timeout_seconds: int = Field(default=15, ge=1)

    @model_validator(mode="after")
    def normalize(self) -> NodeInventory:
        ids = [node.node_id for node in self.nodes]
        if len(set(ids)) != len(ids):
            raise ValueError("nodes must have unique hosts")
        if self.head is not None and self.head not in ids:
            raise ValueError(
                "head must name one of the GPU nodes; the controller can run elsewhere"
            )
        for node in self.nodes:
            if node.user is None:
                node.user = SSHNode.safe_user(self.ssh_user)
            if node.identity_file is None:
                node.identity_file = self.ssh_identity_file
        if self.head:
            self.nodes.sort(key=lambda node: node.host != self.head)
        return self

    def node(self, node_id: str | None) -> SSHNode:
        if node_id is None or node_id == "local":
            local = [node for node in self.nodes if node.is_local]
            if len(local) == 1:
                return local[0]
        for node in self.nodes:
            if node.node_id == node_id:
                return node
        raise ConfigurationError(f"node {node_id!r} is absent from the inventory")

    @classmethod
    def load(cls, path: Path | None) -> NodeInventory:
        if path is None:
            return cls(nodes=[SSHNode(host="localhost", local=True)])
        try:
            inventory = cls.model_validate(yaml.safe_load(path.read_text()))
            # Identity and known_hosts paths refer to the file's directory, not the shell cwd.
            for node in inventory.nodes:
                if node.identity_file:
                    identity = Path(node.identity_file).expanduser()
                    node.identity_file = str((path.parent / identity).resolve())
            if inventory.ssh_known_hosts:
                inventory.ssh_known_hosts = str(
                    (path.parent / Path(inventory.ssh_known_hosts).expanduser()).resolve()
                )
            return inventory
        except (OSError, ValueError, yaml.YAMLError) as exc:
            raise ConfigurationError(f"invalid node inventory {path}: {exc}") from exc
