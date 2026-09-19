"""Engine discovery and lookup."""

from __future__ import annotations

from collections.abc import Iterable

from servepilot.engines.base import InferenceEngine
from servepilot.exceptions import EngineUnavailableError
from servepilot.schemas.plan import EngineName
from servepilot.settings import ServePilotSettings


class EngineRegistry:
    """Holds engine adapter instances keyed by :class:`EngineName`."""

    def __init__(self, engines: Iterable[InferenceEngine]) -> None:
        self._engines: dict[EngineName, InferenceEngine] = {e.engine_name: e for e in engines}

    def all(self) -> list[InferenceEngine]:
        return list(self._engines.values())

    def available(self) -> list[InferenceEngine]:
        return [e for e in self._engines.values() if e.is_available()]

    def get(self, name: EngineName | str) -> InferenceEngine:
        try:
            key = EngineName(name)
        except ValueError as exc:
            raise EngineUnavailableError(
                f"unknown engine {name!r}; choose from {[e.value for e in EngineName if e != EngineName.FAKE]}"
            ) from exc
        try:
            return self._engines[key]
        except KeyError as exc:
            raise EngineUnavailableError(f"engine {key.value!r} is not registered") from exc

    def require(self, name: EngineName | str) -> InferenceEngine:
        engine = self.get(name)
        if not engine.is_available():
            raise EngineUnavailableError(
                f"{engine.name()} is not installed or could not be located.",
                hints=engine.unavailable_hints(),
            )
        return engine

    def versions(self) -> dict[str, str]:
        return {e.name(): v for e in self.available() if (v := e.version())}


def build_registry(settings: ServePilotSettings | None = None) -> EngineRegistry:
    """Instantiate all adapters (probing installations); fake engine only when enabled."""
    from servepilot.engines.sglang import SGLangEngine
    from servepilot.engines.vllm import VLLMEngine

    settings = settings or ServePilotSettings()
    engines: list[InferenceEngine] = [VLLMEngine(), SGLangEngine()]
    if settings.enable_fake_engine:
        from servepilot.testing.fake_engine import FakeEngine, FakeEngineBehavior

        engines.append(FakeEngine(FakeEngineBehavior.load(settings.fake_engine_behavior)))
    return EngineRegistry(engines)


def discover_engines(settings: ServePilotSettings | None = None) -> list[InferenceEngine]:
    return build_registry(settings).available()


def get_engine(
    name: EngineName | str, settings: ServePilotSettings | None = None
) -> InferenceEngine:
    return build_registry(settings).require(name)


def select_engines(registry: EngineRegistry, requested: str) -> list[InferenceEngine]:
    """Resolve ``--engine auto|vllm|sglang|fake`` into adapter instances (all must be available)."""
    if requested == "auto":
        available = [e for e in registry.available() if e.engine_name != EngineName.FAKE]
        fake = [e for e in registry.available() if e.engine_name == EngineName.FAKE]
        if not available and fake:
            return fake
        if not available:
            raise EngineUnavailableError(
                "No inference engine is installed.",
                hints=[
                    'pip install vllm   or   pip install "sglang[all]"',
                    "or point ServePilot at an existing environment: SERVEPILOT_VLLM_PYTHON=/path/to/python",
                    "Run `servepilot doctor` to see what ServePilot detects.",
                ],
            )
        return available
    return [registry.require(requested)]
