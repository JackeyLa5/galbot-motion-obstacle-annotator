from __future__ import annotations

from importlib.metadata import entry_points

from .protocol import MotionPlannerPlugin

ENTRY_POINT_GROUP = "galbot_motion_obstacle_annotator.planners"


class PlannerRegistry:
    def __init__(self) -> None:
        self._planners: dict[str, MotionPlannerPlugin] = {}

    def register(self, planner: MotionPlannerPlugin) -> None:
        planner_id = planner.metadata.planner_id
        if not planner_id:
            raise ValueError("Planner plugin ID must be non-empty")
        if planner_id in self._planners:
            raise ValueError(f"Planner plugin already registered: {planner_id}")
        self._planners[planner_id] = planner

    def get(self, planner_id: str) -> MotionPlannerPlugin:
        try:
            return self._planners[planner_id]
        except KeyError as error:
            raise KeyError(f"Unknown planner plugin: {planner_id}") from error

    def all(self) -> tuple[MotionPlannerPlugin, ...]:
        return tuple(self._planners.values())

    def discover(self) -> None:
        available = entry_points()
        if hasattr(available, "select"):
            discovered = available.select(group=ENTRY_POINT_GROUP)
        else:
            discovered = available.get(ENTRY_POINT_GROUP, ())
        for entry_point in discovered:
            plugin = entry_point.load()()
            self.register(plugin)


def default_registry() -> PlannerRegistry:
    from .galbot_motion import GalbotMotionPlanner
    from .pyroki import PyrokiPlanner

    registry = PlannerRegistry()
    registry.register(GalbotMotionPlanner())
    registry.register(PyrokiPlanner())
    registry.discover()
    return registry
