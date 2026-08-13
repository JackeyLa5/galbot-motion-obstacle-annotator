from galbot_motion_obstacle_annotator.planning.models import PlanResult
from galbot_motion_obstacle_annotator.planning.protocol import PlannerMetadata
from galbot_motion_obstacle_annotator.planning.registry import PlannerRegistry


class FakePlanner:
    metadata = PlannerMetadata("fake", "Fake", "Test planner")

    def is_available(self):
        return True, "available"

    def plan(self, request):
        return PlanResult("fake", True, "SUCCESS")


def test_registry_registers_and_resolves_plugins():
    registry = PlannerRegistry()
    planner = FakePlanner()

    registry.register(planner)

    assert registry.get("fake") is planner
    assert registry.all() == (planner,)


def test_registry_rejects_duplicate_plugin_ids():
    registry = PlannerRegistry()
    registry.register(FakePlanner())

    try:
        registry.register(FakePlanner())
    except ValueError as error:
        assert "already registered" in str(error)
    else:
        raise AssertionError("Expected duplicate planner registration to fail")
