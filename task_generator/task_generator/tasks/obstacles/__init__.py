import enum
import uuid

from task_generator.shared import CustomDynamicObstacle, DynamicObstacle, Obstacle, Pose
from task_generator.tasks.mode import TaskMode
from task_generator.tasks.obstacles._placement import random_placement

from . import environment, parametrized, prompt, random, scenario

Obstacles = tuple[list[Obstacle], list[DynamicObstacle]]
CustomObstacles = tuple[list[Obstacle], list[CustomDynamicObstacle]]


@enum.unique
class ObstacleKind(enum.Enum):
    STATIC = "static"
    DYNAMIC = "dynamic"


class TM_Obstacles(TaskMode):
    async def reset(self, **kwargs: object) -> Obstacles:
        return [], []

    async def extend(self, kind: ObstacleKind, model: str, pose: Pose | None = None) -> str:
        resolved_pose = pose if pose is not None else await random_placement(self._ctx)
        name = f"ext_{model}_{uuid.uuid4().hex[:6]}"

        if kind is ObstacleKind.STATIC:
            obstacle: Obstacle | DynamicObstacle = Obstacle(name=name, model=model, pose=resolved_pose)
            await self._ctx.environment_manager.spawn_obstacles([obstacle])
        else:
            waypoints = self._ctx.world_manager.get_positions_on_map(n=2, safe_dist=1)
            obstacle = DynamicObstacle(name=name, model=model, waypoints=waypoints, pose=resolved_pose)
            await self._ctx.environment_manager.spawn_dynamic_obstacles([obstacle])
        return obstacle.sim_path


__all__ = ["TM_Obstacles", "ObstacleKind", "Obstacles", "CustomObstacles", "environment", "parametrized", "prompt", "random", "scenario"]
