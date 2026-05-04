import asyncio
import itertools
import os
import random
import traceback
import types
import typing
from collections.abc import Sequence

import arena_people_msgs.msg
import arena_robots.Robot
import arena_simulation_setup.tree.assets.Material
import isaacsim_msgs.msg
import std_msgs.msg
import std_srvs.srv
from arena_people_msgs.msg import Pedestrian, SpawnPedestrian
from arena_people_msgs.srv import (
    DeletePedestrians,
    MovePedestrians,
    SpawnPedestrians,
    UpdatePedestrians,
)
from arena_rclpy_mixins import ArenaMixinNode
from arena_rclpy_mixins.Async import ClientWrapper
from arena_rclpy_mixins.shared import Namespace
from arena_simulation_setup.shared import Obstacle as ObstacleDefinition
from arena_simulation_setup.tree.Wall import WallSegment
from isaacsim_msgs.msg import (
    Door,
    Elevator,
    Floor,
    Material,
    Pedestrian,
    Skeleton,
    PedestrianGoal,
    Prim,
    Scale,
    Wall,
)
from isaacsim_msgs.srv import (
    DeletePrims,
    EditPrims,
    ResetWorld,
    SpawnDoors,
    SpawnElevators,
    SpawnFloors,
    SpawnPrims,
    SpawnUrdf,
    SpawnUsd,
    SpawnWalls,
)
from std_msgs.msg import String as StdString
from task_generator.shared import Door as DoorDefinition
from task_generator.shared import (
    DynamicObstacle,
    ModelType,
    Obstacle,
    Pose,
    Robot,
)
from task_generator.shared import Elevator as ElevatorDefinition
from task_generator.shared import Floor as FloorDefinition
from task_generator.shared import Wall as WallDefinition

from arena_runtime._node import NodeInterface
from arena_runtime.sim import BaseSim, SimLifecycle

"""
IsaacHost is constructed once by arena_node and owns process-singleton resources for Isaac: the lifecycle (pause/unpause/cleanup) and the pause/unpause/delete service clients.
IsaacSimulator is per-env on task_generator_node and adapts env-namespace state (per-robot publishers, env-prefixed entity names) over those shared resources.
"""


class IsaacHost(SimLifecycle):
    def __init__(self, node: ArenaMixinNode) -> None:
        self._logger = node.get_logger().get_child(type(self).__name__)
        self._pause_client: ClientWrapper = node.create_client_wrapper(
            std_srvs.srv.Trigger,
            "/isaac/PauseSimulation",
        )
        self._unpause_client: ClientWrapper = node.create_client_wrapper(
            std_srvs.srv.Trigger,
            "/isaac/UnpauseSimulation",
        )
        self._delete_prims_client: ClientWrapper = node.create_client_wrapper(
            DeletePrims,
            "/isaac/DeletePrims",
        )

    async def ensure_ready(self) -> None:
        await asyncio.gather(
            self._pause_client.ensure(),
            self._unpause_client.ensure(),
            self._delete_prims_client.ensure(),
        )

    async def pause(self) -> bool:
        res = await self._pause_client.call_timeout(std_srvs.srv.Trigger.Request())
        return bool(res) and res.success

    async def unpause(self) -> bool:
        res = await self._unpause_client.call_timeout(std_srvs.srv.Trigger.Request())
        return bool(res) and res.success

    async def cleanup_namespace(self, prefix: str) -> int:
        res = await self._delete_prims_client.call_timeout(DeletePrims.Request(names=[prefix]))
        if res is None or not res.ret:
            return 0
        return 1 if res.ret[0] else 0


def material_to_msg(material: arena_simulation_setup.tree.assets.Material.Material) -> isaacsim_msgs.msg.Material:
    return Material(
        name=material.name,
        path=material.path,
    )


class IsaacSimulator(BaseSim, NodeInterface):
    def __init__(self, *args: object, **kwargs: object) -> None:
        """Initialize IsaacSimulator"""
        super().__init__(*args, **kwargs)

        env_prefix = f"env_{self._env_id}"
        self._NS_PRIM = Namespace(env_prefix)('Obstacles')
        self._NS_PEDESTRIAN = Namespace(env_prefix)('Pedestrians')
        self._NS_ROBOT = Namespace(env_prefix)('Robots')
        self._NS_WALL = Namespace(env_prefix)('Walls')
        self._NS_FLOOR = Namespace(env_prefix)('Floors')
        self._NS_DOOR = Namespace(env_prefix)('Doors')
        self._NS_ELEVATOR = Namespace(env_prefix)('Elevators')

        self.wall_counter = itertools.count()
        self.floor_counter = itertools.count()
        self._clients = types.SimpleNamespace(
            DeletePedestrians=self.node.create_client_wrapper(DeletePedestrians, "/isaac/DeletePedestrians"),
            DeletePrims=self.node.create_client_wrapper(DeletePrims, "/isaac/DeletePrims"),
            EditPrims=self.node.create_client_wrapper(EditPrims, "/isaac/EditPrims"),
            MovePedestrians=self.node.create_client_wrapper(MovePedestrians, "/isaac/MovePedestrians"),
            ResetWorld=self.node.create_client_wrapper(ResetWorld, "/isaac/ResetWorld"),
            UpdatePedestrians=self.node.create_client_wrapper(UpdatePedestrians, "/isaac/UpdatePedestrians"),
            SpawnDoors=self.node.create_client_wrapper(SpawnDoors, "/isaac/SpawnDoors"),
            SpawnFloors=self.node.create_client_wrapper(SpawnFloors, "/isaac/SpawnFloors"),
            SpawnPedestrians=self.node.create_client_wrapper(SpawnPedestrians, "/isaac/SpawnPedestrians"),
            SpawnPrims=self.node.create_client_wrapper(SpawnPrims, "/isaac/SpawnPrims"),
            SpawnUrdf=self.node.create_client_wrapper(SpawnUrdf, "/isaac/SpawnUrdf"),
            SpawnUsd=self.node.create_client_wrapper(SpawnUsd, "/isaac/SpawnUsd"),
            SpawnWalls=self.node.create_client_wrapper(SpawnWalls, "/isaac/SpawnWalls"),
            SpawnElevators=self.node.create_client_wrapper(SpawnElevators, "/isaac/SpawnElevators"),
        )

        # Publisher for external registration messages so IsaacSim's DoorManager
        # can be informed about spawned entities in the IsaacSim process.
        self._reg_pub = self.node.create_publisher(StdString, '/isaac/register_entity', 10)

    async def robot_spawn(self, robots: Sequence[Robot]) -> Sequence[bool]:
        async def impl(robot: Robot) -> bool:
            try:
                model = await (await robot.model.resolve()).model.get(
                    (
                        ModelType.URDF,
                        # ModelType.USD
                    ),
                    loader_args=robot.asdict(),
                )

                if model.type == ModelType.URDF:
                    assert model.path is not None, f"URDF model {model.name} must have a valid file path"
                    robot_params = (await arena_robots.Robot.RobotIdentifier(robot.model.name).resolve()).model_params

                    fq_name = self._NS_ROBOT(robot.name)

                    await self._clients.SpawnUrdf.call_timeout(
                        SpawnUrdf.Request(
                            name=fq_name,
                            urdf_path=str(model.path),
                            robot_model=robot.model.name,
                            localization=True,
                            tf_prefix=robot.frame.raw(),
                            base_frame=robot_params.base_frame,
                            odom_frame=robot_params.odom_frame,
                            pose=robot.pose.to_msg(),
                            cmd_vel_topic=self.node.service_namespace(robot.name, 'cmd_vel'),
                            joint_states_topic=self.node.service_namespace(robot.name, 'joint_states'),
                            odom_topic=self.node.service_namespace(robot.name, 'odom'),
                        )
                    )

                    base_frame = robot_params.base_frame
                    robot_prim_path = os.path.join("/World", fq_name, base_frame)

                    # Publish registration message so DoorManager in IsaacSim process
                    # registers the robot. This avoids cross-process direct calls.
                    try:
                        if self._reg_pub:
                            self._reg_pub.publish(StdString(data=f"robot|{robot_prim_path}"))
                            self._logger.debug(f"Published registration for robot: {robot_prim_path}")
                        else:
                            self._logger.warning('Registration publisher not available; robot not registered with IsaacSim DoorManager')
                    except Exception as e:
                        self._logger.warning(f'Failed to publish robot registration: {e}\n{traceback.format_exc()}')

                    return True

                # TODO
                raise NotImplementedError(f"robot model of type {model.type} can't be spawned by {self.__class__.__name__}")

            except Exception as e:
                self._logger.error(f"{repr(e)}\n{traceback.format_exc()}")
                return False

        return await asyncio.gather(*map(impl, robots))

    async def obstacle_spawn(self, obstacles: Sequence[Obstacle]) -> Sequence[bool]:

        async def impl(obstacle: Obstacle) -> Prim | None:
            try:
                model = await (await obstacle.model.resolve()).model.get([ModelType.USD])
                if model.type is ModelType.UNKNOWN:
                    raise ValueError(f"obstacle model {obstacle.model.name} has no USD representation")
            except Exception:
                self._logger.warning(f"Failed to resolve model for obstacle {obstacle.name}")
                self._logger.debug(traceback.format_exc())
                return None
            assert model.path is not None, f"USD model {model.name} must have a valid file path"
            prim = Prim()
            prim.usd_path = str(model.path)
            prim.name = self._NS_PRIM(obstacle.name)
            prim.pose = obstacle.pose.to_msg()
            if obstacle.scale is not None:
                prim.scale.x = obstacle.scale.x
                prim.scale.y = obstacle.scale.y
                prim.scale.z = obstacle.scale.z
            return prim

        prims = await asyncio.gather(*map(impl, obstacles))

        req = SpawnPrims.Request()
        req.prims = list(filter(None, prims))
        response = await self._clients.SpawnPrims.call_timeout(req)
        if response is None:
            return tuple(False for _ in obstacles)

        response_iter = iter(response.ret)

        return tuple((a is not None) and next(response_iter) for a in prims)

    async def obstacle_move(self, obstacles: Sequence[Obstacle]) -> Sequence[bool]:
        return await self._move_entities([(self._NS_PRIM(o.name), o.pose) for o in obstacles])

    async def pedestrian_move(self, pedestrians: Sequence[DynamicObstacle]) -> Sequence[bool]:
        req = MovePedestrians.Request(
            pedestrians=[
                Pedestrian(
                    name=self._NS_PEDESTRIAN(p.name),
                    pose=p.pose.to_msg(),
                )
                for p in pedestrians
            ]
        )
        res = await self._clients.MovePedestrians.call_timeout(req)
        if res is None:
            return tuple(False for _ in pedestrians)
        return tuple(r == MovePedestrians.Response.SUCCESS for r in res.results)

    async def robot_move(self, robots: Sequence[Robot]) -> Sequence[bool]:
        async def move_robot(robot: Robot) -> bool:
            try:
                return await self._move_entity(self._NS_ROBOT(robot.name), robot.pose)
            except Exception as e:
                self._logger.error(f"Failed to move robot {robot.name}: {e}\n{traceback.format_exc()}")
                return False

        return await asyncio.gather(*map(move_robot, robots))

    async def obstacle_delete(self, obstacles: Sequence[Obstacle]) -> Sequence[bool]:
        return await asyncio.gather(*(self._delete_entity(self._NS_PRIM(o.name)) for o in obstacles))

    async def pedestrian_delete(self, pedestrians: Sequence[DynamicObstacle]) -> Sequence[bool]:
        res = await self._clients.DeletePedestrians.call_timeout(DeletePedestrians.Request(names=[self._NS_PEDESTRIAN(p.name) for p in pedestrians]))
        if res is None:
            return tuple(False for _ in pedestrians)
        return tuple(r == DeletePedestrians.Response.SUCCESS for r in res.results)

    async def robot_delete(self, robots: Sequence[Robot]) -> Sequence[bool]:
        return await asyncio.gather(*(self._delete_entity(self._NS_ROBOT(r.name)) for r in robots))

    async def remove_world(self) -> bool:
        res = await self._clients.ResetWorld.call_timeout(ResetWorld.Request())
        return bool(res) and res.ret

    async def spawn_walls(self, walls: Sequence[WallDefinition]) -> bool:
        # return True
        self._logger.debug("Attempting to spawn walls")

        async def create_segment(segment: WallSegment) -> Wall | None:
            end = segment.end.to_msg()
            end.z += segment.height
            try:
                wall_name = self._realizer.realize(f"wall_{next(self.wall_counter)}")
                return Wall(
                    name=self._NS_WALL(wall_name),
                    start=segment.start.to_msg(),
                    end=end,
                    material=material_to_msg(await segment.material.resolve()),
                    thickness=segment.width,
                )

            except Exception as e:
                self._logger.error(f"Failed to spawn wall: {e}\n{traceback.format_exc()}")
                return None

        async def create_obstacle(obstacle: ObstacleDefinition) -> Prim | None:
            try:
                prim_name = self._realizer.realize(f"obstacle_{next(self.wall_counter)}")
                model = await (await obstacle.model.resolve()).model.get(ModelType.USD)
                if model.type is ModelType.UNKNOWN:
                    return None
                assert model.path is not None, f"USD model {model.name} must have a valid file path"
                prim = Prim()
                prim.usd_path = str(model.path)
                prim.name = self._NS_WALL(prim_name)
                prim.pose = obstacle.pose.to_msg()
                return prim

            except Exception as e:
                self._logger.error(f"Failed to spawn wall obstacle: {e}\n{traceback.format_exc()}")
                return None

        async def create_wall(wall: WallDefinition) -> tuple[typing.Iterator[object], typing.Iterator[object]]:
            segments, obstacles = await wall.assets()
            return map(create_segment, segments), map(create_obstacle, obstacles)

        wall_futures = await asyncio.gather(*map(create_wall, walls))
        segment_futures, obstacle_futures = zip(*wall_futures, strict=False)

        walls_req = SpawnWalls.Request()
        prims_req = SpawnPrims.Request()
        walls_req.walls = list(filter(None, await asyncio.gather(*itertools.chain.from_iterable(segment_futures))))
        prims_req.prims = list(filter(None, await asyncio.gather(*itertools.chain.from_iterable(obstacle_futures))))

        walls_res = await self._clients.SpawnWalls.call_timeout(walls_req)
        prims_res = await self._clients.SpawnPrims.call_timeout(prims_req)
        res = bool(walls_res) and all(walls_res.ret) and bool(prims_res) and all(prims_res.ret)

        self._logger.debug("All walls spawned.")
        return res

    async def spawn_floors(self, floors: Sequence[FloorDefinition]) -> bool:
        self._logger.debug("Attempting to spawn floors")

        async def impl(floor: FloorDefinition) -> Floor | None:
            try:
                return Floor(
                    name=self._NS_FLOOR(floor.name),
                    x_length=floor.x_length,
                    y_length=floor.y_length,
                    pos=floor.pos.to_msg(),
                    material=material_to_msg(await floor.material.resolve()),
                )

            except Exception:
                self._logger.error(f"Failed to spawn floor: {floor.name}\n{traceback.format_exc()}")
                return None

        floors_req = SpawnFloors.Request()
        floors_req.floors = list(filter(None, await asyncio.gather(*map(impl, floors))))
        floors_res = await self._clients.SpawnFloors.call_timeout(floors_req)

        res = bool(floors_res) and all(floors_res.ret)
        self._logger.debug("All floors spawned successfully.")
        return res

    async def spawn_doors(self, doors: Sequence[DoorDefinition]) -> bool:
        async def impl(door: DoorDefinition) -> Door | None:
            try:
                end = door.end.to_msg()
                end.z += door.height
                return Door(
                    name=self._NS_DOOR(door.name),
                    start=door.start.to_msg(),
                    end=end,
                    material=material_to_msg(await door.material.resolve()),
                    thickness=0.1,
                    kind=door.kind,
                )
            except Exception as e:
                self._logger.error(f"Failed to spawn door: {e}\n{traceback.format_exc()}")
                return None

        doors_req = SpawnDoors.Request()
        doors_req.doors = list(filter(None, await asyncio.gather(*map(impl, doors))))
        doors_res = await self._clients.SpawnDoors.call_timeout(doors_req)

        res = bool(doors_res) and all(doors_res.ret)
        self._logger.debug("All doors spawned successfully.")
        return res

    async def spawn_elevators(self, elevators: Sequence[ElevatorDefinition]) -> bool:
        self._logger.debug(f"IsaacSimulator.spawn_elevators ENTRY, elevators: {elevators}")
        self._logger.debug(f"IsaacSimulator.spawn_elevators called with: {[e.name for e in elevators]}")
        for e in elevators:
            self._logger.debug(f"Elevator data: {e}")

        req = SpawnElevators.Request()

        async def impl(elevator: ElevatorDefinition) -> Elevator | None:
            try:
                pos = elevator.position
                size = elevator.size
                size = Scale(x=size[0], y=size[1], z=size[2])
                des = elevator.destination
                material_resolved = await elevator.material.resolve()
                result = Elevator(
                    name=self._NS_ELEVATOR(elevator.name),
                    position=pos.to_msg(),
                    size=size,
                    height_min=elevator.height_min,
                    height_max=elevator.height_max,
                    material=material_to_msg(material_resolved),
                    destination=des,
                )
                return result
            except Exception as e:
                self._logger.error(f"Failed to append elevator: {elevator.name}: {e}\n{traceback.format_exc()}")
                return None

        req.elevators = list(filter(None, await asyncio.gather(*map(impl, elevators))))
        elevators_res = await self._clients.SpawnElevators.call_timeout(req)
        res = bool(elevators_res) and all(elevators_res.ret)
        self._logger.debug("All elevators spawned successfully.")
        return res

    async def before_reset_episode(self) -> bool:
        return True

    async def after_reset_episode(self) -> bool:
        return True

    async def step(self, n: int = 1) -> bool:
        async with self.node.unpause_window():
            await asyncio.sleep(0.01 * n)
        return True

    async def pedestrian_spawn(self, pedestrians: Sequence[DynamicObstacle]) -> Sequence[bool]:

        # TODO implement targeted pedestrian models
        available_models: dict[str, str] = {
            # "F_Business_02",
            # "F_Medical_01",
            # "M_Medical_01",
            # "biped_demo",
            # "female_adult_police_01_new",
            # "female_adult_police_02",
            # "female_adult_police_03_new",
            # "male_adult_construction_01_new",
            # "male_adult_construction_03",
            # "male_adult_construction_05_new",
            # "male_adult_police_04",
            "female_adult_business_02": "original_female_adult_business_02",
            "female_adult_medical_01": "original_female_adult_medical_01",
            "female_adult_police_01": "original_female_adult_police_01",
            "female_adult_police_02": "original_female_adult_police_02",
            "female_adult_police_03": "original_female_adult_police_03",
            "male_adult_construction_01": "original_male_adult_construction_01",
            "male_adult_construction_02": "original_male_adult_construction_02",
            "male_adult_construction_03": "original_male_adult_construction_03",
            "male_adult_construction_05": "original_male_adult_construction_05",
            "male_adult_medical_01": "original_male_adult_medical_01",
            "male_adult_police_04": "original_male_adult_police_04",
        }

        items = []
        for pedestrian in pedestrians:
            if pedestrian.model.name in available_models:
                model_name = pedestrian.model.name
            else:
                model_name = random.choice(tuple(available_models.keys()))
            items.append(
                SpawnPedestrian(
                    pedestrian=Pedestrian(
                        name=self._NS_PEDESTRIAN(pedestrian.name),
                        pose=pedestrian.pose.to_msg(),
                    ),
                    model_ref=available_models[model_name],
                )
            )

            ped = Pedestrian()
            ped.name = self._NS_PEDESTRIAN(pedestrian.sim_path)
            ped.character_name = available_models[model_name]
            ped.pose = pedestrian.pose.to_msg()
            ped.controller_stats = False

            # Initialize the new skeleton field
            ped.skeleton = Skeleton()
            ped.skeleton.joint_names = [] # Populate if you have data
            ped.skeleton.joint_poses = []
            ped.skeleton.confidences = []

            on_success.append((pedestrian.name, model_name))
            return ped

        req = SpawnPedestrians.Request()
        req.pedestrians = list(filter(None, await asyncio.gather(*map(impl, pedestrians))))
        res = await self._clients.SpawnPedestrians.call_timeout(req)
        if res is None:
            return tuple(False for _ in pedestrians)

        success = tuple(r == SpawnPedestrians.Response.SUCCESS for r in res.results)

        await self.pedestrian_update(
            arena_people_msgs.msg.Pedestrians(
                pedestrians=[
                    arena_people_msgs.msg.Pedestrian(
                        name=ped.name,
                        pose=ped.pose.to_msg(),
                    )
                    for status, ped in zip(success, pedestrians, strict=False)
                    if status
                ]
            )
        )

        return success

    async def pedestrian_update(self, pedestrians: arena_people_msgs.msg.Pedestrians) -> Sequence[bool]:
        req = UpdatePedestrians.Request(
            pedestrians=[
                Pedestrian(
                    name=self._NS_PEDESTRIAN(ped.name),
                    pose=ped.pose,
                    twist=ped.twist,
                )
                for ped in pedestrians.pedestrians
            ]
        )
        res = await self._clients.UpdatePedestrians.call_timeout(req)
        if res is None:
            return tuple(False for _ in pedestrians.pedestrians)
        return tuple(r == UpdatePedestrians.Response.SUCCESS for r in res.results)

    async def _delete_entity(self, name: str) -> bool:
        self._logger.debug(f"Attempting to delete prim {name}")

        res = await self._clients.DeletePrims.call_timeout(DeletePrims.Request(names=[name]))
        if res is None:
            return False

        return res.ret[0]

    async def _move_entity(self, name: str, pose: Pose) -> bool:
        return (await self._move_entities([(name, pose)]))[0]

    async def _move_entities(self, actions: Sequence[tuple[str, Pose]]) -> Sequence[bool]:
        req = EditPrims.Request(
            prims=[
                Prim(
                    name=name,
                    pose=pose.to_msg(),
                )
                for name, pose in actions
            ],
            pose=True,
        )

        response = await self._clients.EditPrims.call_timeout(req)
        if response is None:
            return [False] * len(actions)

        return response.ret

    async def setup(self):
        """
        Initialize all ROS 2 service clients and wait for their availability.
        """
        self._logger.info("Setting up IsaacSimulator service clients...")
        futures: list[typing.Awaitable] = []
        # Define services with their corresponding client attributes
        for client in self._clients.__dict__.values():
            client = typing.cast(ClientWrapper, client)
            self._logger.debug(f"Initializing service client: {client.client.srv_name}")
            futures.append(client.ensure())
        await asyncio.gather(*futures)

        self._logger.info("All service clients are available.")

        self.node.create_publisher(std_msgs.msg.String, '/isaac/add_pedestrians_topic', 10).publish(std_msgs.msg.String(data=self.node.service_namespace('arena_peds')))

        self._logger.info("All service clients initialized and available.")

    @classmethod
    async def create(cls, *args: object, namespace: Namespace, **kwargs: object) -> "IsaacSimulator":
        self = cls(*args, namespace=namespace, **kwargs)
        self._logger.info("Creating IsaacSimulator instance...")
        await self.setup()
        return self
