from __future__ import annotations

import asyncio
import os
import typing

import ament_index_python
import arena_bringup.extensions.NodeLogLevelExtension as NodeLogLevelExtension
import attrs
import geometry_msgs.msg
import launch
import launch.launch_description_sources
import launch_ros
import rclpy
import rclpy.logging
import rclpy.node
import rclpy.publisher
import rclpy.timer
import tf2_ros
from arena_rclpy_mixins.shared import Namespace
from arena_robots.Robot import RobotView
from arena_runtime._node import NodeInterface

import task_generator.utils.arena as Utils
from task_generator.constants import Constants
from task_generator.manager.environment_manager import EnvironmentManager
from task_generator.shared import Orientation, Pose, Position, Robot

if typing.TYPE_CHECKING:
    from task_generator.tasks.robots.adapters import Adapter
    from task_generator.tasks.robots.request import TaskKind, TaskRequest


_NAV2_QUIET_NODES = (
    'behavior_server',
    'bt_navigator',
    'collision_monitor',
    'controller_server',
    'lifecycle_manager_navigation',
    'nav2_container',
    'planner_server',
    'smoother_server',
    'velocity_smoother',
    'waypoint_follower',
)
_NAV2_QUIET_RULES = '+[' + ', '.join(f'**/{n}:error' for n in _NAV2_QUIET_NODES) + ']'


class RobotManager(NodeInterface):
    """Manages the goal and start position of a robot for all task modes."""

    _namespace: Namespace
    _environment_manager: EnvironmentManager
    _start_pos: Pose
    _goal_pos: Pose
    _robot_radius: float
    _goal_tolerance_distance: float
    _goal_tolerance_angle: float
    _robot: Robot
    _move_base_pub: rclpy.publisher.Publisher
    _goal_pub: rclpy.publisher.Publisher
    _pub_goal_timer: rclpy.timer.Timer
    _rate_setup: rclpy.timer.Rate
    _config: RobotView
    # TaskKind -> Adapter dispatch table. V1 holds exactly one entry
    # derived from robot.navigator; multi-capability composition is TODO.
    _adapters: dict[TaskKind, Adapter]
    _current_request: TaskRequest | None
    _phase_index: int

    @property
    def robot(self) -> Robot:
        return self._robot

    @property
    def robot_view(self) -> RobotView:
        return self._config

    @property
    def tf_buffer(self) -> tf2_ros.Buffer:
        return self.node.tf_buffer

    @property
    def start_pos(self) -> Pose:
        return self._start_pos

    @property
    def goal_pos(self) -> Pose:
        return self._goal_pos

    @property
    def pose(self) -> Pose | None:
        """Current robot pose in the map frame (None during reset/respawn windows)."""
        base = self.frame(self._config.model_params.base_frame).raw()
        try:
            t = self.node.tf_buffer.lookup_transform('map', base, rclpy.time.Time())
        except (tf2_ros.LookupException, tf2_ros.ConnectivityException, tf2_ros.ExtrapolationException):
            return None
        tr = t.transform.translation
        return Pose(
            Position(tr.x, tr.y),
            Orientation.from_msg(t.transform.rotation),
        )

    def __init__(
        self,
        *args: object,
        namespace: Namespace,
        environment_manager: EnvironmentManager,
        robot: Robot,
        **kwargs: object,
    ):
        super().__init__(*args, **kwargs)
        self._rate_setup = self.node.create_rate(0.1)

        self._config = robot.model.resolve_sync()

        self._namespace = namespace
        self._environment_manager = environment_manager

        self._start_pos = Pose()
        self._goal_pos = Pose()
        self._robot_radius = 0.25

        self._goal_tolerance_distance = self.node.conf.Robot.GOAL_TOLERANCE_RADIUS.value
        self._goal_tolerance_angle = self.node.conf.Robot.GOAL_TOLERANCE_ANGLE.value
        self._safety_distance = self.node.conf.Robot.SPAWN_ROBOT_SAFE_DIST.value

        self._robot = robot
        self._robot.sim_path = self._environment_manager.realize(robot.name)
        self._robot.extra.setdefault('namespace', self.namespace)
        self._goal_timer = None

        self._publish_goal_task: asyncio.Task | None = None

        # Precedence: per-robot ``navigator`` in robot_setup YAML wins over
        # the CLI / launch-arg default (Robot.parse resolves it). The
        # model_params ``navigator`` is a fallback for the empty-string case.
        # Matching ``capabilities`` entry's extra keys (minus ``kind``) flow
        # as kwargs to the adapter constructor.
        # TODO: multi-capability adapter composition.
        # Deferred to break the import cycle between this module and
        # task_generator.tasks (which eagerly loads context.py → RobotManager).
        from task_generator.tasks.robots.adapters import ADAPTERS

        navigator_kind = self._robot.navigator or self._config.model_params.navigator
        adapter_cls = ADAPTERS.get(navigator_kind)

        adapter_kwargs: dict[str, typing.Any] = {}
        caps_list = self._config.model_params.capabilities
        matching_caps = [entry for entry in caps_list if str(entry.get('kind', '')) == navigator_kind]
        if len(matching_caps) > 1:
            raise AssertionError(f"robot {self._robot.name!r} declares {len(matching_caps)} 'capabilities' entries for kind {navigator_kind!r}; only one is supported (multi-capability adapter composition is still TODO)")
        if len(matching_caps) == 1:
            adapter_kwargs = {k: v for k, v in matching_caps[0].items() if k != 'kind'}
        other_kinds = sorted({str(entry.get('kind', '')) for entry in caps_list if str(entry.get('kind', '')) != navigator_kind})
        if other_kinds:
            self._logger.info(f"robot {self._robot.name!r} has additional 'capabilities' entries ({other_kinds}) not bound to any adapter (TODO: multi-capability adapter composition)")

        # Nav2 reads its planner triplet + train_mode from the Robot runtime
        # config (populated by CLI / benchmark CLI YAML) unless the capabilities
        # entry overrides it.
        if navigator_kind == 'nav2':
            adapter_kwargs.setdefault('global_planner', self._robot.global_planner)
            adapter_kwargs.setdefault('local_planner', self._robot.local_planner)
            adapter_kwargs.setdefault('inter_planner', self._robot.inter_planner)
            adapter_kwargs.setdefault(
                'train_mode',
                self.node.rosparam[bool].get('train_mode', False),
            )

        try:
            adapter = adapter_cls(robot_manager=self, **adapter_kwargs)
        except TypeError as exc:
            raise AssertionError(f"adapter {navigator_kind!r} rejected capability-derived kwargs {sorted(adapter_kwargs)} for robot {self._robot.name!r}: {exc}") from exc

        robot_caps = self._config.caps.available
        missing = adapter.requires - robot_caps
        if missing:
            raise AssertionError(f"robot {self._robot.name!r} (model {self._robot.model.name!r}) does not honor actuator caps required by navigator {navigator_kind!r}: missing {sorted(missing)}; robot declares {sorted(robot_caps)}")

        self._adapters = {kind: adapter for kind in adapter.accepts}
        if not self._adapters:
            raise AssertionError(f"adapter {navigator_kind!r} declares no ``accepts`` set; cannot bind to robot {self._robot.name!r}")
        # Back-compat alias for the single-adapter path.
        self._adapter = adapter

        self._current_request = None
        self._phase_index = 0

    async def set_up_robot(self, node_names: set[str]):

        self._robot.pose.position.z += self._config.model_params.z_offset
        self._robot = (await self._environment_manager.spawn_robot((self._robot,)))[0]

        _gen_goal_topic = self.namespace("goal_pose")

        self._goal_pub = self.node.create_publisher(
            geometry_msgs.msg.PoseStamped,
            _gen_goal_topic,
            10,
        )

        await self._launch_robot(node_names)

        self._robot_radius = self.node.rosparam[float].get(
            'robot_radius',
            self._robot_radius,
        )

    @property
    def radius(self) -> float:
        return self._robot_radius

    @property
    def safe_distance(self) -> float:
        return self._robot_radius + self._safety_distance

    @property
    def model_name(self) -> str:
        return self._robot.model.name

    @property
    def name(self) -> str:
        return self._robot.name

    @property
    def frame(self) -> Namespace:
        return self._robot.frame

    @property
    def accepts(self) -> frozenset[TaskKind]:
        """Task kinds this robot's bound adapters can dispatch."""
        return frozenset(self._adapters.keys())

    @property
    def namespace(self) -> Namespace:
        if Utils.get_arena_type() == Constants.ArenaType.TRAINING:
            return Namespace(f"{self._namespace}{self._namespace}_{self.model_name}")

        return self._namespace(self._robot.sim_path)

    @property
    async def is_done(self) -> bool:
        """Phase-aware three-tier completion check."""
        request = self._current_request
        if request is None or not request.phases:
            return True

        if self._phase_index >= len(request.phases):
            return True

        phase = request.phases[self._phase_index]

        result: bool | None = None
        if request.done_predicate is not None:
            result = request.done_predicate(self, phase)

        if result is None:
            adapter = self._adapters.get(phase.kind)
            if adapter is not None:
                result = adapter.is_phase_done(phase, self)

        if result is None:
            result = phase.is_satisfied(self)

        if not result:
            return False

        self._phase_index += 1
        if self._phase_index >= len(request.phases):
            return True

        next_phase = request.phases[self._phase_index]
        next_adapter = self._adapters[next_phase.kind]
        await next_adapter.dispatch_phase(next_phase, self)
        return False

    async def submit_task(self, request: TaskRequest) -> None:
        """Validate and dispatch phase 0 of a typed TaskRequest. Phase poses are abstract, realized to map here."""
        from task_generator.tasks.robots.request import GoToPhase

        if not request.phases:
            raise ValueError(f"TaskRequest has no phases; nothing to dispatch (robot={self.name!r})")
        for i, phase in enumerate(request.phases):
            if phase.kind not in self._adapters:
                raise AssertionError(f"robot {self.name!r} cannot dispatch phase[{i}] of kind {phase.kind!r}; accepts {sorted(k.name for k in self.accepts)}")

        realized_phases = [attrs.evolve(phase, pose=self._environment_manager.realize(phase.pose)) if isinstance(phase, GoToPhase) else phase for phase in request.phases]
        request = attrs.evolve(request, phases=realized_phases)

        self._current_request = request
        self._phase_index = 0

        phase0 = request.phases[0]
        adapter = self._adapters[phase0.kind]
        await adapter.dispatch_phase(phase0, self)

        # (Re)start the keep-alive loop that republishes _goal_pos against
        # nav2/AMCL jitter and (crucially) covers the gap when nav2 isn't
        # subscribed yet on the first dispatch. Adapters that own their own
        # goal transport (e.g. action client) opt out via republishes_goal=False
        # so the loop does not race their dispatch.
        if adapter.republishes_goal:
            if self._publish_goal_task is not None:
                self._publish_goal_task.cancel()
            self._publish_goal_task = asyncio.create_task(self._publish_goal_loop())
        elif self._publish_goal_task is not None:
            self._publish_goal_task.cancel()
            self._publish_goal_task = None

        if self._robot.record_data_dir:
            self.node.rosparam[list[float]].set(self.namespace.robot_ns.ParamNamespace()("goal"), [self.goal_pos.position.x, self.goal_pos.position.y, self.goal_pos.orientation.to_yaw()])

    async def _apply_pose(self, pose: Pose):
        pose.position.z += self._config.model_params.z_offset
        self.robot.pose = pose
        await self._environment_manager.move_robot((self.robot,))
        import time

        time.sleep(0.001)  # wait for the robot to move
        await self._adapter.on_move(pose, self)

    async def move(self, pose: Pose) -> None:
        """Teleport the robot to ``pose``. Positioning only — no task dispatch."""
        self._start_pos = pose
        await self._apply_pose(pose)

        if self._robot.record_data_dir:
            realized = self._environment_manager.realize(self._start_pos)
            self.node.rosparam[list[float]].set(self.namespace.robot_ns.ParamNamespace()("start"), [realized.position.x, realized.position.y, realized.orientation.to_yaw()])

    async def _publish_goal_loop(self):
        # Keeps republishing _goal_pos against amcl jitter until a reset
        # swaps _goal_pos to a different object. is_done elsewhere handles
        # completion; this loop only keeps the goal alive.

        target = self._goal_pos
        with self.node.sim_time_rate(1.0, 60) as (done, rate):
            while not done.is_set():
                await rate.get()

                if self._goal_pos is not target:
                    break

                goal = self._goal_pos
                self._logger.debug(f"Publishing goal: x={goal.position.x}, y={goal.position.y}, orientation={goal.orientation.to_yaw()}")

                if self._goal_timer is not None:
                    self._goal_timer.cancel()
                    self._goal_timer.destroy()

                goal_msg = geometry_msgs.msg.PoseStamped()
                goal_msg.header.frame_id = "map"
                goal_msg.header.stamp = self.node.sim_time.to_msg()
                goal_msg.pose = goal.to_msg()
                self._goal_pub.publish(goal_msg)

                self._goal_start_time = self.node.sim_time

    async def _launch_robot(self, node_paths: set[str]):
        """Launch the robot's navstack via the bound adapter."""
        self._logger.info(f"LAUNCH ROBOT {self.name}")

        if Utils.get_arena_type() != Constants.ArenaType.TRAINING:
            launch_description = launch.LaunchDescription()
            current_log_level = rclpy.logging.get_logger_effective_level(self.node.get_logger().name).name.lower()
            launch_description.add_action(NodeLogLevelExtension.SetGlobalLogLevelAction(current_log_level))
            launch_description.add_action(NodeLogLevelExtension.SetGlobalLogLevelAction(_NAV2_QUIET_RULES))

            # Adapter dispatch happens inside robot.launch.py via the
            # ``navigator`` launch arg; Adapter.launch_description is
            # the attachment point for pulling that up here later.
            launch_arguments = {
                'robot': self.model_name,
                'task_generator_node': os.path.join(self.node.get_namespace(), self.node.get_name()),
                'namespace': self.namespace,
                'frame': self._robot.frame('').raw(),  # trailing slash
                'train_mode': str(self.node.rosparam[bool].get('train_mode', False)).lower(),
                'agent_name': self._robot.agent,
                'use_sim_time': 'True',
                # Read by robot.launch.py to gate the rosnav_rl action server.
                'local_planner': self._robot.local_planner,
            }

            if self._robot.record_data_dir:
                launch_arguments.update(
                    {
                        'record_data_dir': self._robot.record_data_dir,
                    }
                )

            launch_description.add_action(
                launch.actions.IncludeLaunchDescription(
                    launch.launch_description_sources.PythonLaunchDescriptionSource(os.path.join(ament_index_python.packages.get_package_share_directory('arena_simulation_setup'), 'launch/robot.launch.py')),
                    launch_arguments=launch_arguments.items(),
                )
            )

            # Navstack adapter dispatch: each adapter owns its own launch
            # file + the kwargs it needs. PushRosNamespace scopes it under
            # this robot (robot.launch.py has its own separate push).
            from task_generator.tasks.robots.adapters import AdapterCtx

            adapter_ctx = AdapterCtx(
                namespace=self.namespace,
                robot_name=self.model_name,
                frame=self._robot.frame('').raw(),
                task_generator_node=os.path.join(self.node.get_namespace(), self.node.get_name()),
                use_sim_time=True,
                base_frame=self._config.model_params.base_frame,
                odom_frame=self._config.model_params.odom_frame,
                sensors=self._config.model_params.sensors,
                tf_buffer=None,
                node_handle=self.node,
            )
            launch_description.add_action(
                launch.actions.GroupAction(
                    [
                        launch_ros.actions.PushRosNamespace(str(self.namespace)),
                        self._adapter.launch_description(adapter_ctx),
                    ]
                )
            )

            await self.node.do_launch(launch_description)

            await self._adapter.wait_until_ready(self, node_paths)

    async def update(self):
        # TODO implement record data dir
        pass

    async def destroy(self):
        if self._goal_timer is not None:
            self._goal_timer.cancel()
            self._goal_timer.destroy()
        await self._environment_manager.remove_robot((self.robot,))
        # TODO kill node in navigation stack
