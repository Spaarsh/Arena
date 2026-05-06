import io
import os
import tarfile
import typing
from collections.abc import Iterator
from pathlib import Path
from typing import Self

import attrs
import yaml

from arena_simulation_setup import ASS_DIR
from arena_simulation_setup.shared import (
    Door,
    DynamicObstacle,
    Elevator,
    Floor,
    Obstacle,
    Wall,
)
from arena_simulation_setup.tree import FallbackResolver, Identifier, PathView
from arena_simulation_setup.tree.assets.Material import (
    Material,
    MaterialIdentifier,
)
from arena_simulation_setup.utils.cattrs import converter
from arena_simulation_setup.utils.geometry import Position

from .Map import Map
from .Scenario import ScenarioView


@attrs.define
class WorldDescription:
    """
    Description of the 3D world
    """

    @attrs.define
    class Zone:
        """
        Description of a zone (e.g. room) within the 3D world
        """

        @attrs.define
        class WorldEntities:
            """
            Description of the entities within the 3D world
            """

            static: list[Obstacle] = attrs.field(factory=list)
            dynamic: list[DynamicObstacle] = attrs.field(factory=list)

        name: str
        description: str = ''
        material: MaterialIdentifier = attrs.field(
            converter=MaterialIdentifier.converter,
            default=Material.default('floor'),
        )
        corners: list[Position] = attrs.field(factory=list)
        walls: list[Wall] = attrs.field(factory=list)
        doors: list[Door] = attrs.field(factory=list)
        elevators: list[Elevator] = attrs.field(factory=list)
        entities: WorldEntities = attrs.field(factory=WorldEntities)

        @property
        def floor(self) -> Floor:
            x_min = min(corner.x for corner in self.corners)
            y_min = min(corner.y for corner in self.corners)
            x_max = max(corner.x for corner in self.corners)
            y_max = max(corner.y for corner in self.corners)
            pos = Position(x=(x_min + x_max) / 2, y=(y_min + y_max) / 2)
            x_length = x_max - x_min
            y_length = y_max - y_min
            return Floor(name=self.name, pos=pos, x_length=x_length, y_length=y_length, material=self.material)

    zones: list[Zone] = attrs.field(factory=list)

    @property
    def all_walls(self) -> typing.Iterable[Wall]:
        return (wall for zone in self.zones for wall in zone.walls)

    @property
    def all_doors(self) -> typing.Iterable[Door]:
        return (door for zone in self.zones for door in zone.doors)

    @property
    def all_elevators(self) -> typing.Iterable[Elevator]:
        return (elevator for zone in self.zones for elevator in zone.elevators)

    @property
    def all_floors(self) -> typing.Iterable[Floor]:
        return (zone.floor for zone in self.zones)

    @property
    def all_static_entities(self) -> typing.Iterable[Obstacle]:
        return (entity for zone in self.zones for entity in zone.entities.static)

    @property
    def all_dynamic_entities(self) -> typing.Iterable[DynamicObstacle]:
        return (entity for zone in self.zones for entity in zone.entities.dynamic)

    def render(
        self,
        resolution: float = 0.05,
        *,
        default_asset_bbox: tuple[tuple[float, float], tuple[float, float]] | None = None,
        asset_color: str | None = None,
        asset_name_color: str | None = None,
    ) -> tuple[bytes, tuple[float, float]]:
        """
        Render the world description to a PNG image.

        Args:
            resolution (float): The resolution of the rendered image in meters per pixel.
            default_asset_bbox (Optional[tuple[tuple[float, float], tuple[float, float]]]): Default bounding box ((xmin, xmax), (ymin, ymax)) to use for static entities if not specified individually.
            asset_color (Optional[str]): Color used to fill static objects in the map.
            asset_name_color (Optional[str]): Color used for static object names in the map.

        Returns:
            Tuple[bytes, tuple[float, float]]: PNG image bytes and the origin (x, y) of the map.

        Notes:
            - Static objects are drawn only if their dimensions can be determined from bbox, width/height, or default_asset_bbox.
            - If asset_color is None, static objects are not drawn.
        """
        import shapely
        import shapely.affinity

        map_kwargs: dict[str, typing.Any] = {}

        if asset_color is not None:
            static_objects: list[tuple[str, shapely.Polygon]] = []
            for entity in self.all_static_entities:
                try:
                    bbox = entity.asdict(expand_extra=True).get('bbox')
                    if bbox is None:
                        if default_asset_bbox is None:
                            raise ValueError(f"Static entity '{entity.name}' does not have a bbox and no default_asset_bbox was provided.")
                        bbox = default_asset_bbox
                    (x_min, x_max), (y_min, y_max), *_ = bbox
                except Exception:
                    continue
                poly = shapely.box(x_min, y_min, x_max, y_max)
                poly = shapely.affinity.rotate(poly, entity.pose.orientation.to_yaw(), use_radians=True)
                poly = shapely.affinity.translate(poly, entity.pose.position.x, entity.pose.position.y)
                static_objects.append((entity.name, poly))

            map_kwargs["static_objects"] = static_objects
            map_kwargs["asset_color"] = asset_color
            map_kwargs["asset_name_color"] = asset_name_color

        png, origin = Map.generate_png(
            rooms=shapely.MultiPolygon([shapely.Polygon(zone.corners) for zone in self.zones]),
            doors=shapely.MultiPolygon([shapely.Polygon(door.corners) for door in self.all_doors]),
            walls=shapely.MultiLineString(list(self.all_walls)),
            resolution=resolution,
            padding=5,
            **map_kwargs,
        )
        return png, origin

    def export(self, resolution: float = 0.05, extra_files: dict[str, bytes] | None = None, **kwargs: object) -> tarfile.TarFile:
        """
        Export the world description to world.yaml, map.png, map.yaml
        """

        if extra_files is None:
            extra_files = {}
        files: dict[str, bytes] = {**extra_files}

        files['world.yaml'] = typing.cast(bytes, yaml.safe_dump(converter.unstructure(self), encoding='utf-8', sort_keys=False))

        render_args: dict[str, typing.Any] = {"resolution": resolution}
        if "default_asset_bbox" in kwargs:
            render_args["default_asset_bbox"] = kwargs["default_asset_bbox"]
        if "asset_color" in kwargs:
            render_args["asset_color"] = kwargs["asset_color"]
        if "asset_name_color" in kwargs:
            render_args["asset_name_color"] = kwargs["asset_name_color"]

        files['map/map.png'], origin = self.render(**render_args)

        files['map/map.yaml'] = Map.generate_map_yaml(resolution=resolution, filename='map.png', origin=origin).encode('utf-8')

        with io.BytesIO() as tar_stream:
            with tarfile.open(mode='w', fileobj=tar_stream) as tarball:
                for filename, content in files.items():
                    info = tarfile.TarInfo(name=os.path.normpath(filename))
                    info.size = len(content)
                    tarball.addfile(tarinfo=info, fileobj=io.BytesIO(content))
            tar_stream.seek(0)
            return tarfile.open(fileobj=io.BytesIO(tar_stream.getvalue()))


class World(PathView):
    @property
    def scenario(self) -> type[Identifier[ScenarioView]]:
        class ScenarioIdentifier(Identifier[ScenarioView]):
            @classmethod
            def listall(cls, **kwargs: object) -> Iterator[Self]:
                scenarios_dir = self.path / 'scenarios'
                if not scenarios_dir.is_dir():
                    yield from ()
                    return
                yield from (cls(entry.name) for entry in os.scandir(scenarios_dir) if entry.is_dir())

            def load(self, path: Path, /, **kwargs: object) -> ScenarioView:
                del kwargs
                return ScenarioView(path)

        ScenarioIdentifier.use(FallbackResolver(ScenarioIdentifier, self.path / 'scenarios'))
        return ScenarioIdentifier

    @property
    def map(self) -> Map:
        return Map(self.path / 'map')

    @property
    def world_path(self) -> Path:
        return self.path / 'world.yaml'

    def load(self) -> WorldDescription:
        with open(self.world_path) as f:
            return converter.structure(yaml.safe_load(f), WorldDescription)

    def save(self, world: WorldDescription, map_only: bool = False, **kwargs: object) -> Path:
        os.makedirs(self.path, exist_ok=True)
        tarball = world.export(**kwargs)

        _filter: typing.Callable[[tarfile.TarInfo, str], tarfile.TarInfo | None] = tarfile.fully_trusted_filter
        if map_only:

            def map_only_filter(member: tarfile.TarInfo, destpath: str) -> tarfile.TarInfo | None:
                if not member.name.startswith('map/'):
                    return None
                return tarfile.fully_trusted_filter(member, destpath)

            _filter = map_only_filter

        tarball.extractall(self.path, filter=_filter)
        return self.path


class WorldIdentifier(Identifier[World]):
    @classmethod
    def listall(cls, **kwargs: object) -> Iterator[Self]:
        yield from (WorldIdentifier(name) for name in os.listdir(ASS_DIR / 'worlds') if name.lower() != 'readme.md')

    def load(self, path: Path, /, **kwargs: object) -> World:
        del kwargs
        return World(path)


WorldIdentifier.use(FallbackResolver(WorldIdentifier, ASS_DIR / 'worlds'))
