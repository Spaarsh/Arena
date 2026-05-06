from __future__ import annotations

import pytest

from arena_simulation_setup.shared.world import Door
from arena_simulation_setup.shared.walls import Wall
from arena_simulation_setup.tree.World.World import WorldDescription
from arena_simulation_setup.utils.geometry import Position


def _make_zone(name: str = "zone", corners=None, walls=None, doors=None, elevators=None) -> WorldDescription.Zone:
    return WorldDescription.Zone(
        name=name,
        corners=corners or [],
        walls=walls or [],
        doors=doors or [],
        elevators=elevators or [],
    )


def test_zone_floor_rectangle():
    zone = _make_zone(
        name="room",
        corners=[
            Position(0.0, 0.0), Position(4.0, 0.0),
            Position(4.0, 3.0), Position(0.0, 3.0),
        ],
    )
    floor = zone.floor
    assert floor.x_length == pytest.approx(4.0)
    assert floor.y_length == pytest.approx(3.0)
    assert floor.pos.x == pytest.approx(2.0)
    assert floor.pos.y == pytest.approx(1.5)


def test_zone_floor_triangular():
    zone = _make_zone(
        name="tri",
        corners=[
            Position(0.0, 0.0), Position(2.0, 0.0), Position(1.0, 2.0),
        ],
    )
    floor = zone.floor
    assert floor.x_length == pytest.approx(2.0)
    assert floor.y_length == pytest.approx(2.0)


def test_zone_floor_empty_corners_raises():
    zone = _make_zone(name="empty")
    with pytest.raises((ValueError, Exception)):
        _ = zone.floor


def test_all_walls_empty():
    wd = WorldDescription(zones=[])
    assert list(wd.all_walls) == []


def test_all_walls_populated():
    w = Wall(start=Position(0, 0), end=Position(1, 0))
    zone = _make_zone(walls=[w])
    wd = WorldDescription(zones=[zone])
    assert list(wd.all_walls) == [w]


def test_all_walls_multiple_zones():
    w1 = Wall(start=Position(0, 0), end=Position(1, 0))
    w2 = Wall(start=Position(2, 0), end=Position(3, 0))
    z1 = _make_zone("z1", walls=[w1])
    z2 = _make_zone("z2", walls=[w2])
    wd = WorldDescription(zones=[z1, z2])
    assert len(list(wd.all_walls)) == 2


def test_all_doors_empty():
    wd = WorldDescription(zones=[])
    assert list(wd.all_doors) == []


def test_all_doors_populated():
    d = Door(name="d", start=Position(0, 0), end=Position(1, 0))
    zone = _make_zone(doors=[d])
    wd = WorldDescription(zones=[zone])
    assert list(wd.all_doors) == [d]


def test_all_elevators_empty():
    wd = WorldDescription(zones=[])
    assert list(wd.all_elevators) == []


def test_all_floors_count():
    z1 = _make_zone("z1", corners=[Position(0, 0), Position(2, 0), Position(2, 2), Position(0, 2)])
    z2 = _make_zone("z2", corners=[Position(0, 0), Position(4, 0), Position(4, 4), Position(0, 4)])
    wd = WorldDescription(zones=[z1, z2])
    assert len(list(wd.all_floors)) == 2


def test_all_static_entities_empty():
    wd = WorldDescription(zones=[_make_zone()])
    assert list(wd.all_static_entities) == []


def test_all_dynamic_entities_empty():
    wd = WorldDescription(zones=[_make_zone()])
    assert list(wd.all_dynamic_entities) == []


def _make_square_zone(name: str = "room", size: float = 10.0) -> WorldDescription.Zone:
    return _make_zone(
        name=name,
        corners=[
            Position(0.0, 0.0), Position(size, 0.0),
            Position(size, size), Position(0.0, size),
        ],
        walls=[
            Wall(start=Position(0.0, 0.0), end=Position(size, 0.0)),
        ],
    )


def test_render_returns_bytes_and_origin_tuple():
    wd = WorldDescription(zones=[_make_square_zone()])
    result = wd.render()
    assert isinstance(result, tuple)
    assert len(result) == 2


def test_render_bytes_is_valid_png():
    import io
    import PIL.Image
    wd = WorldDescription(zones=[_make_square_zone()])
    png_bytes, _ = wd.render()
    assert isinstance(png_bytes, bytes)
    img = PIL.Image.open(io.BytesIO(png_bytes))
    assert img.format == 'PNG'


def test_render_image_mode_is_rgb():
    import io
    import PIL.Image
    wd = WorldDescription(zones=[_make_square_zone()])
    png_bytes, _ = wd.render()
    img = PIL.Image.open(io.BytesIO(png_bytes))
    assert img.mode == 'RGB'


def test_render_image_size_reflects_zone_aabb():
    import io
    import math
    import PIL.Image
    size = 10.0
    resolution = 0.05
    padding = 5
    wd = WorldDescription(zones=[_make_square_zone(size=size)])
    png_bytes, _ = wd.render(resolution=resolution)
    img = PIL.Image.open(io.BytesIO(png_bytes))
    expected_px = math.ceil(size / resolution) + 2 * padding
    assert img.width == expected_px
    assert img.height == expected_px


def test_render_origin_reflects_padding():
    size = 10.0
    resolution = 0.05
    padding = 5
    wd = WorldDescription(zones=[_make_square_zone(size=size)])
    _, origin = wd.render(resolution=resolution)
    expected_origin_x = 0.0 - padding * resolution
    expected_origin_y = 0.0 - padding * resolution
    assert origin[0] == pytest.approx(expected_origin_x)
    assert origin[1] == pytest.approx(expected_origin_y)


def test_world_save_produces_directory_not_tarfile(tmp_path):
    from arena_simulation_setup.tree.World.World import World
    w = World(tmp_path / "myworld")
    wd = WorldDescription(zones=[_make_square_zone()])
    result = w.save(wd)
    assert result.is_dir()


def test_world_save_extracts_world_yaml(tmp_path):
    from arena_simulation_setup.tree.World.World import World
    w = World(tmp_path / "myworld")
    wd = WorldDescription(zones=[_make_square_zone()])
    w.save(wd)
    assert (tmp_path / "myworld" / "world.yaml").exists()


def test_world_save_load_roundtrip_zone_count(tmp_path):
    from arena_simulation_setup.tree.World.World import World
    w = World(tmp_path / "myworld")
    wd = WorldDescription(zones=[_make_square_zone("a"), _make_square_zone("b")])
    w.save(wd)
    loaded = w.load()
    assert len(loaded.zones) == 2


def test_world_save_load_roundtrip_zone_names(tmp_path):
    from arena_simulation_setup.tree.World.World import World
    w = World(tmp_path / "myworld")
    wd = WorldDescription(zones=[_make_square_zone("alpha"), _make_square_zone("beta")])
    w.save(wd)
    loaded = w.load()
    names = {z.name for z in loaded.zones}
    assert names == {"alpha", "beta"}


def test_world_save_load_roundtrip_wall_count(tmp_path):
    from arena_simulation_setup.tree.World.World import World
    w = World(tmp_path / "myworld")
    zone = _make_zone(
        name="z",
        corners=[Position(0, 0), Position(5, 0), Position(5, 5), Position(0, 5)],
        walls=[
            Wall(start=Position(0, 0), end=Position(5, 0)),
            Wall(start=Position(5, 0), end=Position(5, 5)),
        ],
    )
    wd = WorldDescription(zones=[zone])
    w.save(wd)
    loaded = w.load()
    assert len(loaded.zones[0].walls) == 2


def test_world_save_load_roundtrip_corner_positions(tmp_path):
    from arena_simulation_setup.tree.World.World import World
    w = World(tmp_path / "myworld")
    corners = [Position(0.0, 0.0), Position(3.0, 0.0), Position(3.0, 4.0), Position(0.0, 4.0)]
    zone = _make_zone(name="z", corners=corners)
    wd = WorldDescription(zones=[zone])
    w.save(wd)
    loaded = w.load()
    loaded_corners = loaded.zones[0].corners
    assert len(loaded_corners) == 4
    assert loaded_corners[0].x == pytest.approx(0.0)
    assert loaded_corners[1].x == pytest.approx(3.0)


def test_world_save_map_only_writes_map_skips_world_yaml(tmp_path):
    from arena_simulation_setup.tree.World.World import World
    w = World(tmp_path / "myworld")
    wd = WorldDescription(zones=[_make_square_zone()])
    w.save(wd, map_only=True)
    assert (tmp_path / "myworld" / "map" / "map.yaml").exists()
    assert not (tmp_path / "myworld" / "world.yaml").exists()


def test_world_save_through_symlinked_subdir(tmp_path):
    from arena_simulation_setup.tree.World.World import World
    world_dir = tmp_path / "myworld"
    world_dir.mkdir()
    map_target = tmp_path / "shared_map"
    map_target.mkdir()
    (world_dir / "map").symlink_to(map_target, target_is_directory=True)

    w = World(world_dir)
    wd = WorldDescription(zones=[_make_square_zone()])
    w.save(wd)

    assert (map_target / "map.yaml").exists()
    assert (world_dir / "world.yaml").exists()


def test_world_scenario_listall_empty_when_no_scenarios_dir(tmp_path):
    from arena_simulation_setup.tree.World.World import World
    w = World(tmp_path / "myworld")
    (tmp_path / "myworld").mkdir()
    scenarios = list(w.scenario.listall())
    assert scenarios == []


def test_world_scenario_listall_discovers_subdirs(tmp_path):
    from arena_simulation_setup.tree.World.World import World
    world_dir = tmp_path / "myworld"
    scenarios_dir = world_dir / "scenarios"
    scenarios_dir.mkdir(parents=True)
    (scenarios_dir / "alpha").mkdir()
    (scenarios_dir / "beta").mkdir()
    (scenarios_dir / "not_a_dir.txt").write_text("")
    w = World(world_dir)
    names = {s.name for s in w.scenario.listall()}
    assert names == {"alpha", "beta"}


def test_world_scenario_listall_ignores_files(tmp_path):
    from arena_simulation_setup.tree.World.World import World
    world_dir = tmp_path / "myworld"
    scenarios_dir = world_dir / "scenarios"
    scenarios_dir.mkdir(parents=True)
    (scenarios_dir / "only_file.yaml").write_text("")
    w = World(world_dir)
    names = list(w.scenario.listall())
    assert names == []


