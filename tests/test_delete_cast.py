"""Delete a character from the Cast tab (Run 7): the same trash as Assets (DELETE /api/projects/{pid}/cast/{cid}). Shots
and takes that used the character still open with their settings; a new take goes without it, whichever way it was
named (a picked reference, cast_id, an @-mention, a multi-person Recast); Restore puts it back with its uses."""

from test_delete_project_asset import client, make_cast, make_shot, store, tree  # noqa: F401 - the fixtures

from pickups import mentions


def test_delete_a_character_shots_use(client, store):  # noqa: F811
    c, _ = client
    a = store.create_project("Paris")["id"]
    make_cast(store, a, "cast_maya", "Maya")
    make_cast(store, a, "cast_leo", "Leo")
    params = {"prompt": "@Maya walks to @Leo.", "look_refs": [{"kind": "cast", "id": "cast_maya"}, {"kind": "cast", "id": "cast_leo"}]}
    make_shot(store, "shot_p1", a, params=params)
    make_shot(store, "shot_p2", a, params={"who_ref": {"kind": "cast", "id": "cast_maya"}, "swaps": [{"cast_id": "cast_maya"}]})
    assert {x["id"]: x["uses"] for x in c.get(f"/api/projects/{a}").json()["cast"]} == {"cast_maya": 2, "cast_leo": 1}
    leo = tree(store.project_dir(a) / "cast")

    r = c.delete(f"/api/projects/{a}/cast/cast_maya")
    assert r.status_code == 200, r.text
    entry = r.json()
    assert entry["kind"] == "cast" and entry["name"] == "Maya"
    assert (store.trash_root / entry["id"] / "projects" / a / "cast" / "cast_maya.png").is_file()
    assert tree(store.project_dir(a) / "cast") == {k: v for k, v in leo.items() if k.startswith("cast_leo")}

    # The shots and their takes still open, settings untouched.
    for sid, want in (("shot_p1", params), ("shot_p2", {"who_ref": {"kind": "cast", "id": "cast_maya"}, "swaps": [{"cast_id": "cast_maya"}]})):
        shot = c.get(f"/api/shots/{sid}")
        assert shot.status_code == 200 and shot.json()["versions"][-1]["params"] == want
    assert c.get(f"/api/projects/{a}").status_code == 200

    # A new take goes without her: the picked reference is skipped, the @-mention no longer brings her picture.
    leo_png = store.project_dir(a) / "cast" / "cast_leo.png"
    assert store.live_refs(a, params["look_refs"]) == [leo_png]
    cast, _file = store.shot_cast("shot_p1")
    assert [x["id"] for x in mentions.mentioned(params["prompt"], cast)] == ["cast_leo"]
    _text, used = mentions.for_images(params["prompt"], cast)
    assert [x["id"] for x in used] == ["cast_leo"]
    assert c.get(f"/api/projects/{a}/refs/cast/cast_maya").status_code == 404

    # Undo / Restore: back in the cast, with her uses.
    assert c.post(f"/api/trash/{entry['id']}/restore").status_code == 200
    assert {x["id"]: x["uses"] for x in c.get(f"/api/projects/{a}").json()["cast"]} == {"cast_maya": 2, "cast_leo": 1}
    assert c.get("/api/trash").json() == []


def test_a_restored_character_never_takes_a_new_ones_name(client, store):  # noqa: F811
    """Delete Maya, make a new Maya, then Restore: refused (409), and the new Maya is untouched."""
    c, _ = client
    a = store.create_project("Paris")["id"]
    make_cast(store, a, "cast_old", "Maya")
    entry = c.delete(f"/api/projects/{a}/cast/cast_old").json()
    make_cast(store, a, "cast_new", "Maya")
    assert c.post(f"/api/trash/{entry['id']}/restore").status_code == 409
    assert [x["id"] for x in c.get(f"/api/projects/{a}").json()["cast"]] == ["cast_new"]
    assert [e["id"] for e in c.get("/api/trash").json()] == [entry["id"]]
