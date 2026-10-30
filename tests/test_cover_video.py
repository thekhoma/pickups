"""A project card plays its cover on hover (Run 7): cover_video_url is the clip of the take whose still is the cover
(the newest shot's newest take), and None when the cover is an image or there's nothing."""

from test_delete_project_asset import client, make_image, make_shot, store  # noqa: F401 - the fixtures


def test_cover_video_is_the_covers_clip(client, store):  # noqa: F811
    c, _ = client
    a = store.create_project("Paris")["id"]
    make_shot(store, "shot_c1", a, params={"prompt": "snow"})
    p = next(x for x in c.get("/api/projects").json() if x["id"] == a)
    shot = store.shot("shot_c1")
    assert p["cover_url"] == shot["latest_thumb_url"] == shot["versions"][-1]["thumb_url"]
    assert p["cover_video_url"] == shot["versions"][-1]["video_url"] and p["cover_video_url"]


def test_no_cover_video_without_shots(client, store):  # noqa: F811
    c, _ = client
    b = store.create_project("Lisbon")["id"]
    assert next(x for x in c.get("/api/projects").json() if x["id"] == b)["cover_video_url"] is None
    make_image(store, b, "img_b1")
    p = next(x for x in c.get("/api/projects").json() if x["id"] == b)
    assert p["cover_url"] and p["cover_video_url"] is None
