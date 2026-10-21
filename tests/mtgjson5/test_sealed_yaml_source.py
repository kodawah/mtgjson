"""Reading mtg-sealed-content's product YAML.

mtg-sealed-content keeps each sealed product in data/products/SET.yaml, with
what is inside it nested under a ``contents`` key. It used to keep the contents
in data/contents/SET.yaml (folded in by mtg-sealed-content#782); a file there
is ignored, so it can no longer change what the products compile to.
"""

from __future__ import annotations

import asyncio
import io
import logging
import tarfile
import tempfile
from pathlib import Path

import pytest
import yaml

from mtgjson5.pipeline.stages.sealed import compile_contents, compile_products
from mtgjson5.providers.github.provider import SealedDataProvider, _build_sealed_products_records, _to_lazyframe

PRODUCTS = {
    "tst": {
        "Test Anniversary Bundle": {"category": "BUNDLE", "subtype": "DEFAULT", "identifiers": {"mcmId": "3"}},
        "Test Booster Box": {"category": "BOOSTER_BOX", "subtype": "DRAFT", "identifiers": {"tcgplayerProductId": "1"}},
        "Test Booster Pack": {"category": "BOOSTER_PACK", "subtype": "DRAFT", "identifiers": {}},
        "Test Bundle": {
            "category": "BUNDLE",
            "subtype": "DEFAULT",
            "release_date": "2024-08-02",
            "identifiers": {"tcgplayerProductId": "2"},
        },
        "Test Mystery Box": {"category": "BOX_SET", "subtype": "DEFAULT", "identifiers": {}},
        "Test Showdown Booster": {
            "category": "BOOSTER_PACK",
            "subtype": "PREMIER",
            "language": "Japanese",
            "identifiers": {"mcmId": "5"},
        },
    },
    "ts2": {
        "Second Set Booster Pack": {"category": "BOOSTER_PACK", "subtype": "DEFAULT", "identifiers": {}},
        "Second Set Bundle": {"category": "BUNDLE", "subtype": "DEFAULT", "identifiers": {}},
    },
}

# What each product nests under ``contents``; unresearched products have none.
CONTENTS = {
    "tst": {
        # Sorted product names often put a copy ahead of the product it copies.
        "Test Anniversary Bundle": {"copy": "Test Bundle"},
        "Test Booster Box": {
            "sealed": [{"set": "tst", "count": 36, "name": "Test Booster Pack"}],
            # YAML reads one number as a string and the other as an int.
            "card": [
                {"set": "tst", "number": "2a", "name": "Second Card"},
                {"set": "tst", "number": 1, "name": "Test Card"},
            ],
        },
        "Test Booster Pack": {"pack": [{"set": "tst", "code": "draft"}], "card_count": 15},
        "Test Bundle": {
            "sealed": [{"set": "tst", "count": 9, "name": "Test Booster Pack"}],
            "deck": [{"set": "tst", "name": "Test Deck"}],
            "card": [{"set": "tst", "number": 1, "name": "Test Card", "foil": True}],
            "other": [{"name": "Spindown life counter"}],
            "card_count": 1,
        },
        "Test Mystery Box": {
            "variable_mode": {"count": 1},
            "variable": [
                {"deck": [{"set": "tst", "name": "Test Deck"}]},
                {"card": [{"set": "tst", "number": 3, "name": "Third Card"}]},
            ],
        },
    },
    "ts2": {},
}


def _dump(path: Path, data: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.safe_dump(data, allow_unicode=True), encoding="utf-8")


def _write_products(root: Path, with_contents: bool = True) -> Path:
    for code, products in PRODUCTS.items():
        entries = {
            name: {**info, "contents": CONTENTS[code][name]} if with_contents and name in CONTENTS[code] else dict(info)
            for name, info in products.items()
        }
        _dump(root / "data" / "products" / f"{code.upper()}.yaml", {"code": code, "products": entries})
    return root / "data" / "products"


def _uuid_map() -> dict:
    return {
        "tst": {
            "cards": {"1": ("card-1", "Test Card"), "2a": ("card-2a", "Second Card"), "3": ("card-3", "Third Card")},
            "tokens": {},
            "booster": {"draft"},
            "decks": {"Test Deck"},
            "sealedProduct": {name: f"uuid-{name}" for name in PRODUCTS["tst"]},
        }
    }


def test_compile_products_drops_nested_contents(tmp_path):
    assert compile_products(_write_products(tmp_path)) == PRODUCTS


def test_sealed_products_frame_builds_from_products_with_contents(tmp_path):
    """Nested contents would reach the frame, and polars cannot build one list of mixed card numbers."""
    with_contents = compile_products(_write_products(tmp_path / "with"))
    without = compile_products(_write_products(tmp_path / "without", with_contents=False))

    frame = _to_lazyframe(_build_sealed_products_records(with_contents), "sealed_products", "sealed_products").collect()
    expected = _to_lazyframe(_build_sealed_products_records(without), "sealed_products", "sealed_products").collect()

    assert "contents" not in frame.columns
    assert frame.equals(expected)


def test_compile_contents(tmp_path, caplog):
    products_dir = _write_products(tmp_path)
    # An old data/contents/ tree next to it is not read.
    _dump(tmp_path / "data" / "contents" / "TST.yaml", {"code": "tst", "products": {"Test Booster Pack": {}}})

    with caplog.at_level(logging.WARNING, logger="mtgjson5.pipeline.stages.sealed"):
        contents, deck_map = compile_contents(products_dir, _uuid_map())

    assert [r.getMessage() for r in caplog.records] == [
        "Product ts2 - Second Set Booster Pack missing contents",
        "Product ts2 - Second Set Bundle missing contents",
        "Product tst - Test Showdown Booster missing contents",
    ]
    # Nothing researched in ts2, and the unresearched product is left out.
    assert list(contents) == ["tst"]
    assert list(contents["tst"]) == [
        "Test Anniversary Bundle",
        "Test Booster Box",
        "Test Booster Pack",
        "Test Bundle",
        "Test Mystery Box",
    ]
    assert contents["tst"]["Test Anniversary Bundle"] == contents["tst"]["Test Bundle"]
    assert contents["tst"]["Test Booster Pack"]["pack"]
    assert contents["tst"]["Test Bundle"]["card"] == [
        {"name": "Test Card", "set": "tst", "number": "1", "uuid": "card-1", "foil": True}
    ]
    configs = contents["tst"]["Test Mystery Box"]["variable"][0]["configs"]
    assert [sorted(config) for config in configs] == [["deck", "variable_config"], ["card", "variable_config"]]
    assert all(config["variable_config"] == [{"chance": 1, "weight": 2}] for config in configs)
    # A copy holds its target's decks too, so both link back from the deck.
    assert deck_map == {"tst": {"Test Deck": ["uuid-Test Anniversary Bundle", "uuid-Test Bundle"]}}


# ---------------------------------------------------------------------------
# Tarball extraction
# ---------------------------------------------------------------------------

TARBALL_ROOT = "mtgjson-mtg-sealed-content-0123abc/"
SET_FILE = "code: tst\nproducts: {}\n"
TARBALL = {
    "README.md": "# mtg-sealed-content\n",
    "data/products/TST.yaml": SET_FILE,
    "outputs/deck_map.json": "{}",
}
# A tarball that still carries a data/contents/ file, e.g. one an old branch brought back
STRAY_CONTENTS_TARBALL = {**TARBALL, "data/contents/TST.yaml": SET_FILE}


def _tarball(files: dict[str, str]) -> bytes:
    buffer = io.BytesIO()
    with tarfile.open(fileobj=buffer, mode="w:gz") as tar:
        for name, text in files.items():
            payload = text.encode()
            member = tarfile.TarInfo(TARBALL_ROOT + name)
            member.size = len(payload)
            tar.addfile(member, io.BytesIO(payload))
    return buffer.getvalue()


class FakeResponse:
    def __init__(self, payload: bytes):
        self._payload = payload

    async def __aenter__(self) -> FakeResponse:
        return self

    async def __aexit__(self, *args: object) -> None:
        return None

    def raise_for_status(self) -> None:
        return None

    async def read(self) -> bytes:
        return self._payload


class FakeSession:
    def __init__(self, tarball: bytes):
        self._tarball = tarball
        self.calls = 0

    def get(self, url: str) -> FakeResponse:
        self.calls += 1
        return FakeResponse(self._tarball)


def _extract(cache_path: Path | None, session: FakeSession) -> Path:
    provider = SealedDataProvider(cache_path=cache_path)
    return asyncio.run(provider._fetch_and_extract_yaml(session))


def _yaml_names(directory: Path) -> list[str]:
    return sorted(path.name for path in directory.glob("*.yaml"))


@pytest.mark.parametrize("cached", [True, False], ids=["cache_path", "tempdir"])
def test_extracts_only_the_product_files(tmp_path, monkeypatch, cached):
    monkeypatch.setattr(tempfile, "tempdir", str(tmp_path))

    products_dir = _extract(tmp_path if cached else None, FakeSession(_tarball(TARBALL)))

    assert _yaml_names(products_dir) == ["TST.yaml"]
    assert not (products_dir.parent / "contents").exists()


def test_a_stray_data_contents_file_is_ignored(tmp_path, caplog):
    with caplog.at_level(logging.WARNING, logger="mtgjson5.providers.github.provider"):
        products_dir = _extract(tmp_path, FakeSession(_tarball(STRAY_CONTENTS_TARBALL)))

    assert _yaml_names(products_dir) == ["TST.yaml"]
    assert not (products_dir.parent / "contents").exists()
    assert "Ignoring 1 data/contents/ files in the sealed tarball" in caplog.text
    assert "data/contents/TST.yaml" in caplog.text


def test_extraction_starts_from_an_empty_directory(tmp_path):
    """Nothing a crashed extraction left in the staging directory may survive."""
    leftover = tmp_path / "sealed_yaml.partial" / "data" / "products" / "GONE.yaml"
    leftover.parent.mkdir(parents=True)
    leftover.write_text(SET_FILE)

    products_dir = _extract(tmp_path, FakeSession(_tarball(TARBALL)))

    assert _yaml_names(products_dir) == ["TST.yaml"]
    assert not (tmp_path / "sealed_yaml.partial").exists()


def test_cached_extraction_is_reused(tmp_path):
    session = FakeSession(_tarball(TARBALL))

    first = _extract(tmp_path, session)
    second = _extract(tmp_path, session)

    assert second == first
    assert session.calls == 1


def test_a_cached_old_layout_extraction_is_fetched_again(tmp_path):
    """Its product files have no nested contents, so using it would empty every product."""
    for directory in ("products", "contents"):
        stale = tmp_path / "sealed_yaml" / "data" / directory / "OLD.yaml"
        stale.parent.mkdir(parents=True)
        stale.write_text(SET_FILE)
    session = FakeSession(_tarball(TARBALL))

    products_dir = _extract(tmp_path, session)

    assert session.calls == 1
    assert _yaml_names(products_dir) == ["TST.yaml"]
    assert not (tmp_path / "sealed_yaml" / "data" / "contents").exists()
