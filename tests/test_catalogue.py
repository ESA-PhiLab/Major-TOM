"""docs/ARCHIVES.md's products table matches collections.py (offline)."""
import pytest

pytest.importorskip("pystac_client")
from majortom.build import catalogue                             # noqa: E402


def test_products_table_is_current():
    text = catalogue.DOC.read_text()
    assert catalogue.updated(text) == text, "run: python -m majortom.build.catalogue"


def test_every_product_has_a_producer_page():
    from majortom.build.collections import COLLECTIONS, PRODUCT_PAGES

    assert set(COLLECTIONS) <= set(PRODUCT_PAGES), set(COLLECTIONS) - set(PRODUCT_PAGES)


def test_every_product_has_an_example_and_a_preview():
    """Contributors add these with the product: the network product test, the gallery and the entry notebook use
    them, so a new product is tested and shown without other edits (CONTRIBUTING.md section 3)."""
    from majortom.build.collections import COLLECTIONS

    for name, c in COLLECTIONS.items():
        assert c.preview is not None, f"{name}: add preview=Preview(...)"
        assert (c.example is None) == (c.time == "static"), f"{name}: example dates (None only for static products)"
        bands = set(c.assets) | set(c.optional)
        named = list(c.preview.rgb or ()) + [b for b in (c.preview.band, c.preview.classes) if b]
        assert set(named) <= bands, f"{name}: preview bands {set(named) - bands} are not among its bands"
