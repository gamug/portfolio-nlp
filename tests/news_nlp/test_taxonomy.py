"""news_nlp.taxonomy: CATEGORY_GROUPS / CATEGORY_SLUGS invariants."""

from __future__ import annotations

from news_nlp.taxonomy import (
    CATEGORY_GROUP_CHILDREN,
    CATEGORY_GROUP_SLUGS,
    CATEGORY_GROUPS,
    CATEGORY_LABELS_BY_SLUG,
    CATEGORY_SLUG_TO_GROUP,
    CATEGORY_SLUGS,
)


def test_category_groups_partition_category_slugs_exactly() -> None:
    all_children = [c for _, _, _, children in CATEGORY_GROUPS for c in children]
    assert sorted(all_children) == sorted(CATEGORY_SLUGS)  # no leftovers, no duplicates
    assert len(set(all_children)) == len(all_children)


def test_each_group_has_three_children() -> None:
    for slug, _, _, children in CATEGORY_GROUPS:
        assert len(children) == 3, slug


def test_category_group_slugs_are_unique() -> None:
    assert len(set(CATEGORY_GROUP_SLUGS)) == len(CATEGORY_GROUP_SLUGS)


def test_category_group_children_matches_category_groups() -> None:
    for slug, _, _, children in CATEGORY_GROUPS:
        assert CATEGORY_GROUP_CHILDREN[slug] == children


def test_category_slug_to_group_is_the_inverse_of_category_group_children() -> None:
    for group_slug, children in CATEGORY_GROUP_CHILDREN.items():
        for child in children:
            assert CATEGORY_SLUG_TO_GROUP[child] == group_slug


def test_category_labels_by_slug_covers_every_category_slug() -> None:
    assert set(CATEGORY_LABELS_BY_SLUG) == set(CATEGORY_SLUGS)
    for slug, (label_slug, _display, _phrase) in CATEGORY_LABELS_BY_SLUG.items():
        assert slug == label_slug
