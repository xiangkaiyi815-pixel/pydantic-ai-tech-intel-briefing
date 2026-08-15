from search_assistant.search.source_registry import (
    normalize_source_recipe,
    source_contract_for_url,
    source_contracts_as_dicts,
    source_recipe_summary,
)


def test_source_contracts_keep_public_read_only_boundary():
    contracts = source_contracts_as_dicts()

    assert contracts
    assert all(contract["public_access"] is True for contract in contracts)
    assert all(contract["requires_login"] is False for contract in contracts)
    assert all("write" in contract["unsupported_actions"] for contract in contracts)


def test_source_contract_for_url_maps_public_platform_family():
    contract = source_contract_for_url("https://www.bilibili.com/video/BV1test")

    assert contract is not None
    assert contract.slug == "bilibili"


def test_source_recipe_normalizes_aliases_and_ignores_unknown_sources():
    recipe = normalize_source_recipe({"web": 3, "bili": 2, "unknown": 9, "reddit": 0})

    assert recipe == {"general-web": 3.0, "bilibili": 2.0}
    summary = source_recipe_summary(recipe)
    general = next(source for source in summary["sources"] if source["slug"] == "general-web")
    assert general["weight"] == 3.0
