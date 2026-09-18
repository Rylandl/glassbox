"""Version-independent frozen source identity, including adversarial literals."""

import ast

import pytest

from glassbox.experimental import api_migration as migration

NO_RELOCATIONS = dict(import_relocations=[], doc_reference_relocations=[])
COMPATIBLE_SOURCES = (
    "src/glassbox/experimental/harness.py",
    "src/glassbox/experimental/learned_plan.py",
    "examples/cascade_accuracy.py",
)


@pytest.mark.parametrize("name", COMPATIBLE_SOURCES)
def test_frozen_source_pins_do_not_depend_on_runtime_ast_dump(monkeypatch, name):
    plan, _ = migration.frozen_plan()
    contract = plan["qualification_compatibility"]["files"][name]

    def unavailable(*args, **kwargs):
        raise AssertionError("runtime ast.dump formatting must not enter frozen pins")

    monkeypatch.setattr(ast, "dump", unavailable)
    assert migration.qualification_source_matches(
        migration.ROOT, name, contract["original_sha256"]
    )


def test_compact_format_matches_original_empty_field_omission():
    tree = ast.parse("def f():\n    pass\n")
    assert migration._frozen_ast_dump(tree) == (
        "Module(body=[FunctionDef(name='f', args=arguments(), body=[Pass()])])"
    )


def test_empty_container_nodes_and_none_constants_remain_in_format():
    tree = ast.parse("([], (), {}, None)", mode="eval")
    assert migration._frozen_ast_dump(tree) == (
        "Expression(body=Tuple(elts=[List(ctx=Load()), Tuple(ctx=Load()), "
        "Dict(), Constant(value=None)], ctx=Load()))"
    )


def test_string_literals_are_not_edited_as_if_they_were_ast_fields():
    text = "args=[], type_params=[], type_ignores=[], keywords=[], value=None"
    tree = ast.parse(repr(text), mode="eval")
    assert migration._frozen_ast_dump(tree) == (
        f"Expression(body=Constant(value={text!r}))"
    )


@pytest.mark.parametrize(
    "original, altered",
    [
        ("value = []", "value = ()"),
        ("value = []", "value = [None]"),
        ("value = {}", "value = {None: None}"),
        ("value = None", "value = False"),
        ("value = '[]'", "value = ''"),
        ("value = 'args=[]'", "value = 'args='"),
        ("'type_ignores=[], value=None'", "'value=None'"),
        ("def f(): pass", "def f(value): pass"),
        ("def f(value=None): pass", "def f(value=[]): pass"),
        ("def f(value: int): pass", "def f(value: str): pass"),
        ("def f() -> int: pass", "def f() -> str: pass"),
        ("def f(): return", "def f(): return None"),
        ("def f(): pass", "async def f(): pass"),
        ("if ready:\n pass", "if ready:\n pass\nelse:\n pass"),
        ("f()", "f(None)"),
        ("f()", "f(value=None)"),
        ("f(value)", "g(value)"),
        ("value + 1", "value - 1"),
        ("from package import name", "from other import name"),
        ("from .package import name", "from ..package import name"),
        ("from package import name", "from package import name as other"),
        ("import package", "import package as other"),
        ("import a\nf()\nimport b", "import b\nf()\nimport a"),
        ("type Alias[T] = tuple[T]", "type Alias[T] = list[T]"),
    ],
)
def test_nonrelocation_statement_fields_remain_distinguishable(original, altered):
    assert migration.normalized_source(
        original, NO_RELOCATIONS
    ) != migration.normalized_source(altered, NO_RELOCATIONS)


def test_only_declared_import_and_doc_reference_relocations_are_normalized():
    contract = dict(
        import_relocations=[
            dict(old_module="old", old_level=1, new_module="new", new_level=2)
        ],
        doc_reference_relocations=[dict(old="old.path", new="new.path")],
    )
    original = "'old.path; args=[]'\nfrom .old import beta, alpha\nimport z, a\n"
    migrated = "'new.path; args=[]'\nimport a, z\nfrom ..new import alpha, beta\n"
    expected = migration.normalized_source(original, contract)
    assert migration.normalized_source(migrated, contract) == expected
    assert (
        migration.normalized_source(migrated.replace("args=[]", "args="), contract)
        != expected
    )
    assert (
        migration.normalized_source(migrated.replace("..new", ".new"), contract)
        != expected
    )
