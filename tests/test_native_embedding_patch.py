"""Compile the deployed patch's allocation code against a minimal ggml metadata fixture."""

import shutil
import subprocess
from pathlib import Path

import pytest


def test_native_patch_keeps_embedding_storage_type_without_rounding(tmp_path):
    compiler = shutil.which("c++")
    if not compiler:
        pytest.skip("A C++ compiler is needed for the native allocation regression")
    patch = (
        Path(__file__).resolve().parents[1]
        / "workers/vla_cpp/policykit/patches/vla-cpp-smolvla-packed.patch"
    ).read_text()
    # Exercise the exact deployed allocation statements, not a Python model of
    # them. This test would fail for the previous hard-coded BF16 allocation.
    added = [
        line[1:]
        for line in patch.splitlines()
        if line.startswith("+") and not line.startswith("+++")
    ]
    start = added.index("    ggml_type embedding_type = GGML_TYPE_BF16;")
    end = next(i for i in range(start, len(added)) if "m->E_lang = ggml_new_tensor_2d" in added[i])
    allocation = "\n".join(added[start : end + 1])
    source = tmp_path / "embedding.cpp"
    source.write_text(
        """
#include <cassert>
#include <cstdio>
#include <initializer_list>
enum ggml_type { GGML_TYPE_F32, GGML_TYPE_F16, GGML_TYPE_BF16, GGML_TYPE_Q4_0 };
struct ggml_tensor { ggml_type type; };
struct Model { ggml_tensor * E_lang = nullptr; ~Model() { delete E_lang; } };
ggml_tensor * ggml_get_tensor(ggml_tensor * context, const char *) { return context; }
ggml_tensor * ggml_new_tensor_2d(void *, ggml_type type, int, int) {
    return new ggml_tensor{type};
}
Model * allocate(bool use_gguf, ggml_tensor * input) {
    auto * m = new Model;
    struct { int hidden = 960; } cfg;
    struct { ggml_tensor * meta_ctx; } gst{input};
    void * ctx = nullptr;
"""
        + allocation
        + """
    return m;
}
int main() {
    for (auto type : {GGML_TYPE_F32, GGML_TYPE_F16, GGML_TYPE_BF16}) {
        ggml_tensor input{type};
        Model * model = allocate(true, &input);
        assert(model && model->E_lang->type == type);
        delete model;
    }
    ggml_tensor packed{GGML_TYPE_Q4_0};
    assert(allocate(true, &packed) == nullptr);
    assert(allocate(true, nullptr) == nullptr);
    Model * original = allocate(false, nullptr);
    assert(original && original->E_lang->type == GGML_TYPE_BF16);
    delete original;
}
"""
    )
    executable = tmp_path / "embedding"
    subprocess.run(
        [compiler, "-std=c++17", str(source), "-o", str(executable)],
        check=True,
        capture_output=True,
    )
    subprocess.run([str(executable)], check=True, capture_output=True)
