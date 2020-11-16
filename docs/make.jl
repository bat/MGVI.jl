# Use
#
#     DOCUMENTER_DEBUG=true julia --color=yes make.jl local [nonstrict] [fixdoctests]
#
# for local builds.

using Documenter
import Literate
using MGVInference

SRC=joinpath(@__DIR__, "src")

GENERATED = joinpath(@__DIR__, "build")
GENERATED_SRC = joinpath(GENERATED, "src")
GENERATED_DOCS = joinpath(GENERATED, "docs")

EXECUTE_NOTEBOOKS = true

mkdir(GENERATED)
cp(SRC, GENERATED_SRC)

Literate.notebook(joinpath(GENERATED_SRC, "examples/visualize_polyfit_lit.jl"),
                  joinpath(GENERATED_SRC, "examples/"); name="visualize_polyfit", execute=EXECUTE_NOTEBOOKS)
Literate.script(joinpath(GENERATED_SRC, "examples/visualize_polyfit_lit.jl"),
                joinpath(GENERATED_SRC, "examples/"); name="visualize_polyfit")
Literate.notebook(joinpath(GENERATED_SRC, "examples/visualize_fft_gp_lit.jl"),
                  joinpath(GENERATED_SRC, "examples/"); name="visualize_fft_gp", execute=EXECUTE_NOTEBOOKS)
Literate.script(joinpath(GENERATED_SRC, "examples/visualize_fft_gp_lit.jl"),
                joinpath(GENERATED_SRC, "examples/"); name="visualize_fft_gp")

makedocs(
    sitename = "MGVInference",
    modules = [MGVInference],
    root = GENERATED,
    source = "src",
    build = "docs",
    format = Documenter.HTML(
        prettyurls = !("local" in ARGS),
        canonical = "https://bat.github.io/MGVInference.jl/stable/"
    ),
    pages = [
        "Home" => "index.md",
        "Examples" => "examples/examples.md",
        "API" => "api.md",
        "LICENSE" => "LICENSE.md",
    ],
    doctest = ("fixdoctests" in ARGS) ? :fix : true,
    linkcheck = ("linkcheck" in ARGS),
    strict = !("nonstrict" in ARGS),
)

if ("local" in ARGS)
    return
end

deploydocs(
    root = GENERATED,
    build = "docs",
    repo = "github.com/bat/MGVInference.jl.git",
    forcepush = false,
    push_preview = false
)
