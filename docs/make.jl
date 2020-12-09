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

EXECUTE_MD = true

mkdir(GENERATED)
cp(SRC, GENERATED_SRC)

Literate.notebook(joinpath(GENERATED_SRC, "tutorial_lit.jl"),
                  GENERATED_SRC; name="tutorial", execute=false)
Literate.markdown(joinpath(GENERATED_SRC, "tutorial_lit.jl"),
                  GENERATED_SRC; name="tutorial", execute=EXECUTE_MD)
Literate.script(joinpath(GENERATED_SRC, "tutorial_lit.jl"),
                GENERATED_SRC; name="tutorial")

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
        "Tutorial" => "tutorial.md",
        "API" => "api.md",
        "LICENSE" => "LICENSE.md",
    ],
    doctest = ("fixdoctests" in ARGS) ? :fix : true,
    linkcheck = ("linkcheck" in ARGS),
    strict = !("nonstrict" in ARGS),
)

deploydocs(
    root = GENERATED,
    target = "docs",
    repo = "github.com/bat/MGVInference.jl.git",
    forcepush = true,
    push_preview = true
)
