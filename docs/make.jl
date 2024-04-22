# Use
#
#     DOCUMENTER_DEBUG=true julia --color=yes make.jl local [nonstrict] [fixdoctests]
#
# for local builds.

using Documenter
import Literate
using MGVI

SRC=joinpath(@__DIR__, "src")

GENERATED = joinpath(@__DIR__, "build")
GENERATED_SRC = joinpath(GENERATED, "src")
GENERATED_DOCS = joinpath(GENERATED, "docs")

EXECUTE_MD = true

mkpath(GENERATED)
cp(SRC, GENERATED_SRC, force = true)

function dir_replace(content)
    content = replace(content, "@__DIR__" => '"' * GENERATED_SRC * '"')
    return content
end

Literate.notebook(joinpath(GENERATED_SRC, "advanced_tutorial_lit.jl"),
                  GENERATED_SRC; name="advanced_tutorial", execute=false)
Literate.script(joinpath(GENERATED_SRC, "advanced_tutorial_lit.jl"),
                GENERATED_SRC; name="advanced_tutorial")
Literate.markdown(joinpath(GENERATED_SRC, "advanced_tutorial_lit.jl"),
                  GENERATED_SRC; name="advanced_tutorial", execute=EXECUTE_MD, preprocess=dir_replace)

Literate.notebook(joinpath(GENERATED_SRC, "tutorial_lit.jl"),
                  GENERATED_SRC; name="tutorial", execute=false)
Literate.script(joinpath(GENERATED_SRC, "tutorial_lit.jl"),
                GENERATED_SRC; name="tutorial")
Literate.markdown(joinpath(GENERATED_SRC, "tutorial_lit.jl"),
                  GENERATED_SRC; name="tutorial", execute=EXECUTE_MD)

makedocs(
    sitename = "MGVI",
    modules = [MGVI],
    root = GENERATED,
    source = "src",
    build = "docs",
    format = Documenter.HTML(
        prettyurls = !("local" in ARGS),
        canonical = "https://bat.github.io/MGVI.jl/stable/"
    ),
    pages = [
        "Home" => "index.md",
        "Tutorial" => "tutorial.md",
        "Advanced Tutorial" => "advanced_tutorial.md",
        "API" => "api.md",
        "LICENSE" => "LICENSE.md",
    ],
    doctest = ("fixdoctests" in ARGS) ? :fix : true,
    linkcheck = !("nonstrict" in ARGS),
    warnonly = ("nonstrict" in ARGS),
)

deploydocs(
    repo = "github.com/bat/MGVI.jl.git",
    forcepush = true,
    push_preview = true,
)
