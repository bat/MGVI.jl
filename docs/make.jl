# Use
#
#     DOCUMENTER_DEBUG=true julia --color=yes make.jl local [nonstrict] [fixdoctests]
#
# for local builds.

using Documenter
import Literate
using MGVI

SRC=joinpath(@__DIR__, "src")


function fix_literate_output(content)
    content = replace(content, "EditURL = \"@__REPO_ROOT_URL__/\"" => "")
    return content
end

gen_content_dir = joinpath(@__DIR__, "src")

tutorial_src = joinpath(@__DIR__, "src", "tutorial_lit.jl")
Literate.markdown(tutorial_src, gen_content_dir, name = "tutorial", documenter = true, credit = true, postprocess = fix_literate_output)
#Literate.markdown(tutorial_src, gen_content_dir, name = "tutorial", codefence = "```@repl tutorial" => "```", documenter = true, credit = true)
Literate.notebook(tutorial_src, gen_content_dir, execute = false, name = "mgvi_tutorial", documenter = true, credit = true)
Literate.script(tutorial_src, gen_content_dir, keep_comments = false, name = "mgvi_tutorial", documenter = true, credit = false)

advanced_tutorial_src = joinpath(@__DIR__, "src", "advanced_tutorial_lit.jl")
Literate.markdown(advanced_tutorial_src, gen_content_dir, name = "advanced_tutorial", documenter = true, credit = true, postprocess = fix_literate_output)
#Literate.markdown(advanced_tutorial_src, gen_content_dir, name = "advanced_tutorial", codefence = "```@repl advanced_tutorial" => "```", documenter = true, credit = true)
Literate.notebook(advanced_tutorial_src, gen_content_dir, execute = false, name = "mgvi_advanced_tutorial", documenter = true, credit = true)
Literate.script(advanced_tutorial_src, gen_content_dir, keep_comments = false, name = "mgvi_advanced_tutorial", documenter = true, credit = false)


makedocs(
    sitename = "MGVI",
    modules = [MGVI],
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
