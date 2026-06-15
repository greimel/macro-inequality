using Pkg

Pkg.activate(@__DIR__)

using Weave

const EXERCISE_DIR = @__DIR__
const GENERATED_DIR = joinpath(EXERCISE_DIR, "generated")
const COLLECTION_PATH = joinpath(EXERCISE_DIR, "collection.tex")

function exercise_sources()
    sort(filter(path -> endswith(path, ".texw"), readdir(EXERCISE_DIR; join = true)))
end

function weave_exercise(path)
    woven_path = weave(path; informat = "noweb", doctype = "pandoc", out_path = GENERATED_DIR)
    target_path = joinpath(GENERATED_DIR, splitext(basename(path))[1] * ".tex")
    mv(woven_path, target_path; force = true)
    return target_path
end

function _compile_collection(; show_solutions::Bool)
    isfile(COLLECTION_PATH) || error("missing collection file at $COLLECTION_PATH")
    src = read(COLLECTION_PATH, String)

    suffix = show_solutions ? "solutions" : "exercises"
    toggle = show_solutions ? "\\showsolutionstrue" : "\\showsolutionsfalse"
    src = replace(src, r"\\showsolutions(true|false)" => toggle)

    tex_name = "collection-$(suffix).tex"
    write(joinpath(GENERATED_DIR, tex_name), src)

    cd(GENERATED_DIR) do
        run(`latexmk -pdf -shell-escape -interaction=nonstopmode -halt-on-error $tex_name`)
    end

    return joinpath(GENERATED_DIR, "collection-$(suffix).pdf")
end

function build_exercises(; compile_pdf = false)
    sources = exercise_sources()
    isempty(sources) && error("no .texw files found in $EXERCISE_DIR")

    mkpath(GENERATED_DIR)
    outputs = [weave_exercise(source) for source in sources]

    if compile_pdf
        pdf_solutions = _compile_collection(; show_solutions = true)
        pdf_exercises = _compile_collection(; show_solutions = false)
        return (; outputs, pdf_solutions, pdf_exercises)
    end

    return (; outputs)
end

compile_pdf = "--pdf" in ARGS
result = build_exercises(; compile_pdf)

println("Wove $(length(result.outputs)) exercise(s) into $(GENERATED_DIR)")
if compile_pdf
    println("PDFs available at:")
    println("  $(result.pdf_solutions)")
    println("  $(result.pdf_exercises)")
end
