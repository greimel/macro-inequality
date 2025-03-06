Dict(
    :main => [
        "welcome" => collections["welcome"].pages,
        "Julia basics" => collections["julia-basics"].pages,
        #"Preliminaries" => collections["preliminaries"].pages,
        #"Housing" => collections["housing"].pages,
        #"Long-run" => collections["long-run"].pages,
        #"Continuous time" => collections["continuous-time"].pages,
        #"Assignments and Tutorials" => [collections["assignments"].pages; collections["solutions-week1"].pages],
        #"Unfinished notebooks" => collections["unfinished"].pages,
    ],
    :about => Dict(
        :authors => [
            (name = "Fabian Greimel", url = "https://www.greimel.eu")
        ],
        :title => "Macroeconomics and Inequality",
        :subtitle => "Master-level Elective Course",
        :term => "Summer 2025",
        :institution => "University of Vienna",
        :institution_url => "https://econ.univie.ac.at",
        :institution_logo => "tinbergen-institute-logo.svg",
        :institution_logo_darkmode => "tinbergen-logo-white.svg"
    )
)
