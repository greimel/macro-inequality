function aggregate(sim_df, pmf)
    return @chain sim_df begin
        @select(:a, :a_next, :j)
        stack(Not(:j))
        leftjoin(DataFrame(pmf), on = :j)
        disallowmissing!
        @groupby(:variable)
        @combine(:value = dot(:value, :pmf))
        (; (Symbol.(_.variable) .=> _.value)...)
    end
end
