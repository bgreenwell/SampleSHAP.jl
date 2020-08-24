using CSV, DataFrames, Plots, StatsBase, XGBoost

function genOMat(X::Matrix)

  O = falses(size(X, 1), size(X, 2) - 1)

  # Simulate the number of features that appear before the feature of interest
  # in each random permutation; note that each element can range from 0 to
  # (num_cols - 1) with equal probability
  nfeat = sample(1:size(X, 2), size(X, 1), replace=true) .- 1

  for i in 1:size(X, 1)
    if nfeat[i] == size(X, 2)
      O[i, :] .= true  # feature appears at end of random permutation
    elseif nfeat[i] > 0
      O[i, sample(1:(size(X, 2) - 1), nfeat[i], replace=false)] .= true
    end
  end

  return O

end


# Should be (roughly) uniform
proportions(sum(genOMat(rand(Float64, 100000, 20)), dims=2))

@benchmark genOMat(X) setup=(X=rand(Float64, 10000, 20))

import XGBoost
function explaincolumn(bst::Booster, X::Matrix, column::Int64)

  n, p = size(X)

  # if isnothing(newdata)   # FIXME: Should sampling be done with replacement?
      W = X[sample(1:n, n, replace=true), :]
      O = genOMat(X)
  # else
  #     W = X[sample(1:n, size(newdata, 1), replace=true), :]  # randomly sample rows from full X
  #     O = genOMat(X)[sample(1:n, size(newdata, 1), replace=true), :]
  #     X = newdata  # observations of interest
  # end

  # Finish building logical matrix that resembles the random permutation order
  # to use for each row of X and W (a TRUE indicates that the corresponding
  # feature appeared before the feature of interest in the associated
  # permutation)
  if (column == 1)  # case 1
    O = hcat(trues(size(O, 1), 1), O)
  elseif (column == p)  # case 2
    O = hcat(O, trues(size(O, 1), 1))
  else  # case 3
    O = hcat(O[:, 1:(column - 1)], trues(size(O, 1), 1), O[:, column:(p - 1)])
  end

  B = Dict("b1" => copy(X), "b2" => copy(X))
  B["b1"][O] = X[O]
  B["b1"][.!O] = W[.!O]
  OO = copy(O)
  OO[:, column] = falses(size(OO, 1), 1)
  B["b2"][OO] = X[OO]
  B["b2"][.!OO] = W[.!OO]

  # Return difference in predictions
  return XGBoost.predict(bst, B["b1"]) - XGBoost.predict(bst, B["b2"])

end


function explain(fit::Booster, X::Matrix, nsim::Int64=1)
  res1 = zeros(size(X, 1), nsim)
  res2 = zeros(size(X))
  for i in 1:size(X, 2)
    # println(i)
    for j in 1:nsim
      res1[:, j] = explaincolumn(bst, X, i)
    end
    res2[:, i] = mean(res1, dims=2)
  end
  res2
end


# boston = CSV.read("/Users/bgreenwell/Desktop/trees/data/boston.csv", DataFrame)
# X = boston[:, setdiff(names(boston), ["cmedv"])]
# xnames = names(X)
# X = convert(Matrix, X)
# y = convert(Vector, boston.cmedv)
#
# bst = xgboost(X, 200, label = y, eta = 1, max_depth = 6)
#
# About four times faster than fastshap::explain() in R
# shap = explain(bst, X, 1000)
# scatter(X[:, 8], shap[:, 8])
# scatter(X[:, 15], shap[:, 15])
