module SampleSHAP

import Statistics
import StatsBase
import XGBoost

"""
    genOMat(X::Matrix)

Return TBD.
"""
function genOMat(X::Matrix)

  O = falses(size(X, 1), size(X, 2) - 1)

  # Simulate the number of features that appear before the feature of interest
  # in each random permutation; note that each element can range from 0 to
  # (num_cols - 1) with equal probability
  nfeat = StatsBase.sample(1:size(X, 2), size(X, 1), replace=true) .- 1

  for i in 1:size(X, 1)
    if nfeat[i] == size(X, 2)
      O[i, :] .= true  # feature appears at end of random permutation
    elseif nfeat[i] > 0
      O[i, StatsBase.sample(1:(size(X, 2) - 1), nfeat[i], replace=false)] .= true
    end
  end

  return O

end


"""
    explaincolumn(bst::XGBoost.Booster, X::Matrix, column::Int64)

Return TBD.
"""
function explaincolumn(bst::XGBoost.Booster, X::Matrix, column::Int64)

  n, p = size(X)

  W = X[StatsBase.sample(1:n, n, replace=true), :]
  O = genOMat(X)

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


"""
    explaincolumn(bst::XGBoost.Booster, X::Matrix, column::Int64, newdata::Matrix)

Return TBD.
"""
function explaincolumn(bst::XGBoost.Booster, X::Matrix, column::Int64, newdata::Matrix)

  n, p = size(X)

  if isnothing(newdata)   # FIXME: Should sampling be done with replacement?
      W = X[StatsBase.sample(1:n, n, replace=true), :]
      O = genOMat(X)
  else
      W = X[sample(1:n, size(newdata, 1), replace=true), :]  # randomly sample rows from full X
      O = genOMat(X)[sample(1:n, size(newdata, 1), replace=true), :]
      X = newdata  # observations of interest
  end

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


"""
    explain(bst::XGBoost.Booster, X::Matrix, nsim::Int64)

Return TBD.
"""
function explain(bst::XGBoost.Booster, X::Matrix, nsim::Int64=1)
  res1 = zeros(size(X, 1), nsim)
  res2 = zeros(size(X))
  for i in 1:size(X, 2)
    # println(i)
    for j in 1:nsim
      res1[:, j] = explaincolumn(bst, X, i)
    end
    res2[:, i] = Statistics.mean(res1, dims=2)
  end
  res2
end


"""
    explain(bst::XGBoost.Booster, X::Matrix, newdata::Matrix, nsim::Int64)

Return TBD.
"""
function explain(bst::XGBoost.Booster, X::Matrix, newdata::Matrix, nsim::Int64=1)
  res1 = zeros(size(X, 1), nsim)
  res2 = zeros(size(X))
  for i in 1:size(X, 2)
    # println(i)
    for j in 1:nsim
      res1[:, j] = explaincolumn(bst, X, i, newdata)
    end
    res2[:, i] = Statistics.mean(res1, dims=2)
  end
  res2
end

end # module
