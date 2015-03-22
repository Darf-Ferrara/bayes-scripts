#This program fits a regression model with one
#unknown change point using generated data
using DataFrames
using Gadfly
using Distributions
using Stan

dataFaked = false

function generateChangeData(ChangePointLengths,vecOfPoissonLambda) # Create a vector of poisson data of length = sum(ChangePointLengths)
	if length(ChangePointLengths) != length(vecOfPoissonLambda)
		error("length(ChangePointLengths) != length(vecOfPoissonLambda)")
	else
		asdf = [rand(Poisson(vecOfPoissonLambda[ii] ),ChangePointLengths[ii]) for ii = [1:length(vecOfPoissonLambda)]]
		reduce(vcat,asdf)
	end	
end


if dataFaked
	ChangePointLengths = [20, 70]
	vecOfPoissonLambda = [1.2, 3.2]
	numobs = sum(ChangePointLengths)
	df = DataFrame(deaths = generateChangeData(ChangePointLengths,vecOfPoissonLambda))
	
else
	data = int(readdlm("coaldata.txt")[2:end])
	numobs = length(data)
end



const changePointModel = "
data {
	real<lower=0> r_e;
	real<lower=0> r_l;
	int<lower=1> T;
	int<lower=0> D[T];
}
transformed data {
	real log_unif;
	log_unif <- -log(T);
}
parameters {
	real<lower=0> e;
	real<lower=0> l;
}
transformed parameters {
	vector[T] lp;
	lp <- rep_vector(log_unif, T);
	for (s in 1:T)
	for (t in 1:T)
	lp[s] <- lp[s] + poisson_log(D[t], if_else(t < s, e, l));
}
model {
	e ~ exponential(r_e);
	l ~ exponential(r_l);
	increment_log_prob(log_sum_exp(lp));
}
"

stanmodel = Stanmodel(name="changePoint", model=changePointModel);
stanmodel |> display

const changedata = [
	@Compat.Dict("r_e" => 1111.1,"r_l" = 11110.2,"T" => numobs, "D" => data),
	@Compat.Dict("r_e" => 1111.1,"r_l" = 11110.2,"T" => numobs, "D" => data),
	@Compat.Dict("r_e" => 1111.1,"r_l" = 11110.2,"T" => numobs, "D" => data),
	@Compat.Dict("r_e" => 1111.1,"r_l" = 11110.2,"T" => numobs, "D" => data)
]


println("Input observed data, an array of dictionaries:")
changePointModel |> display
println()

## Simulate 
sim1 = stan(stanmodel, changedata, CmdStanDir="/home/pcurry/Documents/cmdstan-2.6.2")
describe(sim1)

