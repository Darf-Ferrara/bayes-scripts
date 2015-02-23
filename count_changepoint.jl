#This program fits a regression model with one
#unknown change point using generated data
using DataFrames
using Gadfly
using Distributions
dataFaked = false

function generateChangeData(ChangePointLengths,vecOfPoissonLambda)
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
	nobs = sum(ChangePointLengths)
	df = DataFrame(deaths = generateChangeData(ChangePointLengths,vecOfPoissonLambda))
	
else
	df = readtable("coaldata.txt")
	nobs = size(df)[1]
end
#Set conditions for experiment and generate the data
#Set up the prior values;
#NOTE: In this exercise, we are using the routine "gam_rnd", 
#which is parameterized differently than the routine 
#in the book. In particular, the second argument 
#of the density is the reciprocal of how it is 
#parameterized in our book. Thus, we change the 
#hyperparameters below to accomodate for this 
#fact, and alter the conditional distributions 
#accordingly. 
a1 = 1; a2 = 1;
d1 = 1; d2 = 1;


#Begin the Gibbs sampler
iter =100000;
burn = 2000;
lambda_grid = [1:1:nobs-1]';
lambda_date = [1851:1:1961]';

gamma_final = zeros(iter-burn,1);	#Initialize the chains
delta_final = zeros(iter-burn,1);
lambda_final = zeros(iter-burn,1);

lambda = 50;	#First guess is that it is at time 50


for ii = 1:iter
    #----------------------------------------------------
    #Sample gamma, the parameter of the "first" Poisson density
    #----------------------------------------------------
    y_gamma = df[:deaths][1:lambda];
    n_lambda = length(y_gamma);
    gammas = rand(Gamma((a1 + sum(y_gamma)),1/(a2 + n_lambda)))
    #------------------------------------------
    #Sample delta, the parameter of the "second" Poisson density
    #----------------------------------------
    y_delta = df[:deaths][lambda+1:nobs];
    deltas = rand(Gamma((d1 + sum(y_delta)),1/(d2 + nobs-n_lambda)));
    
    #-----------------------------------
    #Sample the changepoint lambda
    #-----------------------------------
    
    log_dens_unnorm = zeros(nobs-1,1);
    for jj = 1:(nobs-1); #loop over the number of discrete values for Lambda
       ypart1 = df[:deaths][1:jj];
       ypart2 = df[:deaths][(jj+1):nobs];
       n_temp = length(ypart1);
       log_dens_unnorm[jj] = (sum(ypart1)*log(gammas) - n_temp*gammas + sum(ypart2)*log(deltas) - (nobs-n_temp)*deltas)[1];
       #dens_unnorm(jj,1) = (gammas^(sum(ypart1)))*exp(-n_temp*gammas)*(deltas^(sum(ypart2)))*exp(-(nobs-n_temp)*deltas);
    end;
    d = maximum(log_dens_unnorm);
    log_dens_unnorm = log_dens_unnorm -d;
    dens_unnorm = exp(log_dens_unnorm);
    dens_norm = dens_unnorm/sum(dens_unnorm);
    lambda = rand(Categorical(vec(dens_norm)));
    lambda_keep = lambda_date[lambda];
    
    if ii > burn;
        gamma_final[ii-burn] = gammas;
        delta_final[ii-burn] = deltas;
        lambda_final[ii-burn] = lambda_keep;
    end;
end;

plot(x=gamma_final,Geom.histogram)
plot(x=delta_final,Geom.histogram)
plot(x=lambda_final,Geom.histogram)


#disp('Post means and std deviatiations');
#disp('For betas, thetas, sig, tau and lambda');
#[mean(gamma_final')' std(gamma_final')']
#[mean(delta_final')' std(delta_final')']
#[mean(lambda_final) std(lambda_final)]
