#
#
# Adapted to from matlab code from BEM
using DataFrames
using Distributions
using Gadfly

df1 = DataFrame(readdlm("tobit_data.txt"))
names!(df1,[:weeks,:D,:D2,:AFQT,:Spouse,:Kids,:Ed])

#nobs = length(weeks);
#const = ones(nobs,1);
#y = weeks;
#x = [const tobit_data(:,4:7)];
#k = size(x,2);
#set prior values 

numObs = size(df1,1)
k = 5 # number of prediction variables + the constant
mu_beta = zeros(k,1);
var_beta = 10^2*eye(k);
invarbeta = inv(var_beta);

a = 3;
b = 1/(2*20);
#initialize vectors and begin the Gibbs sampler
iter  = 5500;
burn = 500;
beta_final = zeros(iter-burn,k);
sig_final = zeros(iter-burn,1);

temp1 = array(df1)
D1 = temp1[:,2]
D2 = temp1[:,3]
y = weeks = temp1[:,1]
x = ([ones(size(temp1,1)) temp1[:,4:7]])
bhat = (x' * x)\ (x' * temp1[:,1])

betaParam = bhat;
sig = 500;
for ii in 1:iter
    #augmented latent data
    temp2 = x*betaParam
    ztemp_aug1 = [rand(Truncated(Normal(jj,sig),-1000,0)) for jj in temp2]
	#for those with y=0 (D=1), draw latent data to be truncated from above at zero
    #for now, do this for everyone.
	
    ztemp_aug2 = [rand(Truncated(Normal(jj,sig),52,1000)) for jj in temp2]  #for those with y=52 (D2=1), draw latent data to be truncated from below at 52
                                                                            #for now, do this for everyone. 
                                                                            
    zaug = D1.*ztemp_aug1 + D2.*ztemp_aug2 + (1-D1- D2).*y; #Note that D1,D2 are exclusive!
    
    #regression paramters
    D_beta = inv(x'*x/sig + invarbeta);
    d_beta = x'*zaug/sig + invarbeta*mu_beta;
    H = chol(D_beta);
    
    betaParam = D_beta*d_beta + H'*randn(k,1);
    
    #variance parameter
    
    sig = rand(InverseGamma( (numObs/2) + a, inv(b) + .5*sum( (zaug - x*betaParam).^2 ) ));
    
    if ii > burn
        beta_final[ii-burn,:] = betaParam';
        sig_final[ii-burn,:] = sig;
    end
end


weeksX,weeksDist  = hist(beta_final[:,1],50)
AFQTX,AFQTDist  = hist(beta_final[:,2],50)
SpouseX,SpouseDist  = hist(beta_final[:,3],50)
KidsX,KidsDist  = hist(beta_final[:,4],50)
EdX,EdDist  = hist(beta_final[:,5],50)

Gadfly.plot(x = weeksX,y = weeksDist,x = AFQTX,y = AFQTDist)

beta_final = convert(DataFrame,beta_final)
names!(beta_final,[:weeks,:AFQT,:Spouse,:Kids,:Ed])
Gadfly.plot( x = beta_final[:weeks], Geom.histogram)


print("Means and Standard Deviations of parameters");
print("const afqt spouse_inc kids ed, STD_DEV");
print([mean(beta_final) mean(sqrt(sig_final))])
print([std(beta_final) std(sqrt(sig_final))])

print('Marginal Effects');
print('const afqt spouse_inc kids ed, variance');
mu_x = mean(x);
probs = normcdf ( (52 - mu_x*beta_final')'./(sqrt(sig_final))) - normcdf((-mu_x*beta_final')'./(sqrt(sig_final)));
[mean(beta_final(:,2).*probs) mean(beta_final(:,3).*probs) mean(beta_final(:,4).*probs) mean(beta_final(:,5).*probs) ]
[std(beta_final(:,2).*probs) std(beta_final(:,3).*probs) std(beta_final(:,4).*probs) std(beta_final(:,5).*probs) ]
