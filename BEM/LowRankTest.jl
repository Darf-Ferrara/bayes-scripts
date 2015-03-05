# Testing out LowRankModels in order to test



TrueA = repmat(randn(5,100),20,1); 			# Create a 100x100 matrix of rank 5, then add some noise

A = TrueA + 0.2 * randn(100,100);


u1,s1,v1 = svd(A)		# find svd of matrix and take the top 5 values. u1 * diagm(s1_trunkated) * v1' 
						# should be close to A
						
SVDError = maximum(abs(u1 * diagm([s1[1:5],zeros(95)]) * v1' - TrueA))

using LowRankModels
m,n,k = 100,100,100
losses = fill(quadratic(),n)
rx = onesparse() # each row is assigned to exactly one cluster
ry = zeroreg() # no regularization on the cluster centroids
glrm = GLRM(A,losses,rx,ry,k)

X,Y,ch = fit!(glrm)
