library(ggplot2)
library(viridis)
library(dirmult)
library(rARPACK)
library(clustRviz)
library(tilting)
library(gridExtra)
library(lpSolve)

##### Generate synthetic data #####
initial_centers = function(val,centers){
  quantiles = c()
  for(i in 1:centers){
    quantiles = c(quantiles, i*as.integer(nrow(val)/centers))
  }
  return(quantiles)
}

reorder_row_with_noise = function(row, order, r, K) {
  u = runif(1)
  # noise
  if(u<r){
    row[order[sample(1:K)]]
  }else{
    # not noise
    sorted_row = sort(row, decreasing = TRUE)
    sorted_row[order]
  }
}

generate_data = function(N, n, p, K, r){
  set.seed(127)
  N = N; n = n; p = p; K=K; r=r
  # Generate a network
  ## set initial 20 cluster centers
  df = as.data.frame(matrix(nrow=n,ncol=2))
  colnames(df) = c("lat","long")
  df$lat = runif(n, min=-1, max=1)
  df$long = runif(n, min=-3, max=3)
  cluster_obj = kmeans(df, centers = df[initial_centers(df, 20),])
  df$grp = cluster_obj$cluster
  df$grp = (df$grp %% K + 1)
  
  # Simulate W and A, D
  # W (k x n)
  sample_mn = function(p, N){
    rmultinom(1, N, p)
  }
  W = matrix(0,K,n)
  for(k in 1:K){
    cluster.size = dim(df[df$grp==k,])[1]
    #alpha = runif(K,0.1,0.3)
    alpha = c(0.1, 0.15, 0.2)
    # order rows of W so each doc matches cluster assignment
    order = rep(0,K)
    order[(1:K)[-k]] = sample(2:K, K-1)
    order[k] = 1
    inds = (df$grp==k)
    W[,inds] = apply(rdirichlet(cluster.size,alpha), 1, 
                     reorder_row_with_noise, order, r, K)
    # pure doc
    cano.ind = sample((1:n)[inds],1)
    W[,cano.ind] = diag(K)[k,]
  }
  # A (p x k)
  A = matrix(runif(p*K,0,1),p,K)
  # pure word
  cano.ind = sample((1:p),K)
  A[cano.ind,] = diag(K)
  A = apply(A,2,function(x) x/sum(x))
  # D (p x n)
  D0 = A%*%W 
  D = apply(D0, 2, sample_mn, N)/N
  df$grp_new = apply(W, 2, which.max)
  
  return(list(W=W, A=A, D=D, D0=D0, df=df))
}

# SPOC algorithm
## M = t(D) n x k
get_weights = function(df, m=5, phi=0.1){
  weight_func  = 
    sparse_rbf_kernel_weights(k = m, phi = phi, 
                              dist.method = "euclidean", 
                              p = 2)
  w = weight_func(df[,1:2]) # lon lat
  return(w)
}

get_U_tilde = function(D, K, w, eps=0.001){
  # Initialize V0
  s = svds(D,K)
  V_tilde = s$v
  U_tilde = s$u
  UUT_old = U_tilde%*%t(U_tilde)
  VVT_old = V_tilde%*%t(V_tilde)
  niter = 1
  thres = 1
  while (thres>eps) {
    # Left update
    M = D %*% V_tilde
    carp_fit = CARP(M, weights = w$weight_mat, 
                    t = 1.1,
                    X.center = FALSE, X.scale = FALSE)
    runs = dim(carp_fit$U)[3]
    U_cvx = carp_fit$U[,,runs*0.1]
    QR = qr(U_cvx)
    U_tilde = qr.Q(QR)
    # Right update
    V_hat = t(D)%*%U_tilde
    QR = qr(V_hat)
    V_tilde = qr.Q(QR)
    # Stopping rule
    UUT = U_tilde%*%t(U_tilde)
    VVT = V_tilde%*%t(V_tilde)
    thres = max(norm(UUT-UUT_old, type = "F")^2, norm(VVT-VVT_old, type = "F")^2)
    UUT_old = UUT; VVT_old = VVT
    print(thres)
    niter= niter+1
  }
  return(U_tilde)
}
  
preprocess_U = function(U, K){
  for(k in 1:K){
    if(U[1,k] < 0){
      U[,k] = -1*U[,k]
    }
  }
  return(U)
}

proj_simplex = function(v){
  n = length(v)
  if (sum(v) == 1 && all(v >= 0)) {
    return(v)
  }
  u = rev(sort(v))
  rho = max(which(u*(1:n) > (cumsum(u) - 1)))
  theta = (cumsum(u) - 1) / rho
  w = pmax(v - theta, 0)
  return(w)
}

get_component_mapping = function(stats_1, stats_2) {
  similarity <- stats_1 %*% t(stats_2)
  cost_matrix <- -similarity
  assignment <- lp.assign(cost_matrix)
  P = assignment$solution
  return(P)
}

fit_SPOC = function(df, D, W, U, K, w, method = "spatial"){
  if(method!="spatial"){
    print("vanilla")
    svd = svds(D,K) 
    U = svd$u
  }
  J = c()
  S = preprocess_U(t(U), K) # K x n
  # Run SPA
  for(t in 1:K){
    maxind = which.max(col.norm(S))
    s = matrix(S[,maxind], nrow=K)
    S1 = (diag(K)-S[,maxind]%o%S[,maxind]/norm(s)^2)%*%S
    S = S1
    J[t] = maxind
  }
  # Get \hat{H} and \hat{H}
  H_hat = U[J,]
  W_hat = U %*% solve(H_hat, tol = 1e-07)
  # Postprocess
  #W_hat = apply(W_hat, 1, proj_simplex)
  P = get_component_mapping(t(W_hat), t(W))
  W_hat = W_hat%*%P
  # Results
  n = dim(W_hat)[1]
  clt = apply(W_hat, 1, which.max)
  accuracy = sum(clt == df$grp_new)/n
  print(accuracy)
  err = norm(W-W_hat, type="F")
  print(err)
  return(list(acc = accuracy, f.err=err, What = W_hat))
}

# RUN
N = 20
n = 1000
p = 30
K = 3
r = 0.05
topic.data = generate_data(N=N, n=n, p=p, K=K, r=r)
w = get_weights(topic.data$df, m=5)
df = topic.data$df
D = topic.data$D
W = topic.data$W
U_tilde = get_U_tilde(t(D),K,w=w)  
res_b = fit_SPOC(df, t(D), t(W), U_tilde, K=K, w=w, method = "base")
res_s = fit_SPOC(df, t(D), t(W), U_tilde, K=K, w=w, method = "spatial")


# Plot
# Plot generated data
## Before adding noise
p_n = ggplot(topic.data$df)+
  geom_point(mapping = aes(x=lat, y=long, color=grp))+
  xlim(-1,1)+
  ylim(-3,3)+
  scale_color_viridis(discrete = FALSE, option = "D")+
  scale_fill_viridis(discrete = FALSE)
## After adding noise
p_0 = ggplot(topic.data$df)+
  geom_point(mapping = aes(x=lat, y=long, color=grp_new))+
  xlim(-1,1)+
  ylim(-3,3)+
  scale_color_viridis(discrete = FALSE, option = "D")+
  scale_fill_viridis(discrete = FALSE)

# Compare W hat
plot_W1 = function(df){
  p = 
    ggplot(df)+
    geom_point(mapping = aes(x=lat, y=long, color=W1), show.legend = FALSE)+
    xlim(-1,1)+
    ylim(-3,3)+
    scale_color_viridis(discrete = FALSE, option = "D")+
    scale_fill_viridis(discrete = FALSE)+
    theme(legend.position = "none")+
    theme_bw()
  return(p)
}
plot_W2 = function(df){
  p = 
    ggplot(df)+
    geom_point(mapping = aes(x=lat, y=long, color=W2), show.legend = FALSE)+
    xlim(-1,1)+
    ylim(-3,3)+
    scale_color_viridis(discrete = FALSE, option = "D")+
    scale_fill_viridis(discrete = FALSE)+
    theme(legend.position = "none")+
    theme_bw()
  return(p)
}
plot_W3 = function(df){
  p = 
    ggplot(df)+
    geom_point(mapping = aes(x=lat, y=long, color=W3), show.legend = FALSE)+
    xlim(-1,1)+
    ylim(-3,3)+
    scale_color_viridis(discrete = FALSE, option = "D")+
    scale_fill_viridis(discrete = FALSE)+
    theme(legend.position = "none")+
    theme_bw()
  return(p)
}
W_hat_b = res_b$What
df_new = cbind(df, W_hat_b)
colnames(df_new) = c("lat", "long", "grp","grp_new","W1","W2","W3")
p1 = plot_W1(df_new)
p2 = plot_W2(df_new)
p3 = plot_W3(df_new)

W_hat = res_s$What
df_new = cbind(df, W_hat)
colnames(df_new) = c("lat", "long", "grp","grp_new","W1","W2","W3")
p4 = plot_W1(df_new)
p5 = plot_W2(df_new)
p6 = plot_W3(df_new)

# Original W
df_new = cbind(df, t(W))
colnames(df_new) = c("lat", "long", "grp","grp_new","W1","W2","W3")
p7 = plot_W1(df_new)
p8 = plot_W2(df_new)
p9 = plot_W3(df_new)
main = paste0("N=",N," ","n=",n," ","p=",p," ","K=",K,"r=",r)
sub = paste0("Accuracy/Error:",res_b$acc,",",round(res_b$f.err,2),"(base)"," / ",
             res_s$acc,",",round(res_s$f.err,3),"(spatial)")
grid.arrange(p7,p8,p9,p1,p2,p3,p4,p5,p6, ncol=3,nrow=3,
             top = main, bottom = sub)

# Plot Simplex
plot_simplx = function(df){
  p = 
    ggplot(df)+
    geom_point(mapping = aes(x=lat, y=lon, color=grp), show.legend = FALSE)+
    scale_color_viridis(discrete = FALSE, option = "D")+
    scale_fill_viridis(discrete = FALSE)+
    theme(legend.position = "none")+
    #xlim(-0.0355,-0.025)+
    #ylim(-0.1,0.055)+
    theme_bw()
  return(p)}
simplx = as.data.frame(cbind(df$grp_new, 
                             U_tilde[,1], U_tilde[,2]))
colnames(simplx) = c("grp","lat","lon")
s1 = plot_simplx(simplx)

svd = svds(t(D),K) 
U = svd$u
simplx = as.data.frame(cbind(df$grp_new, U[,1], U[,2]))
colnames(simplx) = c("grp","lat","lon")
s2 = plot_simplx(simplx)

D0 = topic.data$D0
svd = svds(t(D0),K) 
U = svd$u
simplx = as.data.frame(cbind(df$grp_new, U[,1], U[,2]))
colnames(simplx) = c("grp","lat","lon")
s3 = plot_simplx(simplx)

simplx = as.data.frame(cbind(df$grp_new, W[1,], W[2,]))
colnames(simplx) = c("grp","lat","lon")
s4 = plot_simplx(simplx)

main = "U_SpLSI / U_pLSI / U_oracle"
grid.arrange(s1,s2,s3, ncol=2, top = main)


# Miscell
s = svds(D,K)
U0 = s$u
U = U0
for(i in 1:10){
  m = (U[i,]+U[10+i,]+U[20+i,])/3
  U[i,] = m + rnorm(4,0,0.01)
  U[10+i,] = m + rnorm(4,0,0.01)
  U[20+i,] = m + rnorm(4,0,0.01)
}

heatmap(U0, Colv = NA, Rowv = NA)
heatmap(U, Colv = NA, Rowv = NA)
QR = qr(U)
Q = qr.Q(QR)
heatmap(Q, Colv = NA, Rowv = NA)
order = c()
for(i in 1:10){
  order = c(order, c(i, 10+i, 20+i))
}
heatmap(Q[order,], Colv = NA, Rowv = NA)

heatmap(s$u, Colv = NA, Rowv = NA)
heatmap(U_tilde, Colv = NA, Rowv = NA)
M = w$weight_mat
cent = 300
grp = (1:n)[M[cent,]>0]
U_tilde[c(cent,grp),]
s$u[c(cent,grp),]
res = t(U_tilde)%*%t(D)%*%(V_tilde)
res

