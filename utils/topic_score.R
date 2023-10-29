## Ke (2017) SVD approach
library(TopicScore)
library(Matrix)
library(MatrixExtra)
library(rARPACK)
library(plotly)
library(ggplot2)
library(dplyr)
library(MASS)

root_path = "/Users/jeong-yeojin/Desktop/testLDA/data/spleen"
setwd(root_path)
spleen = read.csv("BALBc-3_D_unnorm.csv", row.names = 1)
docnames = rownames(spleen)
wordnames = colnames(spleen)

K = 10
K0 = 1.5*K
m = 2*K # m < p
D = t(as.matrix(spleen))
D = D/colSums(D)

score <- function(K, K0, m, D, scatterplot=FALSE){
  p <- dim(D)[1]
  obj <- svds(D,K)
  Xi <- obj$u
  
  #Step 1
  Xi[,1] <- abs(Xi[,1])
  R <- apply(as.matrix(Xi[,2:K]),2,function(x) x/Xi[,1])
  
  #Step 2
  vertices_est_obj <- vertices_est(R,K0,m, num_restart = 1)
  V <- vertices_est_obj$V
  theta <- vertices_est_obj$theta
  
  if (scatterplot){
    par(mar=c(1,1,1,1))
    plot(R[,1],R[,2])
    points(V[,1],V[,2],col=2,lwd=5)
  }
  
  #Step 3
  Pi <- cbind(R, rep(1,p))%*%solve(cbind(V,rep(1,K)))
  Pi <- pmax(Pi,matrix(0,dim(Pi)[1],dim(Pi)[2]))
  temp <- rowSums(Pi)
  Pi <- apply(Pi,2,function(x) x/temp)
  
  #Step 4
  A_hat <- Xi[,1]*Pi
  
  #Step 5
  temp <- colSums(A_hat)
  A_hat <- t(apply(A_hat,1,function(x) x/temp))
  
  return(list(A_hat=A_hat, R=R,V=V, Pi=Pi, theta=theta))
}
tscore_obj = score(K, K0, m, D, scatterplot = FALSE)

R.data <- as.data.frame(cbind(tscore_obj$R[,1:2], rep("words",dim(tscore_obj$R)[1])))
V.data <- as.data.frame(cbind(tscore_obj$V[,1:2], rep("vertex",dim(tscore_obj$V)[1])))
data = rbind(R.data, V.data)
colnames(data) = c("x","y","type"); rownames(data) = c(wordnames, paste0("V",1:K))
data$x = as.numeric(data$x)
data$y = as.numeric(data$y)

plot_ly(data = data, x = ~x, y = ~y, symbol = ~type,
        marker = list(size = 15)) %>% 
  add_trace(
    type = 'scatter',
    mode = 'markers',
    text = rownames(data), # when you hover on a point it will show it's rowname
    hoverinfo = 'text',
    showlegend = T
  )

Ahat = tscore_obj$A_hat
colnames(Ahat) = paste0("T",1:K)
write.matrix(format(Ahat,scientific=FALSE),file="BALBc-3_Ahat_10.csv",sep = ",")
