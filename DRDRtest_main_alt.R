# Set a different library location
lib_location <- "/home/ynychen/.local/lib"
# Set the new library location
.libPaths(lib_location)

# Set the CRAN mirror
cran_mirror <- "https://cran.r-project.org"


if (require(DRDRtest) == F){
  install.packages("DRDRtest", lib = lib_location, repos = cran_mirror)
}

if (require(SuperLearner) == F){
  install.packages("SuperLearner", lib = lib_location, repos = cran_mirror)
}

if (require(earth) == F){
  install.packages("earth", lib = lib_location, repos = cran_mirror)
}

#if (require(randomForest) == F){
#  install.packages("randomForest", lib = lib_location, repos = cran_mirror)
#}

library(DRDRtest, lib = lib_location)
library(SuperLearner, lib = lib_location)
library(earth, lib = lib_location)
#library(randomForest, lib = lib_location)

num = 100
delta_list = c(0, 0.5) #seq(0, 0.5, 0.1)

save_dir = 'R/logs/simu2/eval/'
if (dir.exists(substr(save_dir, 1, 1)) == F){
    dir.create(substr(save_dir, 1, 1))
}
if (dir.exists(substr(save_dir, 1, 6)) == F){
    dir.create(substr(save_dir, 1, 6))
}
if (dir.exists(substr(save_dir, 1, 12)) == F){
    dir.create(substr(save_dir, 1, 12))
}
if (dir.exists(save_dir) == F){
    dir.create(save_dir)
}

triangle <- function(a,delta){
    y <- exp(-100 * (a-0.5)^2)*delta
    return(y)
}

mu.mod <- function(a,l,delta){
    mu <- as.numeric(l%*%c(0.2,0.2,0.3,-0.1))+triangle(a, delta)+a*(-0.1*l[,1]+0.1*l[,3])
    return(mu)
}

pifunc <- function(a,l){
       l <- as.matrix(l)
       logit.lambda <- as.numeric(l%*%c(0.1,0.1,-0.1,0.2))
       lambda <- exp(logit.lambda)/(1+exp(logit.lambda))
       return(dbeta(a,shape1=lambda,shape2 = 1-lambda))
}

mufunc <- function(a,l){
   l <- as.matrix(l)
   return(mu.mod(a,l,delta))
}

for (delta in delta_list){
    p_val = rep(0, num)
    p_val2 = rep(0, num)
    time_cost = rep(0, num)
    time_cost2 = rep(0, num)

    for (i in 0:(num-1)){
      load_dir = paste('/dataset/simu2/eval/', toString(i), '/', sep = '')
      dat = read.csv(paste(getwd(), load_dir, 'delta_', toString(delta), '_data.txt', sep = ''), header = F, sep = ' ')

      y = dat[[6]]
      a = dat[[1]]
      l = dat[2:5]
      start_time <- Sys.time()
      out <- drdrtest(y, a, l, c(0.01,0.99), pifunc, mufunc, b = 200)
      end_time <- Sys.time()
      time_cost[i] = end_time - start_time
      p_val[i] = out$p.value

      # default algs: "SL.earth", "SL.glm", "SL.gam", "SL.glmnet"
      alg_list = c("SL.earth", "SL.glm", "SL.gam") # "SL.randomForest")
      start_time <- Sys.time()
      out <- drdrtest.superlearner(y, a, l, c(0.01,0.99), pi.sl.lib = alg_list, mu.sl.lib = alg_list, b = 200)
      end_time <- Sys.time()
      time_cost2[i] = end_time - start_time
      p_val2[i] = out$p.value
      }

    write.table(p_val, file=paste(save_dir, "p_val_oracal_delta_", delta , '.txt', sep = ''), row.names=FALSE, col.names=FALSE)
    write.table(p_val2, file=paste(save_dir, "p_val_SuperLearner_delta_", delta , '.txt', sep = ''), row.names=FALSE, col.names=FALSE)
    write.table(time_cost, file=paste(save_dir, "run_time_oracal_delta_", delta , '.txt', sep = ''), row.names=FALSE, col.names=FALSE)
    write.table(time_cost2, file=paste(save_dir, "run_time_SuperLearner_delta_", delta , '.txt', sep = ''), row.names=FALSE, col.names=FALSE)
}


alpha = 0.05
rej_rate = rep(0, length(delta_list))
run_time = rep(0, length(delta_list))
rej_rate2 = rep(0, length(delta_list))
run_time2 = rep(0, length(delta_list))

idx = 1
for (delta in delta_list){
    out = read.csv(paste(save_dir, "p_val_oracal_delta_", delta , '.txt', sep = ''), header  = F, sep = ' ')
    out2 = read.csv(paste(save_dir, "p_val_SuperLearner_delta_", delta , '.txt', sep = ''), header  = F, sep = ' ')
    rej_rate[idx] = mean(out < alpha)
    rej_rate2[idx] = mean(out2 < alpha)

    time_cost = read.csv(paste(save_dir, "run_time_oracal_delta_", delta , '.txt', sep = ''), header  = F, sep = ' ')
    time_cost2 = read.csv(paste(save_dir, "run_time_SuperLearner_delta_", delta , '.txt', sep = ''), header  = F, sep = ' ')
    run_time[idx] = mean(time_cost[[1]])
    run_time2[idx] = mean(time_cost2[[1]])
    idx = idx + 1
}
write.table(rbind(delta_list, rej_rate, run_time), file=paste(save_dir, 'oracal_rej_rate.txt', sep = ''), row.names=FALSE, col.names=FALSE)
write.table(rbind(delta_list, rej_rate2, run_time2), file=paste(save_dir, 'SuperLearner_rej_rate.txt', sep = ''), row.names=FALSE, col.names=FALSE)
