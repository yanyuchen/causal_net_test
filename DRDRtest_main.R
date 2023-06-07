if (require(DRDRtest) == F){
  install.packages("DRDRtest")
}

if (require(SuperLearner) == F){
  install.packages("SuperLearner")
}

if (require(earth) == F){
  install.packages("earth")
}

library(parallel)

num = 1000
delta_list = seq(0, 0.5, 0.1)
cl <- makeCluster(detectCores())

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

for (delta in delta_list){
    f = function(i){
        library(DRDRtest)
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

        load_dir = paste('/dataset/simu2/eval/', toString(i), '/', sep = '')
        dat = read.csv(paste(getwd(), load_dir, 'delta_', toString(delta), '_data.txt', sep = ''), header = F, sep = ' ')
        y = dat[[6]]
        a = dat[[1]]
        l = dat[2:5]
        out <- drdrtest(y, a, l, c(0.01,0.99), pifunc, mufunc)
        return(out$p.value)
    }

    f2 = function(i){
        library(DRDRtest)
        library(SuperLearner)
        library(earth)
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

        load_dir = paste('/dataset/simu2/eval/', toString(i), '/', sep = '')
        dat = read.csv(paste(getwd(), load_dir, 'delta_', toString(delta), '_data.txt', sep = ''), header = F, sep = ' ')
        y = dat[[6]]
        a = dat[[1]]
        l = dat[2:5]
        # default algs: "SL.earth", "SL.glm", "SL.gam", "SL.glmnet"
        out <- drdrtest.superlearner(y, a, l, c(0.01,0.99))
        return(out$p.value)
    }
    clusterExport(cl, "delta")
    out = parLapply(cl, 0:(num-1), f)
    out2 = parLapply(cl, 0:(num-1), f2)
    write.table(out, file=paste(save_dir, "p_val_oracal_delta_", delta , '.txt', sep = ''), row.names=FALSE, col.names=FALSE)
    write.table(out2, file=paste(save_dir, "p_val_SuperLearner_delta_", delta , '.txt', sep = ''), row.names=FALSE, col.names=FALSE)
}


alpha = 0.05
rej_rate = rep(0, length(delta_list))
rej_rate2 = rep(0, length(delta_list))

idx = 1
for (delta in delta_list){
    out = read.csv(paste(save_dir, "p_val_oracal_delta_", delta , '.txt', sep = ''), header  = F, sep = ' ')
    out2 = read.csv(paste(save_dir, "p_val_SuperLearner_delta_", delta , '.txt', sep = ''), header  = F, sep = ' ')
    rej_rate[idx] = mean(out < alpha)
    rej_rate2[idx] = mean(out < alpha)
    idx = idx + 1
}
write.table(rbind(delta_list, rej_rate), file=paste(save_dir, 'oracal_rej_rate.txt', sep = ''), row.names=FALSE, col.names=FALSE)
write.table(rbind(delta_list, rej_rate2), file=paste(save_dir, 'SuperLearner_rej_rate.txt', sep = ''), row.names=FALSE, col.names=FALSE)
