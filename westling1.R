source('R/causal.null.test.R')
source('R/mixed.dens.R')

##############################################################

# Set a different library location
lib_location <- "/home/ynychen/.local/lib"
# Set the new library location
.libPaths(lib_location)

# Set the CRAN mirror
cran_mirror <- "https://cran.r-project.org"

if (require(SuperLearner) == F){
  install.packages("SuperLearner", lib = lib_location, repos = cran_mirror)
}

if (require(earth) == F){
  install.packages("earth", lib = lib_location, repos = cran_mirror)
}

if (require(Rsolnp) == F){
  install.packages("Rsolnp" lib = lib_location, repos = cran_mirror)
}

if (require(sets) == F){
  install.packages("sets" lib = lib_location, repos = cran_mirror)
}

library(SuperLearner, lib = lib_location)
library(earth, lib = lib_location)
library(Rsolnp, lib = lib_location)
iibrary(sets, lib = lib_location)

##############################################################
num = 100
delta_list = c(0, 0.5) #seq(0, 0.5, 0.1)
data_dir = '/dataset/simu2/eval/'
save_dir = 'R/logs/simu2/eval/'
alpha = 0.05
##############################################################

for (delta in delta_list){
    p_val = rep(0, num)
    time_cost = rep(0, num)

    for (i in 0:(num-1)){
      load_dir = paste(data_dir, toString(i), '/', sep = '')
      dat = read.csv(paste(getwd(), load_dir, 'delta_', toString(delta), '_data.txt', sep = ''), header = F, sep = ' ')

      y = dat[[6]]
      a = dat[[1]]
      l = dat[2:5]
      start_time <- Sys.time()
      out <- causalNullTest(Y, A, W, p = c(1,2,Inf), control = list(cross.fit = FALSE, verbose=TRUE, g.n.bins = 2:5))
      end_time <- Sys.time()
      time_cost[i] = end_time - start_time
      p_val[i] = out$p.value

    }

    write.table(p_val, file=paste(save_dir, "p_val_oracal_delta_", delta , '.txt', sep = ''), row.names=FALSE, col.names=FALSE)
    write.table(time_cost, file=paste(save_dir, "run_time_oracal_delta_", delta , '.txt', sep = ''), row.names=FALSE, col.names=FALSE)
}


rej_rate = rep(0, length(delta_list))
run_time = rep(0, length(delta_list))

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
