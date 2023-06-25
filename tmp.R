# Set a different library location
lib_location <- "/home/ynychen/.local/lib"
# Set the new library location
.libPaths(lib_location)

# Set the CRAN mirror
cran_mirror <- "https://cran.r-project.org"

if (require(glmnet) == F){
  install.packages("glmnet", lib = lib_location, repos = cran_mirror)
}
