library(zoo)

# The following function is copied from https://rdrr.io/github/satijalab/seurat/src/R/utilities.R 
LengthCheck <- function(values, cutoff = 0) {
  return(vapply(
    X = values,
    FUN = function(x) {
      return(length(x = x) > cutoff)
    },
    FUN.VALUE = logical(1)
  ))
}



# The following function is mainly copied from https://github.com/satijalab/seurat/blob/master/R/utilities.R AddModuleScore function.
# The difference lies in the computation of the control genes used. The control genes
# are computed in a neighborhood around the signature genes.
AdjustedNeighborhoodScoring <- function(object,
                                        features,
                                        pool = NULL,
                                        ctrl = 100,
                                        k = FALSE,
                                        assay = NULL,
                                        name = 'ANS_scores',
                                        search = FALSE,
                                        rm_genes_wo_valid_ctrl = TRUE,
                                        ...) {
  assay.old <- DefaultAssay(object = object)
  assay <- assay %||% assay.old
  DefaultAssay(object = object) <- assay
  assay.data <- GetAssayData(object = object)
  features.old <- features
  if (k) {
    .NotYetUsed(arg = 'k')
    features <- list()
    for (i in as.numeric(x = names(x = table(object@kmeans.obj[[1]]$cluster)))) {
      features[[i]] <-
        names(x = which(x = object@kmeans.obj[[1]]$cluster == i))
    }
    cluster.length <- length(x = features)
  } else {
    if (is.null(x = features)) {
      stop("Missing input feature list")
    }
    features <- lapply(
      X = features,
      FUN = function(x) {
        missing.features <- setdiff(x = x, y = rownames(x = object))
        if (length(x = missing.features) > 0) {
          warning(
            "The following features are not present in the object: ",
            paste(missing.features, collapse = ", "),
            ifelse(
              test = search,
              yes = ", attempting to find updated synonyms",
              no = ", not searching for symbol synonyms"
            ),
            
            call. = FALSE,
            immediate. = TRUE
          )
          if (search) {
            tryCatch(
              expr = {
                updated.features <-
                  UpdateSymbolList(symbols = missing.features, ...)
                names(x = updated.features) <- missing.features
                for (miss in names(x = updated.features)) {
                  index <- which(x == miss)
                  x[index] <- updated.features[miss]
                }
              },
              error = function(...) {
                warning(
                  "Could not reach HGNC's gene names database",
                  call. = FALSE,
                  immediate. = TRUE
                )
              }
            )
            missing.features <-
              setdiff(x = x, y = rownames(x = object))
            if (length(x = missing.features) > 0) {
              warning(
                "The following features are still not present in the object: ",
                paste(missing.features, collapse = ", "),
                call. = FALSE,
                immediate. = TRUE
              )
            }
          }
        }
        return(intersect(x = x, y = rownames(x = object)))
      }
    )
    cluster.length <- length(x = features)
  }
  if (!all(LengthCheck(values = features))) {
    warning(
      paste(
        'Could not find enough features in the object from the following feature lists:',
        paste(names(x = which(
          x = !LengthCheck(values = features)
        ))),
        'Attempting to match case...'
      )
    )
    features <- lapply(X = features.old,
                       FUN = CaseMatch,
                       match = rownames(x = object))
  }
  if (!all(LengthCheck(values = features))) {
    stop(
      paste(
        'The following feature lists do not have enough features present in the object:',
        paste(names(x = which(
          x = !LengthCheck(values = features)
        ))),
        'exiting...'
      )
    )
  }
  
  #compute average gene expression
  pool <- pool %||% rownames(x = object)
  data.avg <- Matrix::rowMeans(x = assay.data[pool,])
  data.avg <- data.avg[order(data.avg)]
  
  #filter features belonging to the c/2 largest avg. expressed genes
  nr_genes = length(data.avg)
  if (rm_genes_wo_valid_ctrl) {
    for (i in 1:cluster.length) {
      features.use <- features[[i]]
      valid_features <-
        sapply(features.use, function(x)
          which(names(data.avg) == x) <= (nr_genes - round(ctrl / 2)))
      features.use <- names(valid_features[valid_features == TRUE])
      if (length(features.use) == 0) {
        stop(
          paste(
            'After filtering signature genes belonging to the ctrl/2 genes with largest average expression, there are no remaining signature genes. Consider another signautre.',
            'exiting...'
          )
        )
      }
      features[[i]] <- features.use
    }
  }
  
  #compute control sets
  ctrl.use <- vector(mode = "list", length = cluster.length)
  for (i in 1:cluster.length) {
    features.use <- features[[i]]
    curr.data.avg <- data.avg[!(names(data.avg) %in% features.use)]
    rolled.means <-
      rollmean(curr.data.avg, ctrl, fill = NA, align = "right")
    for (j in 1:length(x = features.use)) {
      idx.best.nh = which.min(abs(rolled.means - data.avg[features.use[j]]))
      ctrl.use[[i]] <- c(ctrl.use[[i]],
                         names(rolled.means[(idx.best.nh - ctrl + 1):idx.best.nh]))
      
    }
  }
  ctrl.scores <- matrix(
    data = numeric(length = 1L),
    nrow = length(x = ctrl.use),
    ncol = ncol(x = object)
  )
  for (i in 1:length(ctrl.use)) {
    features.use <- ctrl.use[[i]]
    ctrl.scores[i,] <-
      Matrix::colMeans(x = assay.data[features.use,])
  }
  features.scores <- matrix(
    data = numeric(length = 1L),
    nrow = cluster.length,
    ncol = ncol(x = object)
  )
  for (i in 1:cluster.length) {
    features.use <- features[[i]]
    data.use <- assay.data[features.use, , drop = FALSE]
    features.scores[i,] <- Matrix::colMeans(x = data.use)
  }
  features.scores.use <- features.scores - ctrl.scores
  rownames(x = features.scores.use) <-
    paste0(name, 1:cluster.length)
  features.scores.use <-
    as.data.frame(x = t(x = features.scores.use))
  rownames(x = features.scores.use) <- colnames(x = object)
  object[[colnames(x = features.scores.use)]] <- features.scores.use
  CheckGC()
  DefaultAssay(object = object) <- assay.old
  return(object)
}
