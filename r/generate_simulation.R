#!/usr/bin/env Rscript

suppressPackageStartupMessages({
  library(data.table)
  library(jsonlite)
  library(mvtnorm)
})

parse_args <- function(args) {
  defaults <- list(
    output_dir = "/root/DynaGS/dataset/sim_data/staging/scenario_smoke/rep_01",
    scenario = "smoke", rep = 1L, n_samples = 100L, n_snps = 200L,
    n_qtl = 20L, n_timepoints = 20L, heritability = 0.5,
    ld = 0.3, dominance_ratio = 0.3,
    prop_add = 0.4, prop_dom = 0.3, prop_mixed = 0.3,
    n_epistasis = 0L, epistasis_strength = 0.5,
    seed = 73L, overwrite = FALSE
  )
  if (length(args) %% 2 != 0) stop("Arguments must be supplied as --name value pairs.")
  for (i in seq(1, length(args), by = 2)) {
    key <- gsub("-", "_", sub("^--", "", args[[i]]))
    if (!key %in% names(defaults)) stop("Unknown argument: ", args[[i]])
    value <- args[[i + 1]]
    if (key %in% c("rep", "n_samples", "n_snps", "n_qtl", "n_timepoints",
                   "n_epistasis", "seed")) {
      value <- as.integer(value)
    } else if (key %in% c("heritability", "ld", "dominance_ratio",
                          "prop_add", "prop_dom", "prop_mixed",
                          "epistasis_strength")) {
      value <- as.numeric(value)
    } else if (key == "overwrite") {
      value <- tolower(value) %in% c("true", "1", "yes")
    }
    defaults[[key]] <- value
  }
  defaults
}

script_arg <- grep("^--file=", commandArgs(FALSE), value = TRUE)
script_path <- if (length(script_arg)) sub("^--file=", "", script_arg[[1]]) else "r/generate_simulation.R"
script_dir <- dirname(normalizePath(script_path, mustWork = TRUE))
source(file.path(script_dir, "dyn_curve.R"))
source(file.path(script_dir, "dyn_sim.R"))

args <- parse_args(commandArgs(trailingOnly = TRUE))
if (dir.exists(args$output_dir) && !args$overwrite) {
  stop("Output directory already exists; refusing to overwrite: ", args$output_dir)
}
if (args$n_qtl > args$n_snps) stop("n_qtl cannot exceed n_snps")
if (args$n_timepoints < 2) stop("n_timepoints must be at least 2")

parent <- dirname(args$output_dir)
dir.create(parent, recursive = TRUE, showWarnings = FALSE)
temporary <- tempfile(pattern = ".simulation-", tmpdir = parent)
dir.create(temporary, recursive = FALSE)
success <- FALSE
on.exit(if (!success && dir.exists(temporary)) unlink(temporary, recursive = TRUE), add = TRUE)

sim <- simulate_dynamic_LOP_Logistic_Types(
  N = args$n_samples,
  P = args$n_snps,
  nQTL = args$n_qtl,
  tgrid = seq_len(args$n_timepoints),
  LD_r = args$ld,
  h2_target = args$heritability,
  dom_ratio = args$dominance_ratio,
  prop_qtl_type = c(args$prop_add, args$prop_dom, args$prop_mixed),
  n_epistasis = args$n_epistasis,
  epistasis_strength = args$epistasis_strength,
  seed = args$seed
)

sample_ids <- rownames(sim$G)
time_names <- paste0("T", sim$Time)
colnames(sim$Y) <- time_names
colnames(sim$GV) <- time_names

fwrite(data.table(SampleID = sample_ids, as.data.table(sim$G)),
       file.path(temporary, "genotype.csv"))
fwrite(data.table(SampleID = sample_ids, as.data.table(sim$Y)),
       file.path(temporary, "phenotype.csv"))
fwrite(data.table(SampleID = sample_ids), file.path(temporary, "sample_ids.csv"))
fwrite(data.table(SampleID = sample_ids, as.data.table(sim$GV)),
       file.path(temporary, "true_gv.csv"))

effects <- data.table(
  snp_index = sim$QTL_index,
  snp_name = paste0("SNP", sim$QTL_index),
  effect_type = sim$QTL_Types,
  maf = sim$marker_maf[sim$QTL_index]
)
effects[, additive_coefficients := vapply(sim$True_Effects,
  function(x) paste(signif(x$additive_coef, 12), collapse = ";"), character(1))]
effects[, dominance_coefficients := vapply(sim$True_Effects,
  function(x) paste(signif(x$dominance_coef, 12), collapse = ";"), character(1))]
fwrite(effects, file.path(temporary, "causal_effects.csv"))

if (nrow(sim$epistasis_pairs)) {
  epistasis <- data.table(
    qtl_1_index = sim$QTL_index[sim$epistasis_pairs[, 1]],
    qtl_2_index = sim$QTL_index[sim$epistasis_pairs[, 2]],
    qtl_1_name = paste0("SNP", sim$QTL_index[sim$epistasis_pairs[, 1]]),
    qtl_2_name = paste0("SNP", sim$QTL_index[sim$epistasis_pairs[, 2]])
  )
  epistasis[, coefficients := apply(sim$epistasis_coefs, 1,
    function(x) paste(signif(x, 12), collapse = ";"))]
  fwrite(epistasis, file.path(temporary, "epistatic_effects.csv"))
}

git_commit <- tryCatch(
  system2("git", c("-C", normalizePath(file.path(script_dir, "..")), "rev-parse", "HEAD"), stdout = TRUE)[1],
  error = function(e) NA_character_
)
configuration <- list(
  format_version = "dynags_complete_simulation_v1",
  scenario = args$scenario,
  repetition = args$rep,
  n_samples = nrow(sim$G),
  n_snps = ncol(sim$G),
  time_coordinates = as.numeric(sim$Time),
  time_columns = time_names,
  target_heritability = sim$h2_target,
  realized_heritability = as.numeric(sim$h2_realized),
  mean_realized_heritability = mean(sim$h2_realized),
  ld_parameter = args$ld,
  qtl_count = length(sim$QTL_index),
  additive_dominance = list(
    dominance_ratio = sim$parameters$dom_ratio,
    qtl_type_proportions = unname(c(sim$parameters$prop_add,
      sim$parameters$prop_dom, sim$parameters$prop_mixed))
  ),
  epistasis = list(count = sim$parameters$n_epistasis,
                   strength = sim$parameters$epistasis_strength),
  seeds = list(simulation = args$seed),
  r_version = R.version.string,
  package_versions = list(
    mvtnorm = as.character(packageVersion("mvtnorm")),
    jsonlite = as.character(packageVersion("jsonlite")),
    data.table = as.character(packageVersion("data.table"))
  ),
  simulation_script_version = "complete-v2-dynamic-epistasis-cli",
  git_commit = git_commit
)
write_json(configuration, file.path(temporary, "simulation_config.json"),
           auto_unbox = TRUE, pretty = TRUE, digits = NA)

if (dir.exists(args$output_dir) && args$overwrite) {
  stop("Overwrite is intentionally unsupported for simulation data; choose a new output directory.")
}
if (!file.rename(temporary, args$output_dir)) stop("Could not atomically publish simulation output")
success <- TRUE
cat(toJSON(list(output_dir = normalizePath(args$output_dir),
                    genotype_shape = dim(sim$G), phenotype_shape = dim(sim$Y),
                    seed = args$seed), auto_unbox = TRUE, pretty = TRUE), "\n")
