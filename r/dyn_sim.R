

.allocate_qtl_types <- function(nQTL, prop_qtl_type) {
  probs <- prop_qtl_type / sum(prop_qtl_type)
  raw_counts <- nQTL * probs
  counts <- floor(raw_counts)
  remainder <- nQTL - sum(counts)
  if (remainder > 0) {
    add_to <- order(raw_counts - counts, decreasing = TRUE)[seq_len(remainder)]
    counts[add_to] <- counts[add_to] + 1L
  }
  sample(rep(c("Add", "Dom", "Mixed"), counts))
}

.simulate_ar1_errors <- function(N, nT, rho, target_sd) {
  z <- matrix(0, nrow = N, ncol = nT)
  z[, 1] <- rnorm(N)
  if (nT > 1) {
    innovation_sd <- sqrt(1 - rho^2)
    for (j in 2:nT) {
      z[, j] <- rho * z[, j - 1] + innovation_sd * rnorm(N)
    }
  }

  # 每个时间点中心化并标准化，使有限样本中的残差方差与设定值一致。
  for (j in seq_len(nT)) {
    z[, j] <- z[, j] - mean(z[, j])
    current_sd <- sd(z[, j])
    if (is.finite(current_sd) && current_sd > 0) z[, j] <- z[, j] / current_sd
  }
  sweep(z, 2, target_sd, "*")
}

.simulate_ld_genotypes <- function(N, P, LD_r, maf_range) {
  maf <- runif(P, min = maf_range[1], max = maf_range[2])
  G <- matrix(0L, nrow = N, ncol = P)
  latent_previous <- rnorm(N)
  innovation_sd <- sqrt(1 - LD_r^2)

  for (j in seq_len(P)) {
    latent_current <- if (j == 1) {
      latent_previous
    } else {
      LD_r * latent_previous + innovation_sd * rnorm(N)
    }

    # Gaussian copula + HWE marginal probabilities:
    # P(G=0)=(1-p)^2, P(G=1)=2p(1-p), P(G=2)=p^2.
    threshold_0 <- qnorm((1 - maf[j])^2)
    threshold_1 <- qnorm(1 - maf[j]^2)
    G[, j] <- as.integer(latent_current > threshold_0) +
      as.integer(latent_current > threshold_1)
    latent_previous <- latent_current
  }

  list(G = G, maf = maf)
}


# Simulate longitudinal genotype-phenotype data for the DynaGS Sim_DataLoader.
simulate_dynamic_LOP_Logistic_Types <- function(
    N = 400,
    P = 2000,
    nQTL = 50,
    qtl_pool_size = P,
    tgrid = 1:20,
    LD_r = 0.1,
    maf_range = c(0.05, 0.5),

    # --- Mean growth curve ---
    mu_a = 30, mu_b = 45, mu_r = 0.5,

    # --- Genetic architecture ---
    lop_order = 3,
    h2_target = 0.3,
    dom_ratio = 0.3,
    prop_qtl_type = c(0.4, 0.3, 0.3),
    n_epistasis = 0,
    epistasis_strength = 0.5,

    # --- Residual and scale parameters ---
    phi_rho = 0.6,
    error_cv = 0.01,
    genetic_cv = 0.1,
    truncate_at_zero = TRUE,
    seed = NULL
) {
  if (!is.null(seed)) set.seed(seed)

  if (!exists("get_LOP_basis", mode = "function")) {
    stop("get_LOP_basis() is unavailable. Source dyn_curve.R before dyn_sim.R.")
  }
  if (N < 3 || P < 1 || nQTL < 1 || nQTL > P) {
    stop("Require N >= 3, P >= 1, and 1 <= nQTL <= P.")
  }
  if (length(qtl_pool_size) != 1 || !is.finite(qtl_pool_size) ||
      qtl_pool_size != as.integer(qtl_pool_size) ||
      qtl_pool_size < nQTL || qtl_pool_size > P) {
    stop("qtl_pool_size must be an integer between nQTL and P.")
  }
  if (length(tgrid) < 2 || any(!is.finite(tgrid)) || anyDuplicated(tgrid)) {
    stop("tgrid must contain at least two distinct finite time coordinates.")
  }
  if (!is.finite(LD_r) || abs(LD_r) >= 1) stop("LD_r must satisfy abs(LD_r) < 1.")
  if (length(maf_range) != 2 || maf_range[1] <= 0 ||
      maf_range[2] > 0.5 || maf_range[1] > maf_range[2]) {
    stop("maf_range must be c(min, max) with 0 < min <= max <= 0.5.")
  }
  if (!is.finite(h2_target) || h2_target <= 0 || h2_target >= 1) {
    stop("h2_target must be between 0 and 1.")
  }
  if (!is.finite(dom_ratio) || dom_ratio < 0 || dom_ratio > 1) {
    stop("dom_ratio must be between 0 and 1.")
  }
  if (length(prop_qtl_type) != 3 || any(prop_qtl_type < 0) ||
      sum(prop_qtl_type) <= 0) {
    stop("prop_qtl_type must be three non-negative values with a positive sum.")
  }
  if (!is.finite(phi_rho) || abs(phi_rho) >= 1) {
    stop("phi_rho must satisfy abs(phi_rho) < 1.")
  }
  if (length(n_epistasis) != 1 || !is.finite(n_epistasis) ||
      n_epistasis < 0 || n_epistasis != as.integer(n_epistasis) ||
      n_epistasis > choose(nQTL, 2)) {
    stop("n_epistasis must be between 0 and choose(nQTL, 2).")
  }
  if (length(epistasis_strength) != 1 || !is.finite(epistasis_strength) ||
      epistasis_strength < 0) {
    stop("epistasis_strength must be a non-negative finite scalar.")
  }

  cat("Step 1: Simulating LD genotypes with HWE marginals...\n")
  genotype_result <- .simulate_ld_genotypes(N, P, LD_r, maf_range)
  G_raw <- genotype_result$G
  marker_maf <- genotype_result$maf
  colnames(G_raw) <- paste0("SNP", seq_len(P))
  rownames(G_raw) <- paste0("ID", seq_len(N))

  QTL_idx <- sort(sample.int(qtl_pool_size, nQTL, replace = FALSE))
  G_QTL <- G_raw[, QTL_idx, drop = FALSE]
  qtl_types <- .allocate_qtl_types(nQTL, prop_qtl_type)
  cat("QTL Types Distribution:\n")
  print(table(qtl_types))

  Za <- G_QTL - 1
  Zd <- matrix(as.integer(G_QTL == 1), nrow = N, ncol = nQTL)

  cat("Step 2: Simulating the Logistic mean curve...\n")
  mean_curve <- mu_a / (1 + mu_b * exp(-mu_r * tgrid))
  nT <- length(tgrid)
  LOP_basis <- get_LOP_basis(tgrid, lop_order)

  cat("Step 3: Simulating additive, dominance, and optional epistatic effects...\n")
  scale_factor <- (mu_a * genetic_cv) / sqrt(nQTL)
  coef_decay <- exp(-(0:lop_order))
  GV_additive_raw <- matrix(0, N, nT)
  GV_dominance_raw <- matrix(0, N, nT)
  GV_epistasis_raw <- matrix(0, N, nT)
  alpha_matrix <- matrix(0, nQTL, lop_order + 1)
  delta_matrix <- matrix(0, nQTL, lop_order + 1)
  True_QTL_Effects <- vector("list", nQTL)

  for (j in seq_len(nQTL)) {
    raw_alpha <- rnorm(lop_order + 1) * coef_decay * scale_factor
    raw_delta <- rnorm(lop_order + 1) * coef_decay * scale_factor

    if (qtl_types[j] == "Add") {
      alpha_coefs <- raw_alpha
      delta_coefs <- raw_delta * 0
    } else if (qtl_types[j] == "Dom") {
      alpha_coefs <- raw_alpha * 0
      delta_coefs <- raw_delta
    } else {
      alpha_coefs <- raw_alpha * sqrt(1 - dom_ratio)
      delta_coefs <- raw_delta * sqrt(dom_ratio)
    }

    alpha_matrix[j, ] <- alpha_coefs
    delta_matrix[j, ] <- delta_coefs
    eff_curve_a <- as.vector(LOP_basis %*% alpha_coefs)
    eff_curve_d <- as.vector(LOP_basis %*% delta_coefs)
    GV_additive_raw <- GV_additive_raw + outer(Za[, j], eff_curve_a)
    GV_dominance_raw <- GV_dominance_raw + outer(Zd[, j], eff_curve_d)

    True_QTL_Effects[[j]] <- list(
      snp_index = QTL_idx[j],
      snp_name = colnames(G_raw)[QTL_idx[j]],
      type = qtl_types[j],
      additive_coef = alpha_coefs,
      dominance_coef = delta_coefs,
      additive_curve = eff_curve_a,
      dominance_curve = eff_curve_d
    )
  }

  epistasis_pairs <- matrix(integer(0), ncol = 2)
  epistasis_coefs <- matrix(numeric(0), ncol = lop_order + 1)
  if (n_epistasis > 0) {
    all_pairs <- t(combn(seq_len(nQTL), 2))
    epistasis_pairs <- all_pairs[
      sample.int(nrow(all_pairs), n_epistasis, replace = FALSE), , drop = FALSE
    ]
    epistasis_coefs <- matrix(0, n_epistasis, lop_order + 1)
    for (k in seq_len(n_epistasis)) {
      pair_coef <- rnorm(lop_order + 1) * coef_decay * scale_factor * epistasis_strength
      epistasis_coefs[k, ] <- pair_coef
      pair_curve <- as.vector(LOP_basis %*% pair_coef)
      pair_code <- Za[, epistasis_pairs[k, 1]] * Za[, epistasis_pairs[k, 2]]
      pair_code <- pair_code - mean(pair_code)
      GV_epistasis_raw <- GV_epistasis_raw + outer(pair_code, pair_curve)
    }
  }

  scaling_ratios <- mean_curve / max(mean_curve)
  GV_additive <- sweep(GV_additive_raw, 2, scaling_ratios, "*")
  GV_dominance <- sweep(GV_dominance_raw, 2, scaling_ratios, "*")
  GV_epistasis <- sweep(GV_epistasis_raw, 2, scaling_ratios, "*")
  GV_total <- GV_additive + GV_dominance + GV_epistasis

  cat("Step 4: Adding time-correlated residuals at the target heritability...\n")
  Y_true <- sweep(GV_total, 2, mean_curve, "+")
  Vg_t <- apply(GV_total, 2, var)
  Ve_target <- Vg_t * (1 - h2_target) / h2_target
  min_Ve <- (mean_curve * error_cv)^2
  Ve_t <- pmax(Ve_target, min_Ve)
  Errors <- .simulate_ar1_errors(N, nT, phi_rho, sqrt(Ve_t))
  Y_obs <- Y_true + Errors

  if (truncate_at_zero) {
    Y_true[Y_true < 0.001] <- 0.001
    Y_obs[Y_obs < 0.001] <- 0.001
  }
  Errors_realized <- Y_obs - Y_true
  realized_h2 <- Vg_t / (Vg_t + apply(Errors_realized, 2, var))

  time_names <- paste0("V", seq_len(nT))
  dimnames(Y_obs) <- list(rownames(G_raw), time_names)
  dimnames(Y_true) <- list(rownames(G_raw), time_names)
  dimnames(GV_total) <- list(rownames(G_raw), time_names)
  dimnames(GV_additive) <- list(rownames(G_raw), time_names)
  dimnames(GV_dominance) <- list(rownames(G_raw), time_names)
  dimnames(GV_epistasis) <- list(rownames(G_raw), time_names)
  dimnames(Errors_realized) <- list(rownames(G_raw), time_names)

  list(
    Y = Y_obs,
    Y_true = Y_true,
    G = G_raw,
    QTL_index = QTL_idx,
    QTL_Types = qtl_types,
    Time = tgrid,
    Mean_Curve = mean_curve,
    GV = GV_total,
    GV_additive = GV_additive,
    GV_dominance = GV_dominance,
    GV_epistasis = GV_epistasis,
    Errors = Errors_realized,
    h2_target = h2_target,
    h2_realized = realized_h2,
    marker_maf = marker_maf,
    True_Effects = True_QTL_Effects,
    epistasis_pairs = epistasis_pairs,
    epistasis_coefs = epistasis_coefs,
    LOP_basis = LOP_basis,
    parameters = list(
      N = N, P = P, nQTL = nQTL, qtl_pool_size = qtl_pool_size,
      lop_order = lop_order,
      LD_r = LD_r, maf_min = maf_range[1], maf_max = maf_range[2],
      mu_a = mu_a, mu_b = mu_b, mu_r = mu_r,
      h2_target = h2_target, dom_ratio = dom_ratio,
      prop_add = prop_qtl_type[1] / sum(prop_qtl_type),
      prop_dom = prop_qtl_type[2] / sum(prop_qtl_type),
      prop_mixed = prop_qtl_type[3] / sum(prop_qtl_type),
      n_epistasis = n_epistasis, epistasis_strength = epistasis_strength,
      phi_rho = phi_rho, error_cv = error_cv, genetic_cv = genetic_cv,
      truncate_at_zero = truncate_at_zero, seed = seed
    )
  )
}


.write_dynags_matrix <- function(x, sample_ids, path) {
  output <- data.frame(SampleID = sample_ids, x, check.names = FALSE)
  write.table(output, path, sep = ",", row.names = FALSE, col.names = TRUE,
              quote = FALSE, na = "")
}


# Legacy six-file exporter. The supported pipeline uses generate_simulation.R and
# one complete genotype/phenotype pair. Call this function only for old artifacts.
export_dynags_simulation_legacy <- function(
    sim_data,
    output_dir,
    scenario = "S1",
    rep = 1,
    split_ratio = c(0.8, 0.1, 0.1),
    split_seed = 73,
    export_true_gv = TRUE
) {
  warning("export_dynags_simulation_legacy() writes pre-split CSV files and is deprecated")
  required_objects <- c("G", "Y", "GV", "QTL_index", "QTL_Types",
                        "Time", "h2_target", "h2_realized", "marker_maf",
                        "parameters")
  missing_objects <- setdiff(required_objects, names(sim_data))
  if (length(missing_objects) > 0) {
    stop("sim_data is missing: ", paste(missing_objects, collapse = ", "))
  }
  if (length(split_ratio) != 3 || any(split_ratio <= 0) ||
      abs(sum(split_ratio) - 1) > 1e-8) {
    stop("split_ratio must contain three positive values that sum to 1.")
  }

  N <- nrow(sim_data$G)
  P <- ncol(sim_data$G)
  if (N < 3) stop("At least three individuals are required for Train/Val/Test.")

  dir.create(output_dir, recursive = TRUE, showWarnings = FALSE)
  h_label <- format(sim_data$h2_target, scientific = FALSE, trim = TRUE)
  prefix <- sprintf(
    "DynaGS_%s_P%d_H%s_N%d_Rep%02d",
    scenario, P, h_label, N, as.integer(rep)
  )
  experiment_dir <- file.path(output_dir, prefix)
  dir.create(experiment_dir, recursive = TRUE, showWarnings = FALSE)

  set.seed(split_seed)
  shuffled <- sample.int(N)
  n_train <- floor(N * split_ratio[1])
  n_val <- floor(N * split_ratio[2])
  n_test <- N - n_train - n_val
  if (min(n_train, n_val, n_test) < 1) {
    stop("The requested split leaves an empty Train, Val, or Test set.")
  }
  split_indices <- list(
    Train = shuffled[seq_len(n_train)],
    Val = shuffled[n_train + seq_len(n_val)],
    Test = shuffled[n_train + n_val + seq_len(n_test)]
  )

  file_records <- list()
  for (split_name in names(split_indices)) {
    idx <- split_indices[[split_name]]
    sample_ids <- rownames(sim_data$G)[idx]
    genotype_path <- file.path(
      experiment_dir, paste0(prefix, "_", split_name, "_Genotype.csv")
    )
    phenotype_path <- file.path(
      experiment_dir, paste0(prefix, "_", split_name, "_Phenotype.csv")
    )
    .write_dynags_matrix(sim_data$G[idx, , drop = FALSE], sample_ids, genotype_path)
    .write_dynags_matrix(sim_data$Y[idx, , drop = FALSE], sample_ids, phenotype_path)

    true_gv_path <- NA_character_
    if (export_true_gv) {
      true_gv_path <- file.path(
        experiment_dir, paste0(prefix, "_", split_name, "_TrueGV.csv")
      )
      .write_dynags_matrix(sim_data$GV[idx, , drop = FALSE], sample_ids, true_gv_path)
    }
    file_records[[split_name]] <- data.frame(
      Split = split_name,
      N = length(idx),
      Genotype = normalizePath(genotype_path, winslash = "/", mustWork = FALSE),
      Phenotype = normalizePath(phenotype_path, winslash = "/", mustWork = FALSE),
      TrueGV = if (is.na(true_gv_path)) "" else
        normalizePath(true_gv_path, winslash = "/", mustWork = FALSE),
      stringsAsFactors = FALSE
    )
  }

  split_table <- do.call(rbind, lapply(names(split_indices), function(split_name) {
    idx <- split_indices[[split_name]]
    data.frame(
      SampleID = rownames(sim_data$G)[idx],
      OriginalRow = idx,
      Split = split_name,
      stringsAsFactors = FALSE
    )
  }))
  write.csv(split_table, file.path(experiment_dir, paste0(prefix, "_Split.csv")),
            row.names = FALSE, quote = FALSE)

  qtl_truth <- data.frame(
    SNP_Index = sim_data$QTL_index,
    SNP = paste0("SNP", sim_data$QTL_index),
    Type = sim_data$QTL_Types,
    MAF = sim_data$marker_maf[sim_data$QTL_index],
    stringsAsFactors = FALSE
  )
  write.csv(qtl_truth, file.path(experiment_dir, paste0(prefix, "_QTL_Truth.csv")),
            row.names = FALSE, quote = FALSE)

  write.csv(
    data.frame(TimeColumn = colnames(sim_data$Y), TimeCoordinate = sim_data$Time),
    file.path(experiment_dir, paste0(prefix, "_Time.csv")),
    row.names = FALSE, quote = FALSE
  )
  write.csv(
    data.frame(
      TimeColumn = colnames(sim_data$Y),
      TimeCoordinate = sim_data$Time,
      Target_H2 = sim_data$h2_target,
      Realized_H2 = sim_data$h2_realized
    ),
    file.path(experiment_dir, paste0(prefix, "_Heritability.csv")),
    row.names = FALSE, quote = FALSE
  )

  parameter_values <- sim_data$parameters
  parameter_values$scenario <- scenario
  parameter_values$rep <- rep
  parameter_values$split_seed <- split_seed
  parameter_values$time_coordinates <- paste(sim_data$Time, collapse = ";")
  parameter_table <- data.frame(
    Parameter = names(parameter_values),
    Value = vapply(parameter_values, function(x) paste(x, collapse = ";"), character(1)),
    stringsAsFactors = FALSE
  )
  write.csv(parameter_table,
            file.path(experiment_dir, paste0(prefix, "_Parameters.csv")),
            row.names = FALSE, quote = FALSE)

  # This small manifest is the hand-off point for the DynaGS data loader:
  # each split has one genotype file, one phenotype file, and optional true GV.
  # Relative file names keep the dataset portable between Windows and Linux.
  input_manifest <- do.call(rbind, lapply(names(file_records), function(split_name) {
    record <- file_records[[split_name]]
    data.frame(
      Split = record$Split,
      N = record$N,
      GenotypeFile = basename(record$Genotype),
      PhenotypeFile = basename(record$Phenotype),
      TrueGVFile = if (record$TrueGV == "") "" else basename(record$TrueGV),
      stringsAsFactors = FALSE
    )
  }))
  write.csv(
    input_manifest,
    file.path(experiment_dir, paste0(prefix, "_InputManifest.csv")),
    row.names = FALSE, quote = FALSE)

  list(
    prefix = prefix,
    experiment_dir = normalizePath(experiment_dir, winslash = "/", mustWork = FALSE),
    files = do.call(rbind, file_records),
    split_indices = split_indices
  )
}


# Legacy grid runner retained only to reproduce old pre-split artifacts.
run_dynags_simulation_grid_legacy <- function(
    grid,
    output_root,
    n_reps = 1,
    base_seed = 2025,
    split_ratio = c(0.8, 0.1, 0.1),
    common_args = list()
) {
  warning("run_dynags_simulation_grid_legacy() is deprecated; use generate_simulation.R per replicate")
  required_columns <- c("scenario", "N", "P", "nQTL", "h2_target")
  missing_columns <- setdiff(required_columns, names(grid))
  if (length(missing_columns) > 0) {
    stop("grid is missing columns: ", paste(missing_columns, collapse = ", "))
  }
  if (length(n_reps) != 1 || !is.finite(n_reps) ||
      n_reps < 1 || n_reps != as.integer(n_reps)) {
    stop("n_reps must be a positive integer.")
  }

  dir.create(output_root, recursive = TRUE, showWarnings = FALSE)
  manifest <- list()
  manifest_row <- 1L

  for (i in seq_len(nrow(grid))) {
    for (rep_id in seq_len(n_reps)) {
      simulation_seed <- base_seed + (i - 1L) * n_reps + rep_id - 1L
      row_args <- as.list(grid[i, setdiff(names(grid), "scenario"), drop = FALSE])
      row_args <- lapply(row_args, function(x) {
        value <- x[[1]]
        if (is.factor(value)) as.character(value) else value
      })
      simulation_args <- modifyList(common_args, row_args)
      simulation_args$seed <- simulation_seed

      cat(sprintf(
        "\n[%d/%d] scenario=%s rep=%d seed=%d N=%d P=%d h2=%s\n",
        manifest_row, nrow(grid) * n_reps, grid$scenario[i], rep_id,
        simulation_seed, simulation_args$N, simulation_args$P,
        simulation_args$h2_target
      ))
      sim_data <- do.call(simulate_dynamic_LOP_Logistic_Types, simulation_args)
      export_result <- export_dynags_simulation_legacy(
        sim_data = sim_data,
        output_dir = output_root,
        scenario = as.character(grid$scenario[i]),
        rep = rep_id,
        split_ratio = split_ratio,
        split_seed = simulation_seed + 100000L
      )

      manifest[[manifest_row]] <- data.frame(
        scenario = as.character(grid$scenario[i]),
        rep = rep_id,
        seed = simulation_seed,
        N = simulation_args$N,
        P = simulation_args$P,
        nQTL = simulation_args$nQTL,
        h2_target = simulation_args$h2_target,
        mean_realized_h2 = mean(sim_data$h2_realized, na.rm = TRUE),
        experiment_dir = export_result$experiment_dir,
        stringsAsFactors = FALSE
      )
      manifest_row <- manifest_row + 1L
      rm(sim_data)
      invisible(gc())
    }
  }

  manifest <- do.call(rbind, manifest)
  write.csv(manifest, file.path(output_root, "simulation_manifest.csv"),
            row.names = FALSE, quote = FALSE)
  manifest
}



#Evaluate DC-SIS Performance
evaluate_SIS_performance <- function(sis_result, sim_data, mode = "Auto") {
  
  true_idx <- sim_data$QTL_index
  true_names <- paste0("SNP", true_idx)
  
  # 获取真实的效应类型 
  if (!is.null(sim_data$QTL_Types)) {
    true_types <- sim_data$QTL_Types # "Add", "Dom", "Mixed"
    names(true_types) <- true_names
  } else {
    true_types <- rep("Mixed", length(true_names))
    names(true_types) <- true_names
  }
  
  # --- 2. 准备预测信息 (Prediction) ---
  selected_features <- names(sis_result$rr)
  n_selected <- length(selected_features)
  
  # --- 3. 模式检测 ---
  # 如果特征名里包含 "_Add" 或 "_Dom"，自动判定为 AD 模式
  is_AD_mode <- any(grepl("_Add|_Dom", selected_features))
  
  if (mode == "Standard") is_AD_mode <- FALSE
  if (mode == "AD") is_AD_mode <- TRUE
  
  cat("\n==========================================================\n")
  cat(sprintf(" PERFORMANCE REPORT (Mode: %s)\n", ifelse(is_AD_mode, "A/D Separated", "Standard")))
  cat("==========================================================\n")
  
  # ============================================================
  # 分支 A: A/D 分离模式评估
  # ============================================================
  if (is_AD_mode) {
    # 3.1 还原到位点级别 (Locus Level)
    # 去掉后缀，只看 SNP 编号
    selected_loci <- unique(gsub("_Add|_Dom", "", selected_features))
    
    # 3.2 位点级指标计算
    tp_loci_count <- sum(selected_loci %in% true_names)
    recall_loci <- tp_loci_count / length(true_names)
    prec_loci <- tp_loci_count / length(selected_loci)
    
    cat("[Locus Level - Did we find the gene?]\n")
    cat(sprintf("  - True QTLs:      %d\n", length(true_names)))
    cat(sprintf("  - Selected Loci:  %d (merged from %d features)\n", length(selected_loci), n_selected))
    cat(sprintf("  - True Positives: %d\n", tp_loci_count))
    cat(sprintf("  - Recall (Power): %.2f%%\n", recall_loci * 100))
    cat(sprintf("  - Precision:      %.2f%%\n", prec_loci * 100))
    
    # 3.3 分量级指标计算 (Component Level)
    cat("\n[Component Level - Did we distinguish A vs D?]\n")
    
    # 提取选中的 A 和 D
    sel_Add <- selected_features[grep("_Add", selected_features)]
    sel_Dom <- selected_features[grep("_Dom", selected_features)]
    
    # 提取真实的 A 和 D (根据模拟类型)
    # 真实有加性效应的 = Type is Add or Mixed
    true_has_Add <- names(true_types)[true_types %in% c("Add", "Mixed")]
    # 真实有显性效应的 = Type is Dom or Mixed
    true_has_Dom <- names(true_types)[true_types %in% c("Dom", "Mixed")]
    
    # 还原选中的名称去匹配真实名称
    sel_Add_base <- gsub("_Add", "", sel_Add)
    sel_Dom_base <- gsub("_Dom", "", sel_Dom)
    
    # 计算加性 Recall
    tp_Add <- sum(sel_Add_base %in% true_has_Add)
    rec_Add <- tp_Add / length(true_has_Add)
    
    # 计算显性 Recall
    tp_Dom <- sum(sel_Dom_base %in% true_has_Dom)
    rec_Dom <- tp_Dom / length(true_has_Dom)
    
    cat(sprintf("  - Additive Component Recall:  %.2f%% (Found %d/%d true Add/Mixed)\n", 
                rec_Add*100, tp_Add, length(true_has_Add)))
    cat(sprintf("  - Dominance Component Recall: %.2f%% (Found %d/%d true Dom/Mixed)\n", 
                rec_Dom*100, tp_Dom, length(true_has_Dom)))
    
    # 3.4 详细检查表
    # 为了展示方便，我们看前 15 个选中的特征
    check_n <- min(15, n_selected)
    top_feats <- selected_features[1:check_n]
    top_bases <- gsub("_Add|_Dom", "", top_feats)
    
    # 判断是否为真实位点
    is_true <- top_bases %in% true_names
    # 获取真实类型
    real_type <- true_types[top_bases]
    real_type[is.na(real_type)] <- "Noise" # 如果不是真QTL，标记为 Noise
    
    df_check <- data.frame(
      Rank = 1:check_n,
      Feature = top_feats,
      DC_Score = round(as.numeric(sis_result$rr)[1:check_n], 4),
      Is_True_Locus = is_true,
      True_Type = real_type
    )
    cat("\n[Top Selected Features Check]\n")
    print(df_check, row.names = FALSE)
    
  } else {
    # 
    # 标准模式评估 (Standard Mode)
    
    tp_count <- sum(selected_features %in% true_names)
    recall <- tp_count / length(true_names)
    precision <- tp_count / n_selected
    
    cat("[Standard Screening Performance]\n")
    cat(sprintf("  - True QTLs:      %d\n", length(true_names)))
    cat(sprintf("  - Selected SNPs:  %d\n", n_selected))
    cat(sprintf("  - True Positives: %d\n", tp_count))
    cat(sprintf("  - Recall (Power): %.2f%%\n", recall * 100))
    cat(sprintf("  - Precision:      %.2f%%\n", precision * 100))
    
    # 详细检查表
    check_n <- min(10, n_selected)
    top_feats <- selected_features[1:check_n]
    
    # 获取真实类型
    real_type <- true_types[top_feats]
    real_type[is.na(real_type)] <- "Noise"
    
    df_check <- data.frame(
      Rank = 1:check_n,
      SNP = top_feats,
      DC_Score = round(as.numeric(sis_result$rr)[1:check_n], 4),
      Is_True = (top_feats %in% true_names),
      True_Type = real_type
    )
    cat("\n[Top Selected SNPs Check]\n")
    print(df_check, row.names = FALSE)
  }
  
  cat("==========================================================\n")
}

evaluate_dynamic_selection_results <- function(results, true_qtl_index) {
  
  # --- 1. 均值参数估计 (Mean Parameters) ---
  # 处理 theta 可能是 list 或 vector 的情况
  theta_vals <- if(is.list(results$theta)) unlist(results$theta) else results$theta
  cat("Estimated Mean Params:\n")
  print(round(theta_vals, 4))
  
  # --- 2. 变量选择精度 (Variable Selection Accuracy) ---
  
  # A. 提取被选中的 SNP (去重)
  # 检查 results 中是否有 selected_effects 表格
  if (!is.null(results$selected_effects) && nrow(results$selected_effects) > 0) {
    # 只要 SNP 被选中了任意一种效应 (Add 或 Dom)，就算被选中
    selected_snps <- unique(results$selected_effects$SNP_Index)
  } else {
    selected_snps <- integer(0)
  }
  
  # B. 计算统计指标
  n_true <- length(true_qtl_index)
  n_selected <- length(selected_snps)
  
  # 交集 (真阳性)
  TP_set <- intersect(selected_snps, true_qtl_index)
  n_TP <- length(TP_set)
  n_FP <- n_selected - n_TP # 假阳性
  
  # 指标计算 (防止除以 0)
  precision <- if(n_selected > 0) n_TP / n_selected else 0
  recall    <- if(n_true > 0)     n_TP / n_true     else 0
  f1_score  <- if((precision + recall) > 0) 2 * (precision * recall) / (precision + recall) else 0
  
  # --- 3. 打印报告 ---
  cat("\n--- QTL Detection Accuracy ---\n")
  cat(sprintf("True QTLs       : %d\n", n_true))
  cat(sprintf("Selected SNPs   : %d\n", n_selected))
  cat(sprintf("True Positives  : %d\n", n_TP))
  cat(sprintf("False Positives : %d\n", n_FP))
  cat("--------------------------------\n")
  cat(sprintf("Precision       : %.2f%%\n", precision * 100))
  cat(sprintf("Recall (Power)  : %.2f%%\n", recall * 100))
  cat(sprintf("F1 Score        : %.2f%%\n", f1_score * 100))
  
  # --- 4. 效应类型分布 (Add vs Dom) ---
  if (n_selected > 0 && "Effect_Type" %in% names(results$selected_effects)) {
    cat("\n--- Selected Effect Types ---\n")
    print(table(results$selected_effects$Effect_Type))
    
    # 简单的加性/显性统计
    n_add <- sum(results$selected_effects$Effect_Type == "Add")
    n_dom <- sum(results$selected_effects$Effect_Type == "Dom")
    cat(sprintf("Ratio (Add:Dom) : %d:%d\n", n_add, n_dom))
  } else if (n_selected == 0) {
    cat("\n[Warning] No SNPs were selected by the model.\n")
  }
  
  cat("========================================\n\n")
  
  # 返回列表以便后续画图或存储
  return(invisible(list(
    precision = precision,
    recall = recall,
    f1_score = f1_score,
    TP = n_TP,
    FP = n_FP,
    selected_snps = selected_snps,
    theta_est = theta_vals
  )))
}
