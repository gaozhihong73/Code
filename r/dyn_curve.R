
# 曲线函数库

# Logistic 模型 (3参数: K, b, r)
Logistic_curve <- function(para, time) {
  K <- para[1]
  b <- para[2]
  r <- para[3]
  
  return(K / (1 + b * exp(-r * time)))
}

# Gompertz 模型 (3参数: K, b, r)

Gompertz_curve <- function(para, time) {
  K <- para[1]
  b <- para[2]
  r <- para[3]
  
  return(K * exp(-b * exp(-r * time)))
}

# Richards 模型 (4参数: K, b, r, m)
# 公式: Y = K / (1 + b * exp(-r * t))^(1/m)
Richards_curve <- function(para, time) {
  K <- para[1]
  b <- para[2]
  r <- para[3]
  m <- para[4]
  
  # 保护: 防止底数为负
  base_val <- 1 + b * exp(-r * time)
  # 极小值保护
  base_val[base_val < 1e-9] <- 1e-9
  
  return(K / (base_val^(1 / m)))
}

# 4. Von Bertalanffy 模型 (3参数: L_inf, b, k)
# 公式: Y = L_inf * (1 - b * exp(-k * t))
# 常见于动物体长/鱼类生长
VonBertalanffy_curve <- function(para, time) {
  L_inf <- para[1]
  b     <- para[2]
  k     <- para[3]
  
  return(L_inf * (1 - b * exp(-k * time)))
}

# Double Logistic (双 Logistic, 叠加型)

Double_Logistic_curve <- function(para, time) {
  # 第一阶段
  K1 <- para[1]
  b1 <- para[2]
  r1 <- para[3]
  
  # 第二阶段
  K2 <- para[4]
  b2 <- para[5] # 注意: b2 需要很大才能让第二阶段在后期才启动
  r2 <- para[6]
  
  part1 <- K1 / (1 + b1 * exp(-r1 * time))
  part2 <- K2 / (1 + b2 * exp(-r2 * time))
  
  return(part1 + part2)
}
# 药物曲线
Hill_curve <- function(para, concentration) {
  # 参数解析
  E_inf <- para[1]  # 1. 最小反应值 (Bottom/Plateau): 药物极大剂量下的细胞生存率 (如 0)
  E_0   <- para[2]  # 2. 最大反应值 (Top/Baseline): 药物浓度为0时的生存率 (通常为 1 或 100)
  IC50  <- para[3]  # 3. 半抑制浓度 (IC50): 产生 50% 反应时的剂量
  H     <- para[4]  # 4. 斜率 (Hill Coefficient): 曲线的陡峭程度 (>0 为抑制)
  
  # Hill 方程核心公式
  # Response = Bottom + (Top - Bottom) / (1 + (Concentration / IC50)^Slope)
  cvalue <- E_inf + (E_0 - E_inf) / (1 + (concentration / IC50)^H)
  
  return(cvalue)
}


# LOP 工具函数
get_LOP_basis <- function(t_vec, order) {
  t_min <- min(t_vec); t_max <- max(t_vec)
  t_scaled <- 2 * (t_vec - t_min) / (t_max - t_min) - 1
  basis <- matrix(NA, nrow = length(t_vec), ncol = order + 1)
  basis[, 1] <- 1
  if(order >= 1) basis[, 2] <- t_scaled
  if(order >= 2) {
    for(k in 2:order) {
      n <- k 
      basis[, k+1] <- ((2*n - 1) * t_scaled * basis[, k] - (n - 1) * basis[, k-1]) / n
    }
  }
  colnames(basis) <- paste0("L", 0:order)
  return(basis)
}




loss_function <- function(para,y,time,fun,infect) {
  
  if(is.null(infect)){
    predicted <- fun(para, time)
  }else{
    predicted <- fun(para, time,infect)
  }
  residuals <- y - predicted
  return(sum(residuals^2))
}












