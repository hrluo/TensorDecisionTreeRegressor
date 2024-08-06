# read.csv('clean_code/Example_Figure1/Fig5_noisevar0.0_depth0.csv')
read.csv('clean_code/Example_Figure1/Fig5_lowrank_reg_noisevar0.0_depth0.csv')
read.csv('clean_code/Example_Figure1/Fig4_variance_noisevar0.01_depth4_seed30.csv')
read.csv('clean_code/Example_Figure1/Fig4_variance_noisevar0.01_depth4_seed30.csv')
# Fig5_lowrank_reg_noisevar0.01_depth3_seed39
# Fig4_lowrank_reg_noisevar0.1_depth4_seed37


fndir <- file.path('clean_code', 'Example_Figure1')

getdf <- function(sm='variance', fignum=5) {
  f5 <- lapply(c('0.0','0.01','0.1'), \(v) { # noise variance
    lapply(0:4, \(d) { # depth
      lapply(30:39, \(s) {
        # fn <- paste0('Fig', fignum, '_', sm, '_noisevar', v, '_depth', d, '.csv')
        fn <- paste0('Fig', fignum, '_', sm, '_noisevar', v, '_depth', d, '_seed', s, '.csv')
        read.csv(file.path(fndir, fn))
      }) |> purrr::reduce(rbind)
    }) |> purrr::reduce(rbind)
  }) |> purrr::reduce(rbind) 
  # print(head(f5))
  f5 <- f5 |> select(-X) |> 
    pivot_longer(6:8, names_to='method', values_to='MSE') |> 
    mutate(method = substr(method, 5, length(method))) |> 
    mutate(method = factor(method, levels=c('cp', 'tucker', 'mean'))) |> 
    # mutate(method = factor(method, levels=c('MSE_mean', 'MSE_cp', 'MSE_tucker'))) |> 
    group_by(sm, noisevar, depth, rank, method) |>
    summarise(MSE = mean(MSE)) |>
    ungroup()
  # mutate(rankdepth = paste(method, depth)) |> 
  f5
}
getdf('lowrank_reg', 5) |> 
  filter(noisevar==0.1, depth==2, rank==3, method=='cp')

plotdf <- function(df) {
  p <- df |> 
    ggplot(aes(x=method, y=MSE, fill=method, group=rankdepth)) +
    geom_point(position=position_dodge(width=0.7), aes(color=method), size=0.7) +
    # geom_segment( aes(x=method, xend=method, y=0, yend=MSE, color=method)) +
    # geom_text(aes(label=depth, color=method), position=position_dodge(width=0.7), size=2, fontface = 'bold') +
    # geom_bar(stat='identity', position=position_dodge(width=0.7), width=0.6) +
    ggh4x::facet_nested(noisevar~criterion+rank, labeller='label_both', scales='free_y') + 
    theme_bw() + 
    theme(legend.position = 'none') + 
    theme(axis.text.x = element_text(angle=90, vjust=0.3), axis.title.x=element_blank())
  p
}

plotdf <- function(df) {
  p <- df |> 
    mutate(epstext = paste0('ε ~ N(0, ', noisevar, ')')) |> 
    mutate(epstext = factor(epstext, levels=sapply(c(0,0.01,0.1), \(nv) paste0('ε ~ N(0, ', nv, ')')))) |> 
    mutate(ranktext = paste0('rank: ', rank)) |> 
    mutate(sm2 = if_else(sm=='variance', 'SSE', 'LRE')) |> 
    mutate(crittext = paste0('splitting criterion: ', sm2)) |> 
    # mutate(crittext = factor(crittext, levels=sapply(c('variance', 'lowrank_reg'), \(sc) paste0('splitting criterion: ', sc)))) |> 
    ggplot(aes(x=depth, y=MSE, color=method)) +
    geom_line(aes(linetype=method), alpha=0.5) + 
    geom_text(aes(label=substr(method, 0, 1)), fontface='bold', size=2.5) +
    # geom_text(aes(label=depth, color=method), position=position_dodge(width=0.7), size=2, fontface = 'bold') +
    ggh4x::facet_nested(epstext~crittext+ranktext, scales='free_y') + 
    scale_x_continuous(breaks=0:10) +
    theme_bw() + 
    theme(legend.position = 'bottom', panel.grid=element_blank(), 
          legend.margin = margin(0, 0, 0, 0),
          legend.spacing.x = unit(0, "mm"),
          legend.spacing.y = unit(0, "mm"))
  p
}

f5_SSE <- getdf('variance')
f5_LRE <- getdf('lowrank_reg')
p5_SSE <- plotdf(f5_SSE)
p5_LRE <- plotdf(f5_LRE)

p5_both <- p5_SSE + p5_LRE + 
  plot_annotation(title='Impact of tree depth and rank on prediction MSE', 
                  subtitle='scalar output:  y = 2X[:, 0, 1] * X[:, 2, 3] + 3X[:, 1, 0] * X[:, 2, 0] * X[:, 3, 0] + ε', 
                  theme=theme(plot.title=element_text(hjust = 0.5), 
                              plot.subtitle=element_text(hjust = 0.5))) 
p5_both
ggsave(file.path(fndir, 'Fig5_depth_median10.png'), p5_both, width=10, height=4.5)



f4_SSE <- getdf('variance', 4)
f4_LRE <- getdf('lowrank_reg', 4)
p4_SSE <- plotdf(f4_SSE)
p4_LRE <- plotdf(f4_LRE)

p4_both <- p4_SSE + p4_LRE + 
  plot_annotation(title='Impact of tree depth and rank on prediction MSE', 
                  subtitle='scalar output:  y = X[:,0,1] * X[:,0,1] + 2X[:,1,3] * X[:,1,3] + 3X[:,2,0] * X[:,2,0] * X[:,2,0] + ε', 
                  theme=theme(plot.title=element_text(hjust = 0.5), 
                              plot.subtitle=element_text(hjust = 0.5))) 
p4_both
ggsave(file.path(fndir, 'Fig4_depth_median10.png'), p4_both, width=10, height=4.5)




# f5_LRE <- f5 |> select(-X) |> 
#   pivot_longer(4:6, names_to='method', values_to='MSE') |> 
#   mutate(eq='y = X[:, 0, 1] ∗ X[:, 0, 1] + 2X[:, 1, 3] ∗ X[:, 1, 3] + 3X[:, 2, 0] ∗ X[:, 2, 0] ∗ X[:, 2, 0] + ε') |> 
#   mutate(criterion = 'LRE') |> 
#   mutate(method = substr(method, 5, length(method))) |> 
#   # mutate(method = factor(method, levels=c('MSE_mean', 'MSE_cp', 'MSE_tucker'))) |> 
#   mutate(rankdepth = paste(method, depth)) 
# rbind(f5_SSE, f5_LRE) |> 
#   ggplot(aes(x=method, y=MSE, fill=method, group=rankdepth)) +
#   # geom_point(position=position_dodge(width=0.3)) +
#   # geom_text(aes(label=depth, color=method), position=position_dodge(width=1)) +
#   geom_bar(stat='identity', position=position_dodge(width=0.4), width=0.3) +
#   ggh4x::facet_nested(noisevar~criterion+rank, labeller='label_both', scales='free_y') + 
#   # facet_grid(noisevar~rank, labeller = 'label_both', scales='free_y') + 
#   labs(title='Impact of tree depth and rank on prediction MSE', 
#        subtitle='y = X[:,0,1] * X[:,0,1] + 2X[:,1,3] * X[:,1,3] + 3X[:,2,0] * X[:,2,0] * X[:,2,0] + ε') + 
#   theme_bw() + 
#   theme(legend.position = 'none') + 
#   theme(axis.text.x = element_text(angle=90, vjust=0.5))
# 
# ggplot(f5, aes())