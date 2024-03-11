library(ggplot2)
library(reshape2)
library(dplyr)
setwd("/Users/jeong-yeojin/Desktop/SpLSI/simulation")
res_df= read.csv("sim_all_cv_mean_interpolation_5.csv")
err_df = res_df[c("N", "vanilla_err", "splsi_err", "slda_err")]
err_df_melt = melt(err_df, id.var = 'N', variable.name = 'method')
colnames(err_df_melt) = c("N", "method", "error")

summary_stats = 
  err_df_melt %>%
  group_by(N, method) %>%
  mutate(method = recode(method, 
                         'vanilla_err' = 'PLSI', 
                         'splsi_err' = 'SPLSI',
                         'slda_err' = 'SLDA')) %>%
  summarise(mean_error = mean(error),
            sd_error = sd(error))
summary_stats$method <- factor(summary_stats$method, levels = c('SPLSI','PLSI','SLDA'))

blue_palette <- c( "#a6cee3", "#b2df8a", "#fb9a99")
ggplot(summary_stats, aes(x=N, y=mean_error, fill=method, color=method)) +
  geom_ribbon(aes(ymin = mean_error - sd_error, ymax = mean_error + sd_error), alpha = 0.3, color=NA) +
  geom_line() +
  geom_point() +
  labs(title = bquote(bold(Error~of~Topic~Weight~italic(W))), 
       x = "Document length (N)", 
       y = "Error") +
  #scale_color_brewer(palette="Paired") +
  scale_color_manual(values = blue_palette) +
  scale_fill_manual(values = blue_palette) +
  theme_minimal() +
  theme(legend.position="bottom",
        plot.title = element_text(size = 15, hjust = 0.5, face="bold"),
        axis.title = element_text(size = 13),
        legend.text = element_text(size = 12))



acc_df = res_df[c("N", "vanilla_acc", "splsi_acc", "slda_acc")]
acc_df_melt = melt(acc_df, id.var = 'N', variable.name = 'method')
colnames(acc_df_melt) = c("N", "method", "error")

summary_stats = 
  acc_df_melt %>%
  group_by(N, method) %>%
  mutate(method = recode(method, 
                         'vanilla_acc' = 'PLSI', 
                         'splsi_acc' = 'SPLSI',
                         'slda_acc' = 'SLDA')) %>%
  summarise(mean_error = mean(error),
            sd_error = sd(error))
summary_stats$method <- factor(summary_stats$method, levels = c('SPLSI','PLSI','SLDA'))

blue_palette <- c( "#a6cee3", "#b2df8a", "#fb9a99")
ggplot(summary_stats, aes(x=N, y=mean_error, fill=method, color=method)) +
  geom_ribbon(aes(ymin = mean_error - sd_error, ymax = mean_error + sd_error), alpha = 0.3, color=NA) +
  geom_line() +
  geom_point() +
  labs(title = "Accuracy of Topic Assignment", 
       x = "Document length (N)", 
       y = "Accuracy") +
  #scale_color_brewer(palette="Paired") +
  scale_color_manual(values = blue_palette) +
  scale_fill_manual(values = blue_palette) +
  theme_minimal() +
  theme(legend.position="bottom",
        plot.title = element_text(size = 15, hjust = 0.5, face="bold"),
        axis.title = element_text(size = 13),
        legend.text = element_text(size = 12))



time_df = res_df[c("N", "time_v", "time_splsi", "time_slda")]
time_df_melt = melt(time_df, id.var = 'N', variable.name = 'method')
colnames(time_df_melt) = c("N", "method", "time")

summary_stats = 
  time_df_melt %>%
  group_by(N, method) %>%
  mutate(method = recode(method, 
                         'time_v' = 'PLSI', 
                         'time_splsi' = 'SPLSI',
                         'time_slda' = 'SLDA')) %>%
  summarise(mean_error = mean(time),
            sd_error = sd(time))
summary_stats$method <- factor(summary_stats$method, levels = c('SPLSI','PLSI','SLDA'))

blue_palette <- c( "#a6cee3", "#b2df8a", "#fb9a99")
ggplot(summary_stats, aes(x=N, y=mean_error, fill=method, color=method)) +
  geom_ribbon(aes(ymin = mean_error - sd_error, ymax = mean_error + sd_error), alpha = 0.3, color=NA) +
  geom_line() +
  geom_point() +
  labs(title = "Running Time", 
       x = "Document length (N)", 
       y = "Time") +
  #scale_color_brewer(palette="Paired") +
  scale_color_manual(values = blue_palette) +
  scale_fill_manual(values = blue_palette) +
  theme_minimal() +
  theme(legend.position="bottom",
        plot.title = element_text(size = 15, hjust = 0.5, face="bold"),
        axis.title = element_text(size = 13),
        legend.text = element_text(size = 12))
  


lambd_df = res_df[c("N", "spatial_lambd")]

summary_stats = 
  lambd_df %>%
  group_by(N) %>%
  summarise(mean_error = mean(spatial_lambd),
            sd_error = sd(spatial_lambd))

blue_palette <- c("#a6cee3", "#b2df8a", "#fb9a99")
ggplot(summary_stats, aes(x=N, y=mean_error)) +
  geom_ribbon(aes(ymin = mean_error - sd_error, ymax = mean_error + sd_error), fill = blue_palette[1], alpha = 0.3) +
  geom_line(color = blue_palette[1]) +
  geom_point(color = blue_palette[1]) +
  labs(title = bquote(bold(Spatial~Regularization~lambda^{MST-CV})), 
       x = "Document length (N)", 
       y = "Lambda") +
  #scale_color_brewer(palette="Paired") +
  scale_color_manual() +
  scale_fill_manual() +
  theme_minimal() +
  theme(legend.position="bottom",
        plot.title = element_text(size = 15, hjust = 0.5, face="bold"),
        axis.title = element_text(size = 13),
        legend.text = element_text(size = 12))

