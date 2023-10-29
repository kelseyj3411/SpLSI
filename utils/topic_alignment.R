## Topic Alignment
library(purrr)
library(alto)
spleen = read.csv("spleen_features.csv",row.names = 1)
rowsums = rowSums(spleen)
hist(rowsums)
ntopics = 10
lda_params <- setNames(map(1:ntopics, ~ list(k = .)), 1:ntopics)
lda_models <- run_lda_models(spleen, lda_params)
result <- align_topics(lda_models)
plot(result)