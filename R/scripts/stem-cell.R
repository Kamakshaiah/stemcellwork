library(tm)

stemcelldata <- read.csv(file.choose())
names(stemcelldata)

# titles <- stemcelldata['Title']
# length(titles)
# dim(titles)
# str(titles)
# names(titles)
# head(titles)
# titles[1:2,]
# 
# library(tm)
# 
# titles_corp <- VCorpus((VectorSource(t(titles))))
# titles_corp <- tm_map(titles_corp, stripWhitespace)
# titles_corp <- tm_map(titles_corp, content_transformer(tolower))                          
# titles_corp <- tm_map(titles_corp, removeWords, stopwords("english"))
# 
# titles_dtm <- DocumentTermMatrix(titles_corp)
# inspect(titles_dtm)
# 
# findFreqTerms(titles_dtm, 5)
# 
# findAssocs(titles_dtm, 'treat', 0.6)
# findAssocs(titles_dtm, 'cell', 0.6)
# 
# inspect(DocumentTermMatrix(titles_dtm, list(dictionary = c("market", "sell", "service"))))

# abstracts

abstracts <- stemcelldata['Abstract']
length(t(abstracts)) # 719

abstracts_corp <- VCorpus((VectorSource(t(abstracts))))
abstracts_corp <- tm_map(abstracts_corp, stripWhitespace)
abstracts_corp <- tm_map(abstracts_corp, content_transformer(tolower))                          
abstracts_corp <- tm_map(abstracts_corp, removeWords, stopwords("english"))

abstracts_dtm <- DocumentTermMatrix(abstracts_corp)
inspect(abstracts_dtm)

findFreqTerms(abstracts_dtm, 5)

# findAssocs(abstracts_dtm, 'stem', 0.6)
#findAssocs(titles_dtm, 'cell', 0.6)

inspect(DocumentTermMatrix(abstracts_dtm, list(dictionary = c("market", "sell", "service"))))
inspect(removeSparseTerms(abstracts_dtm, 0.8))

sc_dense_data <- inspect(removeSparseTerms(abstracts_dtm, 0.99))
sc_dense_df <- as.data.frame(sc_dense_data)

# correlation
cor(sc_dense_df)

# names(sc_dense_df)
# typeof(sc_dense_df)

hc <- hclust(dist(sc_dense_df))
plot(hc)

# https://www.statology.org/k-means-clustering-in-r/

abstracts_matrix <- as.matrix(abstracts_dtm)
dim(abstracts_matrix)
head(abstracts_matrix[, 1:10])

# library(wordcloud)
# 
# wordcloud(colnames(abstracts_matrix), abstracts_matrix[2, ], max.words = 200)

# distMatrix <- dist(abstracts_matrix, method="euclidean")
# 
# groups <- hclust(distMatrix,method="ward.D")
# plot(groups, cex=0.9, hang=-1)
# rect.hclust(groups, k=5)

# kmeans_fit <- kmeans(abstracts_matrix, centers = 5, nstart = 20)
# 
# head(kmeans_fit$cluster)

# install.packages("ClusterR")
# install.packages("cluster")

# library(cluster)
# 
# clusplot(abstracts_matrix,
#          kmeans_fit$cluster,
#          lines = 0,
#          shade = TRUE,
#          color = TRUE,
#          labels = 2,
#          plotchar = FALSE,
#          span = TRUE,
#          )
# 
# library('fpc')
# plotcluster(abstracts_matrix, kmeans_fit$cluster) # crashing

library(mclust)
fit <- Mclust(abstracts_matrix[, 1:30])
plot(fit) # plot results
summary(fit) # display the best model

# https://www.geeksforgeeks.org/k-means-clustering-in-r-programming/

# cfa

sc_fa_df <- scale(sc_dense_df)
fa_out <- fa(cov(sc_fa_df), 3)
fa_out
summary(fa_out)

library(lavaan)


sc_model  <- '

  # latent 
  stemcell =~ cell + cells + stem
  medical =~ clinical + drug + treatment
  service =~ marketing + patients
  
  # regressions
  service ~ medical
  service ~ stemcell
  medical ~ stemcell
'

fit_sem <- sem(model = sc_model, data  = sc_dense_df)
fit_cfa <- cfa(model = sc_model, data  = sc_dense_df)
summary(fit_cfa)

write.csv(data.frame(fit_sem_summary$pe), 'sem_output.csv')

sc_med_model  <- '

  # latent 
  stemcell =~ cell + cells + stem
  medical =~ clinical + drug + treatment
  service =~ marketing + patients
  
  # direct
  service ~ c*stemcell
  
  # mediation
  
  medical ~ a*stemcell
  service ~ b*medical 
'

fit_sem_med <- sem(model = sc_med_model, data  = sc_fa_df)
# fit_cfa <- cfa(model = sc_model, data  = sc_dense_df)
summary(fit_sem_med)

sc_med_model  <- '

  # latent 
  stemcell =~ cell + cells + stem
  medical =~ clinical + drug + treatment
  service =~ marketing + patients
  
  # clinical_decisions =~ medical + stemcell
  # 
  # # regressions
  # clinical_decisions ~ stemcell
  # clinical_decisions ~ medical
  
  # direct
  stemcell ~ c*medical
  
  # mediation
  
  service ~ a*stemcell
  medical ~ b*service 
  
  # indirect effect (a*b)
  ab := a*b
  
  # total effect
  total := c + (a*b)
'
fit_sem_med <- sem(model = sc_med_model, data  = sc_fa_df)
summary(fit_sem_med)

write.csv(data.frame(summary(fit_sem_med)$pe), 'sem_med_output.csv')

library(semPlot)
semPaths(fit_sem_med, )
