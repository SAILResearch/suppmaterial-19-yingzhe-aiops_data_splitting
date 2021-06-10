require(ggpubr)
require(effsize)
require(xtable)
require(ScottKnott)
require(gtools)
require(stringi)
require(stringr)
require(scales)
require(tidyr)
require(ggplot2)
require(dplyr)

#install.packages('gtools', repos='http://cran.us.r-project.org')

# RQ1: measuring data leakage (Fig. 4)
#models <- c('rf', 'nn', 'cart', 'rgf', 'svm')
models <- c('cart', 'rf')
datasets <- c('google', 'disk')
named_data <- c('Google', 'Backblaze')
names(named_data) <- datasets

splittings <- c(
  'Oracle',
  'T50', 'T60', 'T70', 'T80', 'T90',
  'R50', 'R60', 'R70', 'R80', 'R90'
)
names(splittings) <- c(
  'Oracle',
  'Time-based 50%/50%',
  'Time-based 60%/40%',
  'Time-based 70%/30%',
  'Time-based 80%/20%',
  'Time-based 90%/10%',
  'Random 50%/50%',
  'Random 60%/40%',
  'Random 70%/30%',
  'Random 80%/20%',
  'Random 90%/10%'
)

df <- NA
for (dataset in datasets) {
  for (model in models) {
    dfi <- read.csv(paste('./results/', paste('leakage', dataset, model, sep='_'), '.csv', sep=''))
    dfi$Dataset <- named_data[dataset]
    dfi$Model <- toupper(model)
    #dfi$Splitting.Approach <- splittings[dfi$Splitting.Approach]
    #dfi <- dfi[substr(dfi$Splitting.Approach, 1 , 4) != 'Time', ]
  
    if (typeof(df) != 'list') {
      df <- dfi
    } else {
      df <- rbind(df, dfi)
    }
  }
}
df$Dataset <- factor(df$Dataset, levels=c('Google', 'Backblaze'))
df$Splitting.Approach <- factor(df$Splitting.Approach)#, levels=c('T50', 'T60', 'T70', 'T80', 'T90', 'R50', 'R60', 'R70', 'R80', 'R90', 'Oracle'))
levels(df$Metric)[levels(df$Metric)=='F-1'] <- 'F1'

ggplot(data=df %>% filter(Splitting.Approach!='Oracle', Dataset=='Google'), aes(x=Splitting.Approach, y=Performance.Value, group=Splitting.Approach)) + 
  geom_boxplot() + 
  geom_hline(data=df %>% filter(Splitting.Approach=='Oracle', Dataset=='Google') %>% group_by(Model, Metric) %>% summarize(Value=median(Performance.Value)), aes(yintercept=Value), color='blue') + 
  theme(axis.text.x=element_text(angle=60, hjust=1), legend.position='none', axis.text=element_text(size=8)) + 
  labs(x='Splitting Approach', y='Performance') + ylim(0, 1) +
  facet_grid(Model ~ Metric)
ggsave('leakage_google.pdf', width=120, height=100, units='mm')

ggplot(data=df %>% filter(Splitting.Approach!='Oracle', Dataset=='Backblaze'), aes(x=Splitting.Approach, y=Performance.Value, group=Splitting.Approach)) + 
  geom_boxplot() + 
  geom_hline(data=df %>% filter(Splitting.Approach=='Oracle', Dataset=='Backblaze') %>% group_by(Model, Metric) %>% summarize(Value=median(Performance.Value)), aes(yintercept=Value), colour='blue') + 
  theme(axis.text.x=element_text(angle=60, hjust=1), legend.position='none', axis.text=element_text(size=8)) + 
  labs(x='Splitting Approach', y='Performance') + ylim(0, 1) +
  facet_grid(Model ~ Metric)
ggsave('leakage_disk.pdf', width=120, height=100, units='mm')

for (dataset in c('Google', 'Backblaze')) {
  for (model in c('RF', 'CART')) {
    for (approach in c('Random 50%/50%', 'Random 60%/40%', 'Random 70%/30%', 'Random 80%/20%', 'Random 90%/10%')) {
      for (metric in c('AUC', 'MCC', 'F1')) {
        obs1 = df[df$Splitting.Approach==approach & df$Metric==metric & df$Dataset==dataset & df$Model==model, ]$Performance.Value
        obs2 = df[df$Splitting.Approach=='Oracle' & df$Metric==metric & df$Dataset==dataset & df$Model==model, ]$Performance.Value
        res = wilcox.test(obs1, obs2, alternative = "two.sided")
        print(paste(dataset, model, approach, metric, res$p.value<0.05, res$p.value))
      }
    }
  }
}

# RQ1: failure rate (Fig. 5)
df <- read.csv('results/failure_rate_google.csv')
ggplot(df, aes(x=x, y=y)) + geom_line() + geom_point() + scale_y_continuous(labels = scales::percent, limits=c(0, 0.03)) +
  scale_x_continuous(breaks=seq(1, 28, 3)) + 
  labs(x='Time Period', y='Failure Rate') 
ggsave('failure_rate_google.pdf', width=90, height=60, units='mm')

df <- read.csv('results/failure_rate_disk.csv')
ggplot(df, aes(x=x, y=y)) + geom_line() + geom_point() + 
  scale_y_continuous(labels = scales::percent_format(accuracy = 0.01), limits=c(0, 0.002)) + 
  labs(x='Time Period', y='Failure Rate') + 
  scale_x_continuous(breaks=seq(1, 12, 2))
ggsave('failure_rate_disk.pdf', width=90, height=60, units='mm')

# RQ1: performance boxplots (Fig. 7 and 8)
models <- c('rf', 'nn', 'cart', 'rgf', 'svm')
datasets <- c('google', 'disk')
named_data <- c('Google', 'Backblaze')
names(named_data) <- datasets
plt_ls <- list()
n <- 1
for (dataset in datasets) {
  for (model in models) {
    df <- read.csv(paste('./results/', paste('splitting', dataset, model, sep='_'), '.csv', sep=''))
    dfo <- df[df$Test.Name=='Oracle',]
    df <- df[df$Test.Name!='Oracle',]
    df$Test.Name <- paste(substr(df$Test.Name, 1, 1), 100-as.integer(substr(df$Test.Name, stri_length(df$Test.Name)-2, stri_length(df$Test.Name)-1)), sep='')
  
    output<-matrix(nrow=10, ncol=3)
    i <- 1
    for (tst in unique(df$Test.Name)) {
      dfi <- df[df$Test.Name==tst, ]
      output[i, 1] <- tst
      output[i, 2] <- 1
      #output[i, 3] <- paste(toupper(substr(cliff.delta(dfi$Test.AUC, dfi$Oracle.AUC)$magnitude, 1, 1)), toupper(substr(cliff.delta(dfi$Test.AUC, dfo$Oracle.AUC)$magnitude, 1, 1)), sep='/')
      output[i, 3] <- toupper(substr(cliff.delta(dfi$Test.AUC, dfi$Oracle.AUC)$magnitude, 1, 1))
      i <- i + 1
    }
    dff <- data.frame(output)
    dff$X2=1.05
    df$Metric='AUC'
    plt_auc <- ggplot(df, aes(x=Test.Name)) + 
      facet_wrap(.~Metric) +
      geom_boxplot(aes(y=Test.AUC), colour='darkblue') +
      geom_boxplot(aes(y=Oracle.AUC), colour='darkgreen') +
      geom_hline(yintercept=mean(dfo$Oracle.AUC, linetype = "Oracle Model"), colour='red') + 
      theme(axis.text.x=element_text(angle=45, hjust=1)) + 
      labs(x=' ', y='Performance') + 
      ylim(0.7, 1.05) + geom_text(data=dff, aes(x=X1, y=X2, label=X3))
  
    output<-matrix(nrow=10, ncol=3)
    i <- 1
    for (tst in unique(df$Test.Name)) {
      dfi <- df[df$Test.Name==tst, ]
      output[i, 1] <- tst
      output[i, 2] <- 1
      #output[i, 3] <- paste(toupper(substr(cliff.delta(dfi$Test.F, dfi$Oracle.F)$magnitude, 1, 1)), toupper(substr(cliff.delta(dfi$Test.F, dfo$Oracle.F)$magnitude, 1, 1)), sep='/')
      output[i, 3] <- toupper(substr(cliff.delta(dfi$Test.F, dfi$Oracle.F)$magnitude, 1, 1))
      i <- i + 1
    }
    dff <- data.frame(output)
    dff$X2=0.75
    df$Metric='F1'
    plt_f1 <- ggplot(df, aes(x=Test.Name)) + 
      facet_wrap(.~Metric) +
      geom_boxplot(aes(y=Test.F), colour='darkblue') +
      geom_boxplot(aes(y=Oracle.F), colour='darkgreen') +
      geom_hline(yintercept=mean(dfo$Oracle.F, linetype = "Oracle Model"), colour='red') + 
      theme(axis.text.x=element_text(angle=45, hjust=1)) + 
      labs(x='Data Splitting Approach', y='Performance on F-Measure') + 
      ylim(0, 0.75) + geom_text(data=dff, aes(x=X1, y=X2, label=X3))
  
    output<-matrix(nrow=10, ncol=3)
    i <- 1
    for (tst in unique(df$Test.Name)) {
      dfi <- df[df$Test.Name==tst, ]
      output[i, 1] <- tst
      output[i, 2] <- 1
      #[i, 3] <- paste(toupper(substr(cliff.delta(dfi$Test.MCC, dfi$Oracle.MCC)$magnitude, 1, 1)), toupper(substr(cliff.delta(dfi$Test.MCC, dfo$Oracle.MCC)$magnitude, 1, 1)), sep='/')
      output[i, 3] <- toupper(substr(cliff.delta(dfi$Test.MCC, dfi$Oracle.MCC)$magnitude, 1, 1))
      i <- i + 1
    }
    dff <- data.frame(output)
    dff$X2=0.75
    df$Metric='MCC'
    df$Model=toupper(model)
    plt_mcc <- ggplot(df, aes(x=Test.Name)) + 
      facet_grid(Model~Metric) +
      scale_colour_manual(name="Test Case", values=c('On Validation Data'="darkblue", 'On Testing Data'='darkgreen')) +
      geom_boxplot(aes(y=Test.MCC, colour='On Validation Data')) +
      geom_boxplot(aes(y=Oracle.MCC, colour='On Testing Data')) +
      geom_hline(yintercept=mean(dfo$Oracle.MCC, linetype = "Oracle Model"), colour='red') + 
      theme(axis.text.x=element_text(angle=45, hjust=1)) + 
      labs(x=' ', y='Performance on MCC') + 
      ylim(0, 0.75) + geom_text(data=dff, aes(x=X1, y=X2, label=X3))
    plt_ls[[n]] <- list(plt_auc, plt_f1, plt_mcc)
    n <- n + 1
  }
}

ggarrange(plt_ls[[1]][[1]] + theme(axis.ticks.x=element_blank(), axis.text.x=element_blank(), axis.title.x=element_blank(), legend.position="none", axis.title.y=element_blank(), plot.margin=unit(c(5.5,5.5,5.5,19.5),"pt")), 
          plt_ls[[1]][[2]] + theme(axis.ticks.x=element_blank(), axis.text.x=element_blank(), axis.title.x=element_blank(), axis.title.y=element_blank(), legend.position="none"), 
          plt_ls[[1]][[3]] + theme(axis.ticks.x=element_blank(), axis.text.x=element_blank(), axis.title.x=element_blank(), axis.title.y=element_blank(), legend.position="none", plot.margin=unit(c(5.5,125.5,5.5,5.5),"pt")),
          plt_ls[[2]][[1]] + theme(axis.ticks.x=element_blank(), axis.text.x=element_blank(), axis.title.x=element_blank(), legend.position="none", axis.title.y=element_blank(), plot.margin=unit(c(5.5,5.5,5.5,19.5),"pt")), 
          plt_ls[[2]][[2]] + theme(axis.ticks.x=element_blank(), axis.text.x=element_blank(), axis.title.x=element_blank(), axis.title.y=element_blank(), legend.position="none"), 
          plt_ls[[2]][[3]] + theme(axis.ticks.x=element_blank(), axis.text.x=element_blank(), axis.title.x=element_blank(), axis.title.y=element_blank(), legend.position="none", plot.margin=unit(c(5.5,125.5,5.5,5.5),"pt")),
          plt_ls[[3]][[1]] + theme(axis.ticks.x=element_blank(), axis.text.x=element_blank(), axis.title.x=element_blank(), legend.position="none"), 
          plt_ls[[3]][[2]] + theme(axis.ticks.x=element_blank(), axis.text.x=element_blank(), axis.title.x=element_blank(), axis.title.y=element_blank(), legend.position="none"), 
          plt_ls[[3]][[3]] + theme(axis.ticks.x=element_blank(), axis.text.x=element_blank(), axis.title.x=element_blank(), axis.title.y=element_blank()),
          plt_ls[[4]][[1]] + theme(axis.ticks.x=element_blank(), axis.text.x=element_blank(), axis.title.x=element_blank(), legend.position="none", axis.title.y=element_blank(), plot.margin=unit(c(5.5,5.5,5.5,19.5),"pt")), 
          plt_ls[[4]][[2]] + theme(axis.ticks.x=element_blank(), axis.text.x=element_blank(), axis.title.x=element_blank(), axis.title.y=element_blank(), legend.position="none"), 
          plt_ls[[4]][[3]] + theme(axis.ticks.x=element_blank(), axis.text.x=element_blank(), axis.title.x=element_blank(), axis.title.y=element_blank(), legend.position="none", plot.margin=unit(c(5.5,125.5,5.5,5.5),"pt")),
          plt_ls[[5]][[1]] + theme(legend.position="none", axis.title.y=element_blank(), plot.margin=unit(c(5.5,5.5,5.5,19.5),"pt")), 
          plt_ls[[5]][[2]] + theme(axis.title.y=element_blank(), legend.position="none"), 
          plt_ls[[5]][[3]] + theme(axis.title.y=element_blank(), legend.position="none", plot.margin=unit(c(5.5,125.5,5.5,5.5),"pt")),
          widths = c(1.05, 1, 1.5), heights = c(1, 1, 1, 1, 1.25), nrow=5, ncol=3)
ggsave('splitting_google.pdf', width=190, height=180, units='mm', scale=1.6)
  
ggarrange(plt_ls[[6]][[1]] + theme(axis.ticks.x=element_blank(), axis.text.x=element_blank(), axis.title.x=element_blank(), legend.position="none", axis.title.y=element_blank(), plot.margin=unit(c(5.5,5.5,5.5,19.5),"pt")), 
          plt_ls[[6]][[2]] + theme(axis.ticks.x=element_blank(), axis.text.x=element_blank(), axis.title.x=element_blank(), axis.title.y=element_blank(), legend.position="none"), 
          plt_ls[[6]][[3]] + theme(axis.ticks.x=element_blank(), axis.text.x=element_blank(), axis.title.x=element_blank(), axis.title.y=element_blank(), legend.position="none", plot.margin=unit(c(5.5,125.5,5.5,5.5),"pt")),
          plt_ls[[7]][[1]] + theme(axis.ticks.x=element_blank(), axis.text.x=element_blank(), axis.title.x=element_blank(), legend.position="none", axis.title.y=element_blank(), plot.margin=unit(c(5.5,5.5,5.5,19.5),"pt")), 
          plt_ls[[7]][[2]] + theme(axis.ticks.x=element_blank(), axis.text.x=element_blank(), axis.title.x=element_blank(), axis.title.y=element_blank(), legend.position="none"), 
          plt_ls[[7]][[3]] + theme(axis.ticks.x=element_blank(), axis.text.x=element_blank(), axis.title.x=element_blank(), axis.title.y=element_blank(), legend.position="none", plot.margin=unit(c(5.5,125.5,5.5,5.5),"pt")),
          plt_ls[[8]][[1]] + theme(axis.ticks.x=element_blank(), axis.text.x=element_blank(), axis.title.x=element_blank(), legend.position="none"), 
          plt_ls[[8]][[2]] + theme(axis.ticks.x=element_blank(), axis.text.x=element_blank(), axis.title.x=element_blank(), axis.title.y=element_blank(), legend.position="none"), 
          plt_ls[[8]][[3]] + theme(axis.ticks.x=element_blank(), axis.text.x=element_blank(), axis.title.x=element_blank(), axis.title.y=element_blank()),
          plt_ls[[9]][[1]] + theme(axis.ticks.x=element_blank(), axis.text.x=element_blank(), axis.title.x=element_blank(), legend.position="none", axis.title.y=element_blank(), plot.margin=unit(c(5.5,5.5,5.5,19.5),"pt")), 
          plt_ls[[9]][[2]] + theme(axis.ticks.x=element_blank(), axis.text.x=element_blank(), axis.title.x=element_blank(), axis.title.y=element_blank(), legend.position="none"), 
          plt_ls[[9]][[3]] + theme(axis.ticks.x=element_blank(), axis.text.x=element_blank(), axis.title.x=element_blank(), axis.title.y=element_blank(), legend.position="none", plot.margin=unit(c(5.5,125.5,5.5,5.5),"pt")),
          plt_ls[[10]][[1]] + theme(legend.position="none", axis.title.y=element_blank(), plot.margin=unit(c(5.5,5.5,5.5,19.5),"pt")), 
          plt_ls[[10]][[2]] + theme(axis.title.y=element_blank(), legend.position="none"), 
          plt_ls[[10]][[3]] + theme(axis.title.y=element_blank(), legend.position="none", plot.margin=unit(c(5.5,125.5,5.5,5.5),"pt")),
          widths = c(1.05, 1, 1.5), heights = c(1, 1, 1, 1, 1.25), nrow=5, ncol=3)
ggsave('splitting_disk.pdf', width=190, height=180, units='mm', scale=1.6)

# RQ2: SK analysis (Fig. 9)
cols = c("AUC", "F1", "MCC")
models <- c('rf', 'nn', 'cart', 'rgf', 'svm')
datasets <- c('google', 'disk')
named_data <- c('Google', 'Backblaze')
names(named_data) <- datasets
df <- NA
for (dataset in datasets) {
  for (model in models) {
    dfi <- read.csv(paste('./results/', paste('splitting', dataset, model, sep='_'), '.csv', sep=''))
    dfi <- dfi[dfi$Test.Name!='Oracle', c("Test.Name", "Test.AUC", "Test.F", "Test.MCC", "Oracle.AUC", "Oracle.F", "Oracle.MCC")]
    dfi$Scenario <- paste(toupper(model), paste(substr(dfi$Test.Name, 1, 1), 100-as.integer(substr(dfi$Test.Name, stri_length(dfi$Test.Name)-2, stri_length(dfi$Test.Name)-1)), sep=''), sep='/')
    dfi$AUC <- dfi$Test.AUC - dfi$Oracle.AUC
    dfi$F1 <- dfi$Test.F - dfi$Oracle.F
    dfi$MCC <- dfi$Test.MCC - dfi$Oracle.MCC
    dfi <- dfi[, c("Scenario", "AUC", "F1", "MCC")]
    
    if (typeof(df) != 'list') {
      df <- dfi
    } else {
      df <- rbind(df, dfi)
    }
  }
  for (i in 1:3) {
    dfi <- data.frame(variable=df$Scenario, value=df[,cols[i]])
    
    # use SK
    sk <- with(dfi, SK(x=variable, y=value, model='y~x', which='x'))
    sk <-data.frame(Scenario=summary(sk)$Levels, Group=as.integer(summary(sk)$`SK(5%)`))
    
    # use SK-ESD
    #sk <- sk_esd(long2wide(dfi))$groups
    #sk <- data.frame(sk)
    #sk <- data.frame(Scenario=rownames(sk), Group=sk$sk)
    #sk$Scenario <- sub('\\.', '/', sk$Scenario)
    
    dfi <- data.frame(Scenario=df$Scenario, Metric=df[,cols[i]])
    dfi <- merge(dfi, sk, by='Scenario')
    dfi$type <- grepl('/T', dfi$Scenario)
    dfi$Group <- max(dfi$Group) - dfi$Group + 1
    ggplot(dfi, aes(x=Scenario, y=Metric, color=type)) + geom_boxplot() + facet_grid(.~Group, scales='free_x', space = "free_x") + 
      theme(axis.text.x = element_text(angle = 60, hjust = 1)) +
      labs(x='Model & Splitting Approach', y='Optimism') + 
      ggtitle(paste('Optimism of ', sub('\\.', '-', substr(cols[i], 1, 3)), ' between validation data and unseen testing data', sep='')) +
      theme(plot.title = element_text(hjust = 0.5)) +
      scale_color_discrete(name = "Splitting Type", labels = c('Random', 'Time-based'))
      
    ggsave(paste('splitting_sk_', dataset, '_', tolower(sub('\\.', '_', cols[i])), '.pdf', sep=''), width=190, height=30, units='mm', scale=2)
  }
}

# RQ2: confusion matrices (Fig. 10 and 11)
df <- read.csv('results/confusion_google_rf.csv')
df <- read.csv('results/confusion_disk_rf.csv')
df %>% group_by(Test.Name, Scenario) %>% summarize(TN=round(mean(TN), 0), FP=round(mean(FP), 0), FN=round(mean(FN), 0), TP=round(mean(TP), 0))

# R70 - Google - RF
Y <- c(92030, 791, 327, 919)  # TN, FP, FN, TP
# T70 - Google - RF
Y <- c(92315, 660, 491, 600)
# R70 - Backblaze - RF
Y <- c(1051488, 2891, 71, 991)
# T70 - Backblaze - RF
Y <- c(1053045, 1717, 145, 534)

TClass <- factor(c(0, 0, 1, 1), levels=c(1, 0))
PClass <- factor(c(0, 1, 0, 1), levels=c(0, 1))
df <- data.frame(TClass, PClass, Y)
ggplot(df, aes(x = TClass, y = PClass, fill = log(Y))) +
  geom_tile() +
# geom_text(aes(label = sprintf("%1.0f", Y)), vjust = 1) +
  geom_text(aes(label=str_c(round(Y/sum(Y)*100, 2), '%'))) + 
  scale_fill_gradient(low = "white", high = "dodgerblue") +
  labs(x='True class', y='Predicted class') +
  scale_x_discrete(position = "top") +
  theme(legend.position = "none") + 
  geom_tile(color = "black", fill = "black", alpha = 0)

ggsave('confusion_google_rf_r70.pdf', width=90, height=60, units='mm')
ggsave('confusion_google_rf_t70.pdf', width=90, height=60, units='mm')

ggsave('confusion_disk_rf_r70.pdf', width=90, height=60, units='mm')
ggsave('confusion_disk_rf_t70.pdf', width=90, height=60, units='mm')

# RQ3: concept drift (Fig. 12)
datasets <- c('google', 'disk')
named_data <- c('Google', 'Backblaze')
names(named_data) <- datasets
models <- c('rf', 'cart', 'svm', 'rgf', 'nn')
df <- NA
for (dataset in datasets) {
  dfi <- NA
  for (model in models) {
    dfj <- read.csv(paste('./results/', paste('concept_drift', dataset, model, sep='_'), '.csv', sep=''))
    dfj$Model <- toupper(model)
    if (typeof(dfi) != 'list') {
      dfi <- dfj
    } else {
      dfi <- rbind(dfi, dfj)
    }
  }
  dfi <- dfi[dfi$Testing.Period==dfi$Training.Period+1, ]
  dff <- data.frame(dfi %>% group_by(Training.Period, Model) %>% summarise(Train.E=as.integer(mean(Training.Error)), Test.E=as.integer(mean(Testing.Error)), Train.S=as.integer(mean(Training.Size)), Test.S=as.integer(mean(Testing.Size))))
  pvals <- numeric(nrow(dff))
  for (i in 1:nrow(dff)) {
    res <- prop.test(c(dff[i, 3], dff[i, 4]), c(dff[i, 5], dff[i, 6]), p = NULL, alternative = "two.sided", correct = TRUE)
    pvals[i] <- res$p.value
  }
  dff$diff = abs((dff$Test.E/dff$Test.S) - (dff$Train.E/dff$Train.S)) / (dff$Train.E/dff$Train.S)
  dfi <- data.frame(X=factor(2:(nrow(dff)/length(models)+1)), Sig=(pvals < 0.05), Y=dff$diff, Model=dff$Model)
  dfi$Dataset = named_data[dataset]

  if (typeof(df) != 'list') {
    df <- dfi
  } else {
    df <- rbind(df, dfi)
  }
}

ggplot(df %>% filter(Dataset=='Google'), aes(x=X, y=Y, shape=Sig, color=Sig)) + geom_point() + 
  scale_x_discrete(breaks=seq(2, 28, 3)) + geom_hline(yintercept=0) + 
  facet_grid(.~Model, scales='free_x') +
  labs(x='Time Period', y='Relative difference of error rate') + scale_color_discrete(name='Concept drift?') + 
  scale_shape_manual(name='Concept drift?', values=c(19, 17))
ggsave('concept_drift_google.pdf', width=190, height=60, units='mm')

ggplot(df %>% filter(Dataset=='Backblaze'), aes(x=X, y=Y, shape=Sig, color=Sig)) + geom_point() + 
  scale_x_discrete(breaks=seq(2, 12, 2)) + geom_hline(yintercept=0) + 
  facet_grid(.~Model, scales='free_x') +
  labs(x='Time Period', y='Relative difference of error rate') + scale_color_discrete(name='Concept drift?') + 
  scale_shape_manual(name='Concept drift?', values=c(19, 17))
ggsave('concept_drift_disk.pdf', width=190, height=60, units='mm')

# RQ3: correlation (Fig. 13)
#values_disk <- c('Target~smart_5_diff', 'smart_7~smart_9', 'smart_12~smart_193', 'smart_4~smart_193', 'smart_9~smart_193_diff', 'smart_5_diff~smart_187_diff', 'Target~smart_187')
#values_google <- c('Scheduling Class~Std CPU', 'Scheduling Class~CPU Requested', 'Num Tasks~Avg CPU', 'Scheduling Class~Avg CPU', 'User ID~Mem Requested', 'Target~CPU Requested')
values_google <- c('Scheduling Class~Std CPU', 'Scheduling Class~CPU Requested', 'Num Tasks~Avg CPU')
values_disk <- c('smart_7~smart_9', 'Target~smart_5_diff', 'smart_4~smart_193', 'smart_12~smart_193', 'smart_9~smart_193_diff')

dataset <- 'google'
df1 <- read.csv(paste('results/target_corr_', dataset, '.csv.', sep=''))
df2 <- read.csv(paste('results/explanatory_corr_', dataset, '.csv.', sep=''))
df <- rbind(df1, df2)
df$Index <- gsub(".*\\-(.*)", "\\1", df$Index)
ggplot(df[df$Index %in% values_google, ], aes(x=Period, y=Corr, color=Index, group=Index)) + geom_line(aes(linetype=Index), size=1) +
  labs(x='Time Period', y='Pair-wise Correlation') + 
  scale_color_brewer(name='Variable pairs', palette="Dark2") +
  scale_linetype_discrete(name='Variable pairs') + 
  scale_y_continuous(labels=c('-0.6', 'moderate', '-0.4', 'weak', '-0.2', 'very weak', '0.0', 'very weak', '0.2', 'weak', '0.4', 'moderate', '0.6'), breaks=seq(-0.6, 0.6, 0.1)) +
  scale_x_continuous(breaks=seq(1, 28, 3)) + 
  geom_hline(yintercept=seq(-0.6, 0.6, 0.2), color='gray') 
ggsave('corr_google.pdf', width=140, height=40, units='mm', scale=1.5)

dataset <- 'disk'
df1 <- read.csv(paste('results/target_corr_', dataset, '.csv.', sep=''))
df2 <- read.csv(paste('results/explanatory_corr_', dataset, '.csv.', sep=''))
df <- rbind(df1, df2)
df$Index <- gsub(".*\\-(.*)", "\\1", df$Index)
df$Index <- str_remove_all(df$Index, '_raw')
ggplot(df[df$Index %in% values_disk, ], aes(x=Period, y=Corr, color=Index, group=Index)) + geom_line(aes(linetype=Index), size=1) +
  labs(x='Time Period', y='Pair-wise Correlation') + 
  scale_color_brewer(name='Variable pairs', palette="Dark2") +
  scale_linetype_discrete(name='Variable pairs') + 
  scale_y_continuous(labels=c('-0.6', 'moderate', '-0.4', 'weak', '-0.2', 'very weak', '0.0', 'very weak', '0.2', 'weak', '0.4', 'moderate', '0.6', 'strong', '0.8'), breaks=seq(-0.6, 0.8, 0.1)) +
  scale_x_continuous(breaks=seq(1, 28, 3)) + 
  geom_hline(yintercept=seq(-0.6, 0.8, 0.2), color='gray') 
ggsave('corr_disk.pdf', width=140, height=40, units='mm', scale=1.5)

# RQ4: update vs static (Fig. 15)
df <- read.csv('results/update_disk.csv')
df$Scenario <- as.character(df$Scenario)
df$Scenario[df$Scenario=='Static Model'] <- 'Stationary Model'
df$Scenario[df$Scenario=='Update Model'] <- 'Updated Model'
colnames(df)[7:9] <- c('F1', 'AUC', 'MCC')
df <- df[, c('Scenario', 'Model', 'K', 'F1', 'AUC', 'MCC')]
df$K <- factor(df$K)
df <- gather(df, Metric, Value, -Scenario, -Model, -K, factor_key=TRUE)
df$Model <- factor(df$Model, levels=c('RF', 'NN', 'CART', 'RGF', 'SVM'))
df$Metric <- factor(df$Metric, levels=c('AUC', 'F1', 'MCC'))

plt_auc <- ggplot(df %>% group_by(Model, Scenario, K, Metric) %>% dplyr::filter(Metric=='AUC') %>% summarize(Value=mean(Value)), aes(x=K, y=Value, group=Scenario, color=Scenario)) + 
  geom_line() + facet_grid(Metric~Model) +
  labs(x='Time Period', y='Performance') + ylim(0.5, 1)
plt_f1 <- ggplot(df %>% group_by(Model, Scenario, K, Metric) %>% dplyr::filter(Metric=='F1') %>% summarize(Value=mean(Value)), aes(x=K, y=Value, group=Scenario, color=Scenario)) + 
  geom_line() + facet_grid(Metric~Model) +
  labs(x='Time Period', y='Performance') + ylim(0, 0.8)
plt_mcc <- ggplot(df %>% group_by(Model, Scenario, K, Metric) %>% dplyr::filter(Metric=='MCC') %>% summarize(Value=mean(Value)), aes(x=K, y=Value, group=Scenario, color=Scenario)) + 
  geom_line() + facet_grid(Metric~Model) +
  labs(x='Time Period', y='Performance') + ylim(0, 0.8)

ggarrange(plt_auc + theme(axis.title.y=element_blank(), axis.text.x=element_blank(), axis.ticks.x=element_blank(), axis.title.x=element_blank(), legend.position="none", plot.margin=unit(c(5.5,118.5,5.5,20),"pt")), 
          plt_f1 + theme(strip.text.x = element_blank(), axis.text.x=element_blank(), axis.ticks.x=element_blank(), axis.title.x=element_blank(), plot.margin=unit(c(0,5.5,5.5,5.5),"pt")), 
          plt_mcc + theme(axis.title.y=element_blank(), strip.text.x = element_blank(), legend.position="none", plot.margin=unit(c(0,118.5,5.5,20),"pt")),
          heights=c(1.2,0.9,1.25), nrow=3, ncol=1)
ggsave('update_disk.pdf', width=190, height=85, units='mm')

df <- read.csv('results/update_google.csv')
df$Scenario <- as.character(df$Scenario)
df$Scenario[df$Scenario=='Static Model'] <- 'Stationary Model'
df$Scenario[df$Scenario=='Update Model'] <- 'Updated Model'
colnames(df)[7:9] <- c('F1', 'AUC', 'MCC')
df <- df[, c('Scenario', 'Model', 'K', 'F1', 'AUC', 'MCC')]
df$K <- factor(df$K)
df <- gather(df, Metric, Value, -Scenario, -Model, -K, factor_key=TRUE)
df$Model <- factor(df$Model, levels=c('RF', 'NN', 'CART', 'RGF', 'SVM'))
df$Metric <- factor(df$Metric, levels=c('AUC', 'F1', 'MCC'))
df$K <- factor(df$K)

plt_auc <- ggplot(df %>% group_by(Model, Scenario, K, Metric) %>% dplyr::filter(Metric=='AUC') %>% summarize(Value=mean(Value)), aes(x=K, y=Value, group=Scenario, color=Scenario)) + 
  geom_line() + facet_grid(Metric~Model) +
  scale_x_discrete(breaks= seq(14, 28, 4)) +
  labs(x='Time Period', y='Performance') + ylim(0.5, 1)
plt_f1 <- ggplot(df %>% group_by(Model, Scenario, K, Metric) %>% dplyr::filter(Metric=='F1') %>% summarize(Value=mean(Value)), aes(x=K, y=Value, group=Scenario, color=Scenario)) + 
  geom_line() + facet_grid(Metric~Model) +
  scale_x_discrete(breaks= seq(14, 28, 4)) +
  labs(x='Time Period', y='Performance') + ylim(0, 0.8)
plt_mcc <- ggplot(df %>% group_by(Model, Scenario, K, Metric) %>% dplyr::filter(Metric=='MCC') %>% summarize(Value=mean(Value)), aes(x=K, y=Value, group=Scenario, color=Scenario)) + 
  geom_line() + facet_grid(Metric~Model) +
  scale_x_discrete(breaks= seq(14, 28, 4)) +
  labs(x='Time Period', y='Performance') + ylim(0, 0.8)

ggarrange(plt_auc + theme(axis.title.y=element_blank(), axis.text.x=element_blank(), axis.ticks.x=element_blank(), axis.title.x=element_blank(), legend.position="none", plot.margin=unit(c(5.5,118.5,5.5,20),"pt")), 
          plt_f1 + theme(strip.text.x = element_blank(), axis.text.x=element_blank(), axis.ticks.x=element_blank(), axis.title.x=element_blank(), plot.margin=unit(c(0,5.5,5.5,5.5),"pt")), 
          plt_mcc + theme(axis.title.y=element_blank(), strip.text.x = element_blank(), legend.position="none", plot.margin=unit(c(0,118.5,5.5,20),"pt")),
          heights=c(1.2,0.9,1.25), nrow=3, ncol=1)
ggsave('update_google.pdf', width=190, height=85, units='mm')

#RQ4: chunk size trend (Fig. 16)
ls <- c(
  'window_google_rf',
  'window_google_nn',
  'window_google_cart',
  'window_google_rgf',
  'window_google_svm',
  'window_disk_rf',
  'window_disk_nn',
  'window_disk_cart',
  'window_disk_rgf',
  'window_disk_svm'
)
datasets <- c(replicate(5, 'Google'), replicate(5, 'Backblaze'))
df <- NA
for (i in 1:10) {
  dfi <- read.csv(paste('./results/', ls[i], '.csv', sep=''))
  dfi$Dataset <- datasets[i]
  dfi <- dfi[dfi$K == -1, ]
  dfi <- dfi[dfi$N <= 24, ]
  if (typeof(df) != 'list') {
    df <- dfi
  } else {
    df <- rbind(df, dfi)
  }
}
df$Dataset <- factor(df$Dataset, levels=c('Google', 'Backblaze'))

dfo <- data.frame(df %>% group_by(Dataset, Model) %>% dplyr::filter(Scenario == 'Static Model') %>% summarize(Test.AUC=quantile(Test.AUC, 0.2)))
dfo$N <- 2
df <- df[df$Scenario != 'Static Model', ]
df <- rbind(df[, c('Dataset', 'Model', 'N', 'Test.AUC')], dfo)

ggplot(df %>% group_by(N, Dataset, Model) %>% summarize(AUC=mean(Test.AUC)), aes(x=N, y=AUC)) + 
  geom_line() + geom_point() + 
  facet_grid(Dataset~Model) + labs(x='Number of Time Periods', y='Performance of AUC') + 
  scale_x_continuous(breaks=seq(2, 24, 4))
ggsave('impact_chunk_size.pdf', width=190, height=85, units='mm')


# RQ4: time impact (Fig. 17)
ls <- c(
  'window_google_rf',
  'window_google_nn',
  'window_google_cart',
  'window_google_rgf',
  'window_google_svm',
  'window_disk_rf',
  'window_disk_nn',
  'window_disk_cart',
  'window_disk_rgf',
  'window_disk_svm'
)
datasets <- c(replicate(5, 'Google'), replicate(5, 'Backblaze'))
df <- NA
for (i in 1:10) {
  dfi <- read.csv(paste('./results/', ls[i], '.csv', sep=''))
  dfi$Dataset <- datasets[i]
  dfi <- dfi[dfi$K == -1, ]
  dfi <- dfi[dfi$N <= 24, ]
  if (typeof(df) != 'list') {
    df <- dfi
  } else {
    df <- rbind(df, dfi)
  }
}
df$Dataset <- factor(df$Dataset, levels=c('Google', 'Backblaze'))
dff <- df %>% group_by(N, Model, Dataset) %>% dplyr::filter(Scenario != 'Stationary Model') %>% summarize(Static.Time=mean(Training.Time)+mean(Testing.Time)*first(N)/2, Sliding.Time=(mean(Training.Time)+mean(Testing.Time))*first(N)/2)
dff$Sliding.Time <- dff$Sliding.Time/dff$Static.Time
dff$Static.Time <- 1
dfo <- df %>% group_by(Dataset, Model) %>% dplyr::filter(Scenario == 'Stationary Model') %>% summarize(Sliding.Time=1, Static.Time=1)
dfo$N <- 2
df <- rbind(data.frame(dfo), data.frame(dff))

ggplot(df, aes(x=N, y=Sliding.Time)) + 
  geom_line() + geom_point() + 
  facet_grid(Dataset~Model) + labs(x='Number of Time Periods', y='Time Cost Relative to Stationary Model') + 
  scale_x_continuous(breaks=seq(2, 24, 4)) + scale_y_continuous(breaks=seq(0, 12, 2))
ggsave('time_impact_chunk_size.pdf', width=190, height=85, units='mm')

# RQ4: prequential (Fig. 18)
df <- read.csv('results/prequential_google_rf_result.csv')
levels(df$Approach) <- c('Stationary Model', 'Updated Model')
ggplot(df, aes(x=Index, y=PAUC, color=Approach, group=Approach)) +
  geom_line() +
  labs(x='Processed testing samples', y='Prequential AUC') + 
  scale_x_continuous(labels = unit_format(unit = "K", scale = 1e-3)) + 
  labs(color='Scenario')
ggsave('prequential_auc_google_rf.pdf', width=190, height=60, units='mm')

df <- read.csv('results/prequential_disk_rf_result.csv')
levels(df$Approach) <- c('Stationary Model', 'Updated Model')
ggplot(df, aes(x=Index, y=PAUC, color=Approach, group=Approach)) +
  geom_line() +
  labs(x='Processed testing samples', y='Prequential AUC') + 
  scale_x_continuous(labels = unit_format(unit = "M", scale = 1e-6)) + 
  labs(color='Scenario')
ggsave('prequential_auc_disk_rf.pdf', width=190, height=60, units='mm')
