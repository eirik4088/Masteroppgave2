#Load data
data <- read.csv("C:\\Users\\workbench\\eirik_master\\Results\\linear_reg_2\\df")

#modell the data
model <- lm(sqrt(ComponentIncrease)~Treatment+DataPoint,data=data)
summary(model)
confint(model)

#Quality controll plots
windowsFonts(A = windowsFont("Times New Roman"), windowsFont("Times New Roman"))
library(performance)
library(see)
library(ggplot2)
pp <- check_model(model)
p <- plot(pp)

p[[1]] <- p[[1]] + theme(text=element_text(size=14,  family="A")) + theme(plot.title=element_text(size=18,  family="A")) + theme(axis.title=element_text(size=14,  family="A"))
p[[2]] <- p[[2]] + theme(text=element_text(size=14,  family="A")) + theme(plot.title=element_text(size=18,  family="A")) + theme(axis.title=element_text(size=14,  family="A"))
p[[3]] <- p[[3]]  + theme(text=element_text(size=14,  family="A")) + theme(plot.title=element_text(size=18,  family="A")) + theme(axis.title=element_text(size=14,  family="A"))
p[[4]] <- p[[4]] + theme(text=element_text(size=14,  family="A")) + theme(plot.title=element_text(size=18,  family="A")) + theme(axis.title=element_text(size=14,  family="A"))
p[[5]] <- gg_cooksd(model, show.threshold = FALSE, label = FALSE) + theme(text=element_text(size=14,  family="A")) + theme(plot.title=element_text(size=18,  family="A")) + theme(axis.title=element_text(size=14,  family="A")) + theme(plot.title = element_text(hjust = -0.28, vjust=1.3))
p[[6]] <- p[[6]] + theme(text=element_text(size=14,  family="A")) + theme(plot.title=element_text(size=18,  family="A")) + theme(axis.title=element_text(size=14,  family="A"))
p

#Modified Tukey test
turkey = as.matrix(read.csv("C:\\Users\\workbench\\eirik_master\\Results\\linear_reg_1\\for_turkey_mpi"))
library(additivityTests)
mtukey.test(turkey, alpha=0.05)

