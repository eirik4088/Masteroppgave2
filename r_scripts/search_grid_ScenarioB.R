#Load data
data <- read.csv("C:\\Users\\workbench\\eirik_master\\Results\\linear_reg_1_art\\df")

#modell the data
model <- glm(ComponentIncrease+0.25~Treatment+DataPoint, family = Gamma("inverse"),data=data)
summary(model)

#Quality controll plots
library(DHARMa)
testDispersion(model)
sim <- simulateResiduals(fittedModel=model, plot=FALSE)
plot(sim)

#Modified Tukey test
turkey = as.matrix(sqrt(read.csv("C:\\Users\\workbench\\eirik_master\\Results\\linear_reg_1\\for_turkey_search_art")))
library(additivityTests)
mtukey.test(turkey, alpha=0.05)


