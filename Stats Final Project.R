data = read.csv("/Users/ballanr/Desktop/342/Data.csv", header=TRUE)
attach(data)

#################### Box-Cox ####################
library(Car)
boxCox(data$Dist2Bus+1~1)
boxCox(data$Dist2MSP~1)
boxCox(data$PopDensity_per100m2~1)
boxCox(data$Large_Road+1~1)
boxCox(data$TotAmenities+1~1)

#################### Variables ####################
dist2bus = data$Dist2Bus
dist2bus = (dist2bus + 1)^(-1/2)

dist2msp = data$Dist2MSP
dist2msp = dist2msp^(1/2)

popdens = data$PopDensity_per100m2
popdens = (popdens)^(1/3)

largeroad = data$Large_Road
largeroad = (largeroad+1)^(-1)

totamenities = data$TotAmenities
totamenities = (totamenities)^(1/3)

pudtud = (1/2)*(log(data$PUD_mean+1) + log(data$TUD_mean+1))

#################### Stepwise Regression ####################

n=1581
alpha.in = 0.1
alpha.out = 0.1

### STEP 1

### The constant mean model (the reduced model)
model0 = lm(pudtud~1) 

### The "full" models (the candidate models)
model11 = lm(pudtud~dist2bus)
model12 = lm(pudtud~dist2msp)
model13 = lm(pudtud~popdens)
model14 = lm(pudtud~largeroad)
model15 = lm(pudtud~totamenities)

### Calculation of the partial F statistic
F.stat11 = anova(model0, model11)[2,"F"]
F.stat12 = anova(model0, model12)[2,"F"]
F.stat13 = anova(model0, model13)[2,"F"]
F.stat14 = anova(model0, model14)[2,"F"]
F.stat15 = anova(model0, model15)[2,"F"]

### The partial F statistics are combined in one vector
F.stats1 = c(F.stat11, F.stat12, F.stat13, F.stat14, F.stat15)
F.stats1

### Location of the largest F statistic
which.max(F.stats1)

F.in1 = qf(alpha.in, 1, n-2, lower.tail=FALSE)
F.in1

# -----> totamenities gets added here

### STEP 2 w/ two variables

model15 = lm(pudtud~totamenities)

### The "full" models (the candidate models)
model21 = lm(pudtud~dist2bus+totamenities)
model22 = lm(pudtud~dist2msp+totamenities)
model23 = lm(pudtud~popdens+totamenities)
model24 = lm(pudtud~largeroad+totamenities)


### Calculation of the partial F statistic
F.stat21 = anova(model15, model21)[2,"F"]
F.stat22 = anova(model15, model22)[2,"F"]
F.stat23 = anova(model15, model23)[2,"F"]
F.stat24 = anova(model15, model24)[2,"F"]

### The partial F statistics are combined in one vector
F.stats2 = c(F.stat21, F.stat22,F.stat23,F.stat24)
F.stats2

### Location of the largest F statistic
which.max(F.stats2)

F.in2 = qf(alpha.in, 1, n-3, lower.tail=FALSE)
F.in2

# -----> dist2bus gets added in

### STEP 3 w/ two variables
### The "reduced" model with dist2bus (the candidate model)
model11 = lm(pudtud~dist2bus)

### The "full" model (the model from Step 2)
model21 = lm(pudtud~dist2bus+totamenities)

### Calculation of the partial F statistic
F.stat31 = anova(model11, model21)[2,"F"]
F.stat31

F.out3 = qf(alpha.out, 1, n-3, lower.tail=FALSE)
F.out3

# -----> totamenities isn't dropped

### STEP 2 w/ three variables

model21 = lm(pudtud~dist2bus+totamenities)

### The "full" models (the candidate models)
model221 = lm(pudtud~dist2bus+totamenities+dist2msp)
model222 = lm(pudtud~dist2bus+totamenities+popdens)
model223 = lm(pudtud~dist2bus+totamenities+largeroad)

### Calculation of the partial F statistic
F.stat221 = anova(model21, model221)[2,"F"]
F.stat222 = anova(model21, model222)[2,"F"]
F.stat223 = anova(model21, model223)[2,"F"]

### The partial F statistics are combined in one vector
F.stats22 = c(F.stat221, F.stat222,F.stat223)
F.stats22

### Location of the largest F statistic
which.max(F.stats22)

F.in2 = qf(alpha.in, 1, n-3, lower.tail=FALSE)
F.in2

# -----> largeroad is added

### STEP 3 w/ three variables
### The "reduced" model with largeroad (the candidate models)
modelred1 = lm(pudtud~dist2bus+largeroad)
modelred2 = lm(pudtud~totamenities+largeroad)

### The "full" model (the model from Step 2)
modelfull = lm(pudtud~dist2bus+totamenities+largeroad)

### Calculation of the partial F statistic
F.stat1 = anova(modelred1, modelfull)[2,"F"]
F.stat2 = anova(modelred2, modelfull)[2,"F"]
F.stats = c(F.stat1,F.stat2)
which.min(F.stats)

F.out3 = qf(alpha.out, 1, n-3, lower.tail=FALSE)
F.out3

# -----> nothing is dropped

### STEP 2 w/ four variables

modelbase = lm(pudtud~dist2bus+totamenities+largeroad)

### The "full" models (the candidate models)
model1 = lm(pudtud~dist2bus+totamenities+largeroad+dist2msp)
model2 = lm(pudtud~dist2bus+totamenities+largeroad+popdens)

### Calculation of the partial F statistic
F.stat1 = anova(modelbase, model1)[2,"F"]
F.stat2 = anova(modelbase, model2)[2,"F"]

### The partial F statistics are combined in one vector
F.stats = c(F.stat1, F.stat2)
F.stats

### Location of the largest F statistic
which.max(F.stats)

F.in2 = qf(alpha.in, 1, n-3, lower.tail=FALSE)
F.in2

# -----> dist2msp is added

### STEP 3 w/ four variables

### The "reduced" model with dist2msp (the candidate models)
modelred1 = lm(pudtud~dist2bus+totamenities+dist2msp)
modelred2 = lm(pudtud~dist2bus+largeroad+dist2msp)
modelred3 = lm(pudtud~totamenities+largeroad+dist2msp)

### The "full" model (the model from Step 2)
modelfull = lm(pudtud~dist2bus+totamenities+largeroad+dist2msp)

### Calculation of the partial F statistic
F.stat1 = anova(modelred1, modelfull)[2,"F"]
F.stat2 = anova(modelred2, modelfull)[2,"F"]
F.stat3 = anova(modelred3, modelfull)[2,"F"]
F.stats = c(F.stat1,F.stat2,F.stat3)
which.min(F.stats)

F.out3 = qf(alpha.out, 1, n-3, lower.tail=FALSE)
F.out3

# -----> nothing is dropped

### STEP 2 w/ five variables

modelbase = lm(pudtud~dist2bus+totamenities+largeroad+dist2msp)

### The "full" models (the candidate models)
model1 = lm(pudtud~dist2bus+totamenities+largeroad+dist2msp+popdens)

### Calculation of the partial F statistic
F.stat1 = anova(modelbase, model1)[2,"F"]
F.stat1

F.in2 = qf(alpha.in, 1, n-3, lower.tail=FALSE)
F.in2

# -----> popdens is not added

ourmodel = lm(pudtud~dist2bus+totamenities+largeroad+dist2msp)

#################### Subset testing ####################
library(leaps)
best.subset.cp=leaps(x=cbind(dist2bus,dist2msp,popdens,largeroad,totamenities), y=pudtud, method="Cp")
min.cp.location=which.min(best.subset.cp$Cp)
best.subset.cp$which[min.cp.location,]

#################### Residual Testing ####################

### Predicting the CI ###
library(MASS)
ourmodel = lm(pudtud~dist2bus+totamenities+largeroad+dist2msp)
newdata = data.frame(dist2bus=5.6,totamenities=2,largeroad=2,dist2msp=75)
pred.future = predict(ourmodel, newdata, interval="predict", level=0.95)
pred.future

### Shapiro-Wilk Test ###
resid = ourmodel$residuals
shapiro.test(resid)

### Normality Plot ###
qqnorm(resid, main="Normal Probability Plot of Residuals")
qqline(resid)

### Studentized Residuals Plot ###
library(MASS)
std.res = studres(ourmodel)
plot(std.res, main="Studentized Residuals", ylab="e.star",xlab="Observation Number", ylim=c(-4,4), xaxt = "n")
axis(1, at=1:1581)
abline(h=0)
abline(h=-3, col="blue", lty=2)
abline(h=3, col="blue", lty=2)

### Leverage Plot ###
h.model = influence.measures(ourmodel)$infmat[,"hat"]
plot(h.model, main="Plot of hi", ylab="Leverage", xlab="Observation Number", xaxt = "n")
axis(1, at=1:1581)
abline(h=0.006325, col="blue", lty=2)
length(which(h.model > 0.006325))

### Cook's Distance ###
cook.d.model = influence.measures(ourmodel)$infmat[,"cook.d"]
plot(cook.d.model, main="Plot of Cook's Distance", ylab="Cook's Distance", xlab="Observation Number", xaxt = "n")
axis(1, at=1:1581)
abline(h=0.0007683, col="blue", lty=2)
abline(h=2*0.0007683, col="blue", lty=2)
abline(h=3*0.0007683, col="blue", lty=2)
abline(h=4*0.0007683, col="blue", lty=2)
abline(h=5*0.0007683, col="blue", lty=2)
length(which(cook.d.model > 5*0.0007683))
