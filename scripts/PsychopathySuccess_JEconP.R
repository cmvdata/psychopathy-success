##################################################
library(ez)
library(zoo)
library(eyeTrackR)
library(car)
library(lme4)
library(pastecs)
library(psych)
library(lavaan)
library(semTools)

# DATA ######################################################################################################################

DT <- data.table(read.table("PsychopathySuccess_dfFinal.txt", header=TRUE, quote="\""))

# PREPARE DATA ##############################################################################################################

### create new outcome variables #####
DT[,ObjectiveSuccess := OwnOffice + CarAccess + Budget + Employee + AnnSalary + PromFreq]
DT[,Satisfaction := CareerSa + PromSat + SalSat]

### standardizing variables #####
DT[,CareerSa_z := scale(DT$CareerSa, center = TRUE, scale = FALSE)]
DT[,PromSat_z := scale(DT$PromSat, center = TRUE, scale = FALSE)]
DT[,SalSat_z := scale(DT$SalSat, center = TRUE, scale = FALSE)]

DT[,SatisfactionZ := CareerSa_z + PromSat_z + SalSat_z]


### compute reliability for outcome variables ####
ObSucc <- data.frame(DT[,.(ProfStd, AnnSalary, PromFreq)])
alpha(ObSucc)

SubSucc <- data.frame(DT[,.(CareerSa, PromSat, SalSat)])
alpha(SubSucc)

### compute reliability for personality subscales ####
bf_Ex <- data.frame(DT[,.(bf_01,bf_06r)])
bf_Ag <- data.frame(DT[,.(bf_02r,bf_07)])
bf_Co <- data.frame(DT[,.(bf_03,bf_08r)])
bf_Em <- data.frame(DT[,.(bf_04r,bf_09)])
bf_Op <- data.frame(DT[,.(bf_05,bf_10r)])

alpha(bf_Op)

ppi_co_rel <- data.frame(DT[,.(ppi_r_40_07r,ppi_r_40_17r,ppi_r_40_24r,ppi_r_40_26r,ppi_r_40_39r)])
alpha(ppi_co_rel)

# Descriptives ####################################################################################################

stat.desc(DT$Whatisyourage, basic=F)
count(DT$Whatisyourgender)
describeBy(DT$Whatisyourage, DT$Whatisyourgender)
t.test(DT$Whatisyourage~DT$Whatisyourgender)

stat.desc(DT$MonthsInJob, basic=F)
describeBy(DT$MonthsInJob, DT$Whatisyourgender)
t.test(DT$MonthsInJob~DT$Whatisyourgender)

c <- count(DT$EmployeeLevel)
empCats<- c("Upper Management","Middle Management","Junior Management","Administrative","Support", "Trained Professional","Skilled Laborour","Consultant","Other","Selfemployed", "missing")
cbind( category = empCats, Freq=c$freq, Cumul=cumsum(c$freq), relative=prop.table(c$freq))
cbind(empCats, table(DT$EmployeeLevel,DT$Whatisyourgender),prop.table(table(DT$EmployeeLevel,DT$Whatisyourgender),2))
kruskal.test(DT$EmployeeLevel~DT$Whatisyourgender) 

c <- count(DT$Theorganisationyouworkforisinwhichofthefollowing)
typeCats<- c("public sector","private sector","large national ","NFO","dontknow","Other")
cbind( category = typeCats, Freq=c$freq, Cumul=cumsum(c$freq), relative=prop.table(c$freq))
cbind(typeCats, table(DT$Theorganisationyouworkforisinwhichofthefollowing,DT$Whatisyourgender),prop.table(table(DT$Theorganisationyouworkforisinwhichofthefollowing,DT$Whatisyourgender),2))
kruskal.test(DT$Theorganisationyouworkforisinwhichofthefollowing~DT$Whatisyourgender) 

c <- count(DT$Pleaseindicatethetypeoforganisationforwhichyouareemplo)
typeCats<- c("large international","small to medium size international","large national ","small to medium size national","localFamily","Other","dontknow")
cbind( category = typeCats, Freq=c$freq, Cumul=cumsum(c$freq), relative=prop.table(c$freq))

c <- count(DT$ProfCat)
profCats <- c("finance&retail","education","health&social", "software&tele","government&legal","Service&manu","Science", "other")
cbind( category = profCats, Freq=c$freq, Cumul=cumsum(c$freq), relative=prop.table(c$freq))
cbind(profCats, table(DT$ProfCat,DT$Whatisyourgender),prop.table(table(DT$ProfCat,DT$Whatisyourgender),2))
kruskal.test(DT$ProfCat~DT$Whatisyourgender) 

c <- count(DT$Whatisthesizeofyourorganisation)
sizeCats<- c("< 30 employees","30-50 employees","50-100 employees","100-500 employees","500-1000 employees","> 1000 employees","dontknow")
cbind( category = sizeCats, Freq=c$freq, Cumul=cumsum(c$freq), relative=prop.table(c$freq))
cbind(sizeCats, table(DT$Whatisthesizeofyourorganisation,DT$Whatisyourgender),prop.table(table(DT$Whatisthesizeofyourorganisation,DT$Whatisyourgender),2))
kruskal.test(DT$Whatisthesizeofyourorganisation~DT$Whatisyourgender) 


# Group differences ############################################################################################
## gender ###
stat.desc(DT$CareerSa, basic=F)
describeBy(DT$CareerSa, DT$Whatisyourgender)
t.test(DT$CareerSa~DT$Whatisyourgender)

stat.desc(DT$PromSat, basic=F)
describeBy(DT$PromSat, DT$Whatisyourgender)
t.test(DT$PromSat~DT$Whatisyourgender)

stat.desc(DT$SalSat, basic=F)
describeBy(DT$SalSat, DT$Whatisyourgender)
t.test(DT$SalSat~DT$Whatisyourgender)

stat.desc(DT$AnnSalL, basic=F)
describeBy(DT$AnnSalL, DT$Whatisyourgender)
t.test(DT$AnnSalL~DT$Whatisyourgender)

stat.desc(DT$ProfStd, basic=F)
describeBy(DT$ProfStd, DT$Whatisyourgender)
t.test(DT$ProfStd~DT$Whatisyourgender)

stat.desc(DT$Satisfaction, basic=F)
describeBy(DT$Satisfaction, DT$Whatisyourgender)
t.test(DT$Satisfaction~DT$Whatisyourgender)

stat.desc(DT$SatisfactionZ, basic=F)
describeBy(DT$SatisfactionZ, DT$Whatisyourgender)
t.test(DT$SatisfactionZ~DT$Whatisyourgender)

c <- count(DT$PromFreq)
promCats<- c("never","1x","2x","3x","4x", "5x",">5x","not applicable")
cbind( category = promCats, Freq=c$freq, Cumul=cumsum(c$freq), relative=prop.table(c$freq))
cbind(promCats, table(DT$PromFreq,DT$Whatisyourgender),prop.table(table(DT$PromFreq,DT$Whatisyourgender),2))
kruskal.test(DT$PromFreq~DT$Whatisyourgender) 

stat.desc(DT$ObjectiveSuccess, basic=F)
describeBy(DT$ObjectiveSuccess, DT$Whatisyourgender)
t.test(DT$ObjectiveSuccess~DT$Whatisyourgender)

stat.desc(DT$PPI_R_40_SUM, basic=F)
describeBy(DT$PPI_R_40_SUM, DT$Whatisyourgender)
t.test(DT$PPI_R_40_SUM~DT$Whatisyourgender)

stat.desc(DT$PPIR40FD, basic=F)
describeBy(DT$PPIR40FD, DT$Whatisyourgender)
t.test(DT$PPIR40FD~DT$Whatisyourgender)

stat.desc(DT$PPIR40SC, basic=F)
describeBy(DT$PPIR40SC, DT$Whatisyourgender)
t.test(DT$PPIR40SC~DT$Whatisyourgender)

stat.desc(DT$PPI_R_40_Co, basic=F)
describeBy(DT$PPI_R_40_Co, DT$Whatisyourgender)
t.test(DT$PPI_R_40_Co~DT$Whatisyourgender)

stat.desc(DT$bf_Ex, basic=F)
describeBy(DT$bf_Ex, DT$Whatisyourgender)
t.test(DT$bf_Ex~DT$Whatisyourgender)

stat.desc(DT$bf_Ag, basic=F)
describeBy(DT$bf_Ag, DT$Whatisyourgender)
t.test(DT$bf_Ag~DT$Whatisyourgender)

stat.desc(DT$bf_Co, basic=F)
describeBy(DT$bf_Co, DT$Whatisyourgender)
t.test(DT$bf_Co~DT$Whatisyourgender)

stat.desc(DT$bf_Em, basic=F)
describeBy(DT$bf_Em, DT$Whatisyourgender)
t.test(DT$bf_Em~DT$Whatisyourgender)

stat.desc(DT$bf_Op, basic=F)
describeBy(DT$bf_Op, DT$Whatisyourgender)
t.test(DT$bf_Op~DT$Whatisyourgender)

stat.desc(DT$IM, basic=F)
describeBy(DT$IM, DT$Whatisyourgender)
t.test(DT$IM~DT$Whatisyourgender)

## management position ###
stat.desc(DT$CareerSa, basic=F)
describeBy(DT$CareerSa, DT$PositionType4)
t.test(DT$CareerSa~DT$PositionType4)

stat.desc(DT$AnnSalL, basic=F)
describeBy(DT$AnnSalL, DT$PositionType4)
t.test(DT$AnnSalL~DT$PositionType4)

stat.desc(DT$ProfStd, basic=F)
describeBy(DT$ProfStd, DT$PositionType4)
t.test(DT$ProfStd~DT$PositionType4)

stat.desc(DT$PPIR40FD, basic=F)
describeBy(DT$PPIR40FD, DT$PositionType4)
t.test(DT$PPIR40FD~DT$PositionType4)

stat.desc(DT$PPIR40SC, basic=F)
describeBy(DT$PPIR40SC, DT$PositionType4)
t.test(DT$PPIR40SC~DT$PositionType4)

stat.desc(DT$IM, basic=F)
describeBy(DT$IM, DT$PositionType4)
t.test(DT$IM~DT$PositionType4)

stat.desc(DT$bf_Op, basic=F)
describeBy(DT$bf_Op, DT$PositionType4)
t.test(DT$bf_Op~DT$PositionType4)



# Correlations #####################################################################################################
dt <- data.frame(DT[,.(ProfStd,AnnSalary,PromFreq)])
corr.test(dt, adjust = "none")

dt <- data.frame(DT[,.(CareerSa, PromSat, SalSat)])
corr.test(dt, y = NULL, use = "pairwise",method="pearson",adjust="holm", alpha=.05,ci=F)

dt <- data.frame(DT[,.(ProfStd,AnnSalary,PromFreq)])
corr.test(dt, adjust = "none")

dt <- data.frame(DT[,.(ObjectiveSuccess,SubjectiveSuccess)])
corr.test(dt, adjust = "none")

dt <- data.frame(DT[,.(PPIR40FD,PPIR40SC,PPI_R_40_Co,bf_Ex,bf_Ag,bf_Co,bf_Em,bf_Op,IM)])
corr.test(dt, adjust = "none")

dt <- data.frame(DT[,.(ObjectiveSuccess,SubjectiveSuccess,PPIR40FD,PPIR40SC,PPI_R_40_Co,bf_Ex,bf_Ag,bf_Co,bf_Em,bf_Op,IM)])
corr.test(dt, adjust = "none")

dt <- data.frame(DT[,.(PPI_R_40_Co,ObjectiveSuccess,SubjectiveSuccess)])
corr.test(dt, adjust = "none")

dt <- data.frame(DT[,.(MonthsInJob,ObjectiveSuccess,SubjectiveSuccess)])
corr.test(dt, adjust = "none")



# SEM ##############################################################################################################

## Satisfaction =CareerSa + PromSat + SalSat

model_ProfSat_s <- '
Satisfaction =~ CareerSa_z + PromSat_z + SalSat_z 
Satisfaction ~ PPIR40SC + PPIR40FD + PPI_R_40_Co 
'

model_ProfSat_l <- '
Satisfaction =~ CareerSa_z + PromSat_z + SalSat_z 
Satisfaction ~ PPIR40SC + PPIR40FD + PPI_R_40_Co + bf_Ag + bf_Co + bf_Ex + bf_Em + bf_Op + Whatisyourgender + MonthsInJob
'

fit_ProfSat_s <- sem(model_ProfSat_s, data=DT) #std.ov=TRUE
summary(fit_ProfSat_s)
parTable(fit_ProfSat_s)
fitMeasures(fit_ProfSat_s)[c('chisq', 'df', 'pvalue', 'cfi', 'rmsea')]

fit_ProfSat_l <- sem(model_ProfSat_l, data=DT) #std.ov=TRUE
summary(fit_ProfSat_l)
parTable(fit_ProfSat_l)
fitMeasures(fit_ProfSat_l)[c('chisq', 'df', 'pvalue', 'cfi', 'rmsea')]

cbind(m1=inspect(fit_ProfSat_s, 'fit.measures'), m2=inspect(fit_ProfSat_l, 'fit.measures'))
anova(fit_ProfSat_s,fit_ProfSat_l)


## ObjectiveSuccess = AnnSalary + PromFreq + OwnOffice + CarAccess + Budget + Employee 

model_MatSucc_s <- '
ObjectiveSuccess =~ AnnSalary + PromFreq + OwnOffice + CarAccess + Budget + Employee  
ObjectiveSuccess ~ PPIR40SC + PPIR40FD + PPI_R_40_Co 
'

model_MatSucc_l <- '
ObjectiveSuccess =~ AnnSalary + PromFreq + OwnOffice + CarAccess + Budget + Employee 
ObjectiveSuccess ~ PPIR40SC + PPIR40FD + PPI_R_40_Co + bf_Ag + bf_Co + bf_Ex + bf_Em + bf_Op + Whatisyourgender + MonthsInJob
'

fit_MatSucc_s <- sem(model_MatSucc_s, data=DT) #std.ov=TRUE
summary(fit_MatSucc_s)
parTable(fit_MatSucc_s)
fitMeasures(fit_MatSucc_s)[c('chisq', 'df', 'pvalue', 'cfi', 'rmsea')]

fit_MatSucc_l <- sem(model_MatSucc_l, data=DT) #std.ov=TRUE
summary(fit_MatSucc_l)
parTable(fit_MatSucc_l)
fitMeasures(fit_MatSucc_l)[c('chisq', 'df', 'pvalue', 'cfi', 'rmsea')]

cbind(m1=inspect(fit_MatSucc_s, 'fit.measures'), m2=inspect(fit_MatSucc_l, 'fit.measures'))
anova(fit_MatSucc_s,fit_MatSucc_l)



# CFA for predictors ##############################################################################################

model_cfa_Satisfaction <- '
SatisfactionL =~  PromSat_z + SalSat_z +CareerSa_z
'

fit_cfa_sat <- cfa(model_cfa_Satisfaction, data=DT)
summary(fit_cfa_sat)
fitMeasures(fit_cfa_sat)

model_cfa_MaterialSuccess <- '
MaterialSuccess =~ AnnSalary + PromFreq + OwnOffice + CarAccess + Budget + Employee 
'
fit_cfa_matsuc <- cfa(model_cfa_MaterialSuccess, data=DT)
summary(fit_cfa_matsuc)
fitMeasures(fit_cfa_matsuc)
