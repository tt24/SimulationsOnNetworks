descr = read.table("../ExperimentRunner/src/experiment1.csv", header=T, sep=",")
data = read.table("../ExperimentRunner/src/experiment1_1.csv", header=F, sep = "," ,fill=T)

timesteps = numeric()
events = numeric()
s = numeric()
e = numeric()
i = numeric()
d = numeric()
dr = numeric()
r = numeric()
num = 1
file  <- file("../ExperimentRunner/src/experiment1_1.csv", open = "r")
while (length(oneLine <- readLines(file, n = 1)) > 0) {
	line <- as.numeric(unlist((strsplit(oneLine, ",")))[-1])
	print(line)
	if(num%%8==1) {
		timesteps = c(timesteps, line)
	}
	if(num%%8==2) {
		events = c(events, line)
	}
	if(num%%8==3) {
		s = c(s, line)
	}
	if(num%%8==4) {
		e = c(e, line)
	}
	if(num%%8==5) {
		i = c(i, line)
	}
	if(num%%8==6) {
		d = c(d, line)
	}
	if(num%%8==7) {
		r= c(r, line)
	}
	if(num%%8==0) {
		dr= c(dr, line)
	}
	num = num + 1
} 
close(file)

all = data.frame(timesteps[!is.na(timesteps)])
all$events = events[!is.na(events)]
all$s = s[!is.na(s)]
all$e =e[!is.na(e)]
all$i = i[!is.na(i)]
all$d = d[!is.na(d)]
all$r = r[!is.na(r)]
all$dr = dr[!is.na(dr)]

maxY = 0
for (i in ncol(all)) {
	if(max(all[,i])>maxY) {
		maxY = max(all[,i])
	}
}


#png(filename="SEIDR.png")
plot(all$d~all$timesteps, bg='black', pch=21, cex=0.5, xlim=c(0.0, max(all$timesteps)), 
ylim=c(0.0, maxY), xlab='timesteps', ylab='s,e,i,d,rec,rem')




par(new=T)
plot(all$s~all$timesteps, bg='yellow', pch=21, cex=0.5, xlim=c(0.0, max(all$timesteps)), 
ylim=c(0.0, maxY), xlab='', ylab='')
par(new=T)
plot(all$e~all$timesteps, bg='blue', pch=21, cex=0.5, xlim=c(0.0, max(all$timesteps)), 
ylim=c(0.0, maxY), xlab='', ylab='')
par(new=T)
plot(all$i~all$timesteps, bg='red', pch=21, cex=0.5, xlim=c(0.0, max(all$timesteps)), 
ylim=c(0.0, maxY), xlab='', ylab='')
par(new=T)
plot(all$r~all$timesteps, bg='green', pch=21, cex=0.5, xlim=c(0.0, max(all$timesteps)), 
ylim=c(0.0, maxY), xlab='', ylab='')
par(new=T)


plot(all$dr~all$timesteps, bg='grey', pch=21, cex=0.5, xlim=c(0.0, max(all$timesteps)), 
ylim=c(0.0, maxY), xlab='', ylab='')
#fit first degree polynomial equation:
timesteps = all$timesteps
fit  <- lm(all$dr~timesteps)
#second degree
fit2 <- lm(all$i~poly(timesteps,2,raw=TRUE))
#third degree
fit3 <- lm(all$dr~poly(timesteps,3,raw=TRUE))
#fourth degree
fit4 <- lm(all$dr~poly(timesteps,4,raw=TRUE))
x = (min(timesteps):max(timesteps))
lines(x, predict(fit2, data.frame(timesteps=x)), lwd=2, col='blue')
lines(x, predict(fit2, data.frame(timesteps=x)), lwd=2, col="green")
lines(x, predict(fit3, data.frame(timesteps=x)), lwd=2, col="blue")
lines(x, predict(fit4, data.frame(timesteps=x)), lwd=2, col="purple")

par(new=F)
#dev.off()


plot(r~timesteps)
plot(i~timesteps)
plot(e~timesteps)
plot(dr~timesteps)


library(splines)
splinemodel3 = lm(dr~bs(timesteps,knots = seq(min(timesteps, na.rm=T),max(timesteps, na.rm=T),40),degree=3))


splinemodel = lm(dr~bs(timesteps))
lines(min(timesteps, na.rm=T):max(timesteps, na.rm=T), predict(splinemodel3,
data.frame(timesteps=min(timesteps, na.rm=T):max(timesteps, na.rm=T))), lwd=2, col='red')

