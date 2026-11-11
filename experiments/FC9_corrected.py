# -*- coding: utf-8 -*-
"""
Created on Wednesday - 2023 12 April

@author: Ali Ghanbari Sorkhi
"""

from scipy import ndimage
import random
import numpy
from solution import solution
import time
import math
numLegs=5
def calculateCost(objf, population, popSize, lb, ub):
    scores = numpy.full(popSize, numpy.inf)
    for i in range(0, popSize):
        # Return back the search agents that go beyond the boundaries of the search space
        population[i] = numpy.clip(population[i], lb, ub)
        population[i, :]=numpy.nan_to_num(population[i, :],nan=lb[0])
        # Calculate objective function for each search agent
        scores[i] = objf(population[i, :])
        if numpy.isnan(scores[i]):
            scores[i] = objf(population[i, :])
    return scores
def mass_segmentation(scores,numLegs):
        
        
        out_leg_val=[]
        out_leg_Inx=[]
        out={'m_parts':[],'ind_parts':[],'numLeg':numLegs}
        
        sortedIndices = scores.argsort()
        scores = scores[sortedIndices]
        len_step=math.floor(scores.size/numLegs)
        for i in range(numLegs):
             out_leg_val.append(scores[i*len_step:(i+1)*len_step])
             out_leg_Inx.append(sortedIndices[i*len_step:(i+1)*len_step])
        out['m_parts']=out_leg_val
        out['ind_parts']=out_leg_Inx

        return out
def distance_ogh(mass_center,parts):
        dist = numpy.sqrt(numpy.sum((mass_center-parts)**2,axis=1))
        return dist
def massCenterCalc(Legs,M,D,lb,ub,minFitnessId):
    def calMass(D1,M1):
        
        CM = numpy.average(D1, axis=0, weights=numpy.max(M1)-M1+random.random())
        #CM = numpy.average(D1, axis=0, weights=numpy.arange(0,len(M1)))
        return CM
    LegsIndex=Legs['ind_parts']
    Legsfitness=Legs['m_parts']
    numLeg=Legs['numLeg']
    for i in range(numLeg):
        M[i,:]=calMass(D[LegsIndex[i],:],Legsfitness[i])
    return M        
   
def Legs_movement(Legs_new,Legs_velocity,MC,Legs,bestfitPop,lb,ub,minFitnessId,dim):
    LegsIndex=Legs['ind_parts']
    Legsfitness=Legs['m_parts']
    numLeg=Legs['numLeg']
    inertia=1
    correction_factor=5
    iterationcount=-5
    for i in range(numLeg):
        IndexcurrentPop=LegsIndex[i]
        ValuecurrentPop=Legs_new[IndexcurrentPop]
       
        for i1,valpop in enumerate(ValuecurrentPop):
            iterationcount=iterationcount+1
            if i==0 and i1==0:
                continue
            indexvel=IndexcurrentPop[i1]

            for j in range(valpop.shape[0]):
                factorMC=1
                r1 = random.random()
                r2 = random.random()
                MCweight=0
                for ii in range(numLeg):
                    if ii!=i:
                        MCweight=(MCweight+(ii-numLeg)*iterationcount/1000*(MC[ii,j]-valpop[j]))  # corrected bracket

                '''if i==0 :
                    factorMC=abs(MC[i,j]+MC[i+1,j])
                else:
                    factorMC=abs(MC[i-1,j]+MC[i,j])'''
                
                #w1=(bestfitPop[0,j]-valpop[j])/(bestfitPop[0,j]+valpop[j])
                w1 = 0.8 - 8*iterationcount/1000
                Legs_velocity[indexvel,j]=(w1)*(Legs_velocity[indexvel,j]
                                           +factorMC*correction_factor*r1*(MC[0,j]-valpop[j])
                                           +factorMC*inertia*r2*(bestfitPop[0,j]-valpop[j]))+0.3*MCweight
                
                Thre=dim/max(min(5,abs(ub[j]-lb[j])),3)
                if Legs_velocity[indexvel,j]>Thre:
                    Legs_velocity[indexvel,j]=Thre
                elif Legs_velocity[indexvel,j]<-Thre:
                     Legs_velocity[indexvel,j]=-Thre    
                Legs_new[indexvel, j]=Legs_new[indexvel,j]+Legs_velocity[indexvel,j]
                #Legs_new[indexvel, j] = numpy.clip(Legs_new[indexvel, j], lb[j], ub[j])
          
    return Legs_new,Legs_velocity



      
def FC(objf, lb, ub, dim, PopSize, iters):

    # FC parameters


    s = solution()
    if not isinstance(lb, list):
        lb = [lb] * dim
    if not isinstance(ub, list):
        ub = [ub] * dim

    ######################## Initializations
    
    pos = numpy.zeros((PopSize, dim))
    mass_center=numpy.zeros((numLegs, dim))
    bestfitPop=numpy.zeros((1, dim))
    for i in range(dim):
        pos[:, i] = numpy.random.uniform(0, 1, PopSize) * (ub[i] - lb[i]) + lb[i]
    for i in range(dim):
        mass_center[:, i] = numpy.random.uniform(0, 1, numLegs) * (ub[i] - lb[i]) + lb[i]
    convergence_curve = numpy.zeros(iters)
    fc_pos=pos[:]
    fc_vel=numpy.zeros((PopSize, dim))
    ############################################
    print('FC7 is optimizing  "' + objf.__name__ + '"')

    timerStart = time.time()
    s.startTime = time.strftime("%Y-%m-%d-%H-%M-%S")
    bestIndividual=[]
    firstIndividual=[]
    for l in range(0, iters):
        fitness_population = calculateCost(objf, fc_pos, PopSize, lb, ub)
        Legs=mass_segmentation(fitness_population,numLegs)
        minFitnessId =Legs['ind_parts'][0][0]
        bestScore=Legs['m_parts'][0][0]
        bestfitPop[0, :] = fc_pos[minFitnessId, :]
        mass_center = massCenterCalc(Legs,mass_center,fc_pos,lb,ub,minFitnessId)
        fc_pos,fc_vel=Legs_movement(fc_pos,fc_vel,mass_center,Legs,bestfitPop,lb,ub,minFitnessId,dim)
        
        convergence_curve[l] = bestScore
        bestIndividual.append( fc_pos[minFitnessId, :])
        firstIndividual.append( fc_pos[0, :])
        if l % 1 == 0:
            print(
                [
                    "FC7 At iteration "
                    + str(l + 1)
                    + " the best fitness is "
                    + str(bestScore)
                ]
            )
    timerEnd = time.time()
    s.endTime = time.strftime("%Y-%m-%d-%H-%M-%S")
    s.executionTime = timerEnd - timerStart
    s.convergence = convergence_curve
    s.bestIndividual
    s.bestIndividual=bestIndividual
    s.firstIndividual=firstIndividual
    s.optimizer = "FC"
    s.objfname = objf.__name__

    return s
