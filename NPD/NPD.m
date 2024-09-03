classdef NPD < ALGORITHM
% <single> <real/integer/label/binary/permutation> <large/none> <constrained/none>
%  AT---  0.1 --- Probability of crossover
% Nd --- 0.1 --- Distribution index of simulated binary crossover
% a ---  0.7 --- Expectation of the number of mutated variables
% d --- 0.1--- Distribution index of polynomial mutation

%------------------------------- Reference --------------------------------
% T. X. Wu, A brain-inspired neuron population dynamic optimization algorithm. MIT Press,
% 2023.
%------------------------------- Copyright --------------------------------
% Copyright (c) 2023 BIMK Group. You are free to use the PlatEMO for
% research purposes. All publications which use this platform or any code
% in the platform should acknowledge the use of "PlatEMO" and reference "Ye
% Tian, Ran Cheng, Xingyi Zhang, and Yaochu Jin, PlatEMO: A MATLAB platform
% for evolutionary multi-objective optimization [educational forum], IEEE
% Computational Intelligence Magazine, 2017, 12(4): 73-87".
%--------------------------------------------------------------------------

    methods
        function main(Algorithm,Problem)
            %% Parameter setting
            [AT,Nd,a,d] = Algorithm.ParameterSet(0.1,0.1,0.7,0.1);
            
            %% Generate random population
            Population = Problem.Initialization();
            
            
            %% Optimization
            while Algorithm.NotTerminated(Population)
                % 将最优个体视为吸引子，将附近的神经状态活动模式拉向稳定状态

                % 将多个最优个体视为吸引子集，将附近的神经状态活动模式不同程度的吸引
                [~,rank]   = sort(FitnessSingle(Population),"ascend");
                Population = Population(rank);
%                 [~,best]   = min(FitnessSingle(Population));
%                 Gbest      = Population(best);
                attractordec = Population(1:floor(end*AT));
                % 由神经群网络组成的动力学模型通常假设神经群之间的相互作用是通过加性耦合或扩散耦合进行的。
                % 当使用加性耦合时，种群的活动受到邻近种群活动总和的影响。相反，当使用扩散耦合时，神经种群受到其活动与其邻居活动之差之和的影响
                Offspring = Population.decs;
                T = Problem.FE/Problem.maxFE;
                for i = floor(Problem.N*AT):Problem.N
                    newone = Population(i).decs;
                    for j = 1:floor(Problem.N*AT)
                    % 随机创建该神经群与吸引子的邻接矩阵
%                         A = rand(1,Problem.D);
%                         A((A<T))=1;
%                         A((A~=1))=0;
                        A = randi([0,1] ,1, Problem.D);
                        SA = randi([1, floor(Problem.N*AT)]);
                        bestdec = attractordec(SA).decs;
    %                     A=randi([1,1],1,Problem.D);
                        newdec = Population(i).decs;
                        L = rand(1, Problem.D);
                        L1= rand(1, Problem.D);
                        newone = newone + a .*L.* L .*A .* (bestdec-newdec) ;
%                     newone = newdec;
                    end
                    % 高斯噪声
                    Lower = min(Population.decs);
                    Upper = max(Population.decs);
                    noiseMean = 0 * ones(1,Problem.D);
                    noiseStd = Nd*(Upper - Lower);
%                     noiseStd = Nd;
%                     mu = rand(1,1);
%                     if mu > 0.99
%                         noiseStd = noiseStd * 2;
%                     end
%                     noiseStd = noiseStd * (1-Problem.FE/Problem.maxFE) + 0.1;
                    noise = randn() * noiseStd + noiseMean;
                    newone = newone + noise;
                    % 随机创建神经群间的接矩阵
%                     Newdecs = ones(Problem.N, Problem.D);
%                     for j = 1:Problem.N
%                         Newdecs(j,:) = newdec;
%                     end
                     Offspring(i,:) = newone;
                end

                meanN = mean(Offspring);
                for i = 1:Problem.N
%                     addcontact = zeros(1, Problem.D);
%                     difcontact = zeros(1, Problem.D);
                    newone = Offspring(i,:);
%                     for j = 1:Problem.N
%                         tempdec = Offspring(j,:);
%                         A = randi([0,1] ,1, Problem.D);
% %                         A = randi([1,1],1,Problem.D);
%                         tempadd = L.* A.*(tempdec-newone);
% %                         tempdif =  A.*(tempdec);
% % 
%                         addcontact = addcontact + tempadd;
% %                         difcontact = difcontact + tempdif;
%                     end
                    
                    l1 = 1/2*d*rand(1, Problem.D);
                    l2 = 1/2*d*rand(1, Problem.D);
                    L= randi([-1,1] ,1, Problem.D);
                    L1= randi([-1,1] ,1, Problem.D);
                    L2= randi([-1,1] ,1, Problem.D);
                    A = randi([0,1] ,1, Problem.D);
                    A1 = randi([0,1] ,1, Problem.D);
                    A2 = randi([0,1] ,1, Problem.D);
%                     A1=A2;
%                     L1 = L2;
%                     A = rand(1,Problem.D);
%                     A((A<T))=1;
%                     A((A~=0))=1;
%                     newone = newone + L.*addcontact/Problem.N + L.*difcontact/Problem.N;
%                     newone = newone + L.*addcontact/Problem.N;
                    r = rand();
%                     if r >= 0.5
                    
                    newone = newone + (L1.*l1.* A1.* (newone-meanN)+L2.*A2.*l2.*meanN) * rand()*(1-T);

%                     newone = newone + (L1.*l1.* A.* (newone-meanN)) * rand()*(1-T);
%                     else
%                         newone = newone - L1.* L.*A.* (newone-meanN)* (1-T)*rand();
%                     end
                    
                    Offspring(i,:) = newone;
                end

                Offspring  = Problem.Evaluation(Offspring);

                Population = Offspring;
            end
        end
    end
end
