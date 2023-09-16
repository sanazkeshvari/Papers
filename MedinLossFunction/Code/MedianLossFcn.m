function [Mu,Mur,MuIn]=MedianLossFcn(x)

alfa=1;Mu=mean(x);Landa=5;
Mur=mean(x);Gamma=0.1;Landar=0.2;
MuIn=mean(x);alfaIn=5;Epsilon=2;LandaIn=1;

for iter=1:1
    for i=1:size(x,1)
        Mu=Mu+Landa*tanh(alfa*(x(i)-Mu));
        Mur=Mur+Landar*tanh(alfa*(x(i)-Mur))*exp(-Gamma*log(cosh(x(i)-Mur)));
        %insensitive loss
        %lossIn=-1./(1+exp(alfaIn*(x(i)-MuIn+Epsilon)))+1./(1+exp(-alfaIn*(x(i)-MuIn-Epsilon)));
        lossIn= 0.5 *( tanh(alfaIn * (x(i)- MuIn + Epsilon)) + tanh(alfaIn * (x(i)- MuIn - Epsilon)));
        MuIn=MuIn+LandaIn*lossIn;
    end
end